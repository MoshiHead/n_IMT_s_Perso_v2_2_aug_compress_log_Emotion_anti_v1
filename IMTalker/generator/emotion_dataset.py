"""
Emotion-aware dataset for IMTalker emotion LoRA fine-tuning.

Drop-in replacement for generator/dataset.py's AudioMotionSmirkGazeDataset
that additionally:
  - infers the emotion label from the preprocessed file stem
  - returns an "emotion_id" key (long tensor) in every batch dict

Filename parsing rules (derived from user's MEAD dataset naming):
  neutral_XXXX.mp4          -> "neutral"  (index 0)
  M003_happy_level_1_001.mp4 -> "happy"   (index 1)
  M003_sad_level_2_018.mp4   -> "sad"     (index 2)
  M011_angry_level_3_030.mp4 -> "angry"   (index 3)
  M022_surprise_level_2_016  -> "surprise"(index 4)
  M009_fear_level_1_016.mp4  -> "fear"    (index 5)

The parser splits the stem on '_' and checks each token against EMOTION_MAP.
Unknown filenames default to neutral (0) with a printed warning.
"""
from __future__ import annotations

import random
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from generator.dataset import load_pose

# ── Emotion vocabulary ──────────────────────────────────────────────────────
EMOTION_MAP: dict[str, int] = {
    "neutral":   0,
    "neutrality":0,
    "happy":     1,
    "happiness": 1,
    "sad":       2,
    "sadness":   2,
    "angry":     3,
    "anger":     3,
    "surprise":  4,
    "surprised": 4,
    "fear":      5,
    "fearful":   5,
}
EMOTION_NAMES: dict[int, str] = {
    0: "neutral",
    1: "happy",
    2: "sad",
    3: "angry",
    4: "surprise",
    5: "fear",
}
NUM_EMOTIONS = 6          # real classes (0-5)
NULL_EMOTION_IDX = 6      # reserved for CFG dropout (dim_e-1 in 7-d one-hot)


def parse_emotion_from_stem(stem: str) -> int:
    """
    Infer emotion label from a preprocessed file stem.

    Splits on '_' and returns the EMOTION_MAP index of the first token that
    matches a known emotion name.  Falls back to 0 (neutral) with a warning.

    Examples
    --------
    >>> parse_emotion_from_stem("neutral_0001")          # -> 0
    >>> parse_emotion_from_stem("M003_happy_level_1_001") # -> 1
    >>> parse_emotion_from_stem("M009_fear_level_1_016")  # -> 5
    """
    tokens = stem.lower().split("_")
    for token in tokens:
        if token in EMOTION_MAP:
            return EMOTION_MAP[token]
    warnings.warn(
        f"[EmotionDataset] Could not infer emotion from stem '{stem}'; "
        "defaulting to neutral (0). Check EMOTION_MAP if this is unexpected.",
        stacklevel=2,
    )
    return 0  # default: neutral


# ── Dataset ─────────────────────────────────────────────────────────────────

class EmotionAudioMotionDataset(Dataset):
    """
    Emotion-aware dataset for emotion LoRA fine-tuning.

    Identical to AudioMotionSmirkGazeDataset but:
      1. Each sample dict contains an additional "emotion_id" key (long tensor
         scalar, value in 0..NUM_EMOTIONS-1).
      2. Can be filtered to only include specific emotion(s) via
         ``emotion_filter``.
      3. Prints a per-emotion breakdown during initialisation.

    Parameters
    ----------
    opt :
        The same options object used for AudioMotionSmirkGazeDataset.
    start, end :
        Slice of the sorted motion file list (same semantics as the parent).
    split :
        "train" or "val" (informational only; not used for filtering).
    list_path :
        Optional explicit list of file stems (one per line, relative to
        dataset_path).  Overrides start/end when provided.
    emotion_filter :
        If set, only samples whose inferred emotion is in this list are kept
        (e.g. ["happy", "sad"]).  None = keep all.
    """

    def __init__(
        self,
        opt,
        start: Optional[int] = None,
        end: Optional[int] = None,
        split: Optional[str] = None,
        list_path: Optional[str] = None,
        emotion_filter: Optional[list[str]] = None,
    ):
        super().__init__()
        self.opt = opt
        self.num_frames_for_clip = int(opt.wav2vec_sec * opt.fps)
        self.num_prev_frames = int(opt.num_prev_frames)
        self.required_len = self.num_frames_for_clip + self.num_prev_frames

        root_path = Path(opt.dataset_path)
        motion_dir  = root_path / "motion"
        audio_dir   = root_path / "audio"
        smirk_dir   = root_path / "smirk"
        gaze_dir    = root_path / "gaze"

        # Build file list (same logic as parent)
        if list_path and Path(list_path).exists():
            motion_files = []
            with open(list_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    stem = Path(line.strip()).stem
                    if not stem:
                        continue
                    mp = motion_dir / f"{stem}.pt"
                    if mp.exists():
                        motion_files.append(mp)
        else:
            motion_files = sorted(motion_dir.glob("*.pt"))
            if start is not None or end is not None:
                motion_files = motion_files[start:end]

        # Optional emotion filter
        _filter_ids: Optional[set[int]] = None
        if emotion_filter:
            _filter_ids = set()
            for name in emotion_filter:
                name_lc = name.strip().lower()
                if name_lc not in EMOTION_MAP:
                    raise ValueError(
                        f"Unknown emotion '{name}' in emotion_filter. "
                        f"Valid: {list(EMOTION_MAP.keys())}"
                    )
                _filter_ids.add(EMOTION_MAP[name_lc])

        # Scan valid samples
        self.samples: list[dict] = []
        emotion_counts: dict[int, int] = {i: 0 for i in range(NUM_EMOTIONS)}
        unknown_count = 0

        for motion_path in tqdm(motion_files, desc=f"[EmotionDataset/{split or 'all'}] scanning"):
            stem = motion_path.stem

            audio_path = audio_dir / f"{stem}.npy"
            gaze_path  = gaze_dir  / f"{stem}.npy"
            smirk_path = smirk_dir / f"{stem}.pt"

            if not (audio_path.exists() and gaze_path.exists() and smirk_path.exists()):
                continue

            eid = parse_emotion_from_stem(stem)
            if eid not in emotion_counts:
                unknown_count += 1
                eid = 0  # fallback

            if _filter_ids is not None and eid not in _filter_ids:
                continue

            try:
                motion = torch.load(str(motion_path), map_location="cpu")
                smirk  = torch.load(str(smirk_path),  map_location="cpu")
                audio  = np.load(str(audio_path), mmap_mode="r")
                gaze   = np.load(str(gaze_path),  mmap_mode="r")

                min_len = min(
                    len(motion),
                    len(audio),
                    len(gaze),
                    len(smirk["pose_params"]),
                )
                if min_len < self.required_len:
                    continue

                self.samples.append({
                    "motion_path": str(motion_path),
                    "audio_path":  str(audio_path),
                    "smirk_path":  str(smirk_path),
                    "gaze_path":   str(gaze_path),
                    "emotion_id":  eid,
                    "stem":        stem,
                })
                emotion_counts[eid] = emotion_counts.get(eid, 0) + 1

            except Exception as exc:
                print(f"[EmotionDataset] skip {stem}: {exc}")
                continue

        if not self.samples:
            raise RuntimeError(
                f"No valid samples found in {root_path}. "
                "Run preprocessing notebooks first."
            )

        print(f"[EmotionDataset/{split or 'all'}] {len(self.samples)} valid samples")
        for eid, cnt in sorted(emotion_counts.items()):
            if cnt > 0:
                print(f"  {EMOTION_NAMES[eid]:10s} (id={eid}): {cnt}")
        if unknown_count:
            print(f"  [unknown->neutral]: {unknown_count}")

        # Pre-cache a pool of neutral reference frames (m_ref)
        # When training on emotion clips (happy, sad, etc.), using a neutral m_ref forces
        # the model to learn to transform a neutral resting face into the target emotion.
        self.neutral_indices = [i for i, s in enumerate(self.samples) if s["emotion_id"] == 0]
        self.neutral_ref_pool = []
        if self.neutral_indices:
            n_pool_samples = min(50, len(self.neutral_indices))
            for s_idx in self.neutral_indices[:n_pool_samples]:
                try:
                    m_data = torch.load(self.samples[s_idx]["motion_path"], map_location="cpu")
                    if m_data.ndim == 2 and m_data.shape[0] > 0:
                        step = max(1, m_data.shape[0] // 5)
                        for f_idx in range(0, m_data.shape[0], step):
                            self.neutral_ref_pool.append(m_data[f_idx].clone().float())
                except Exception:
                    pass
            if self.neutral_ref_pool:
                self.neutral_ref_pool = torch.stack(self.neutral_ref_pool)  # [N_neutral, 32]
                print(f"[EmotionDataset/{split or 'all'}] Cached {len(self.neutral_ref_pool)} neutral reference frames for expression transformation")

    def __len__(self) -> int:
        return len(self.samples)

    def _get_full_clip(self, index: int):
        item = self.samples[index]

        motion  = torch.load(item["motion_path"], map_location="cpu")
        audio   = np.load(item["audio_path"], mmap_mode="r")
        gaze    = np.load(item["gaze_path"],  mmap_mode="r")
        smirk   = torch.load(item["smirk_path"], map_location="cpu")
        pose, cam = load_pose(smirk)

        min_len = min(len(motion), len(audio), len(gaze), len(pose))
        start_idx = random.randint(0, min_len - self.required_len)
        end_idx   = start_idx + self.required_len

        audio_seg  = torch.from_numpy(audio[start_idx:end_idx].copy()).float()
        motion_seg = motion[start_idx:end_idx]
        gaze_seg   = torch.from_numpy(gaze[start_idx:end_idx].copy()).float()
        pose_seg   = pose[start_idx:end_idx]
        cam_seg    = cam[start_idx:end_idx]

        np_ = self.num_prev_frames
        return (
            motion_seg[np_:],  audio_seg[np_:],  # clip
            motion_seg[:np_],  audio_seg[:np_],  # prev
            motion_seg,                           # full (for ref)
            gaze_seg[np_:],   gaze_seg[:np_],
            pose_seg[np_:],   pose_seg[:np_],
            cam_seg[np_:],    cam_seg[:np_],
        )

    def __getitem__(self, index: int) -> dict:
        try:
            (mc, ac, mp_, ap_, ms,
             gc, gp, pc, pp, cc, cp) = self._get_full_clip(index)
        except Exception as exc:
            print(f"[EmotionDataset] error idx={index}: {exc}; retrying random")
            return self.__getitem__(random.randint(0, len(self) - 1))

        emotion_id = self.samples[index]["emotion_id"]

        # Expression-morphing reference selection:
        # If the sample is an emotional clip (happy, sad, etc.), with 80% probability
        # pick m_ref from the NEUTRAL reference pool.
        # This breaks the shortcut where m_ref already contains the target emotion,
        # forcing the model to actively use emotion_id to morph the neutral face into
        # the requested emotion.
        if (
            emotion_id != 0
            and hasattr(self, "neutral_ref_pool")
            and isinstance(self.neutral_ref_pool, torch.Tensor)
            and len(self.neutral_ref_pool) > 0
            and random.random() < 0.8
        ):
            n_idx = random.randint(0, len(self.neutral_ref_pool) - 1)
            m_ref = self.neutral_ref_pool[n_idx]
        else:
            ref_idx = torch.randint(0, ms.shape[0], (1,)).item()
            m_ref = ms[ref_idx]

        return {
            # motion / audio
            "m_now":   mc,  "a_now":   ac,
            "m_prev":  mp_, "a_prev":  ap_,
            "m_ref":   m_ref,
            # gaze
            "gaze_now": gc, "gaze_prev": gp,
            # pose / cam (smirk)
            "pose_now": pc, "pose_prev": pp,
            "cam_now":  cc, "cam_prev":  cp,
            # emotion label
            "emotion_id": torch.tensor(emotion_id, dtype=torch.long),
        }
