"""Fix 06_emotion_live_pipeline.ipynb: use the moshi package bundled in the
brianmatzelle/personaplex-7b-v1-bnb-4bit snapshot (the only loader that can
load the pre-quantized model_bnb_4bit.pt checkpoint), instead of the repo's
own incompatible personaplex/moshi.

Root cause of the live-server crash:
  size mismatch for transformer.layers.N....: copying a param with shape
  torch.Size([numel//2, 1]) from checkpoint, the shape in current model is
  torch.Size([out, in]).
The checkpoint stores every quantized weight as flat uint8 [numel//2, 1]
(bitsandbytes NF4 packing). The repo's personaplex/moshi loaders swap in
fresh bnb.nn.Linear4bit layers (weight shape (out, in)) and call plain
load_state_dict, which can never accept those tensors. The snapshot's
bundled moshi reconstructs them via Params4bit.from_prequantized().
"""

import json
import shutil
from pathlib import Path

NB = Path(r"e:\LAB\codeV9\Analysis\n_IMT_s_Perso_v2_2_aug_compress_log_Emotion_anti_v1\06_emotion_live_pipeline.ipynb")
BAK = NB.parent / "scratch" / (NB.name + ".bak")

nb = json.loads(NB.read_text(encoding="utf-8"))
cells = nb["cells"]


def src_lines(cell):
    return [l[:-1] if l.endswith("\n") else l for l in cell.get("source", [])]


def find_cell(marker):
    hits = [
        i for i, c in enumerate(cells)
        if c.get("cell_type") == "code" and marker in "".join(c.get("source", []))
    ]
    assert hits, f"no code cell contains marker: {marker!r}"
    return hits[0]


def replace_block(cell, old_block, new_block, desc):
    old_lines = old_block.strip("\n").split("\n")
    new_lines = new_block.strip("\n").split("\n")
    src = src_lines(cell)
    for i in range(len(src) - len(old_lines) + 1):
        if src[i:i + len(old_lines)] == old_lines:
            src[i:i + len(old_lines)] = new_lines
            cell["source"] = [l + "\n" for l in src]
            print(f"[ok] {desc}: replaced {len(old_lines)} lines with {len(new_lines)} lines")
            return
    raise AssertionError(
        f"block not found ({desc}). First old lines:\n" + "\n".join(old_lines[:6])
    )


# ---------------------------------------------------------------------------
# EDIT 1 — cell 2 (env setup, step 9): stop installing the repo's own
#          personaplex/moshi as a placeholder; bundled-snapshot copies only.
# ---------------------------------------------------------------------------
c2 = cells[find_cell("# 9. Install PersonaPlex moshi package if present")]

replace_block(
    c2,
    '''# 9. Install PersonaPlex moshi package if present
moshi_candidates = [
    Path(PERSONAPLEX_DIR) / "moshi",
    Path(CHECKPOINT_DIR) / "personaplex_bnb4" / "moshi",
    Path(PROJECT_ROOT) / "personaplex" / "moshi",
    Path("/workspace/speech2avatar/personaplex/moshi"),
    Path("/workspace/personaplex/moshi"),
]
moshi_dir = next((d for d in moshi_candidates if d.exists()), None)
if moshi_dir:
    print(f"[install] installing bundled Moshi from {moshi_dir}...")
    run(f"{VENV_ACTIVATE} && pip install --quiet --no-deps -e {moshi_dir}", check=False)''',
    '''# 9. Install PersonaPlex moshi package -- ONLY the copy bundled in the bnb4
#    weights snapshot (it ships the fix for loading the pre-quantized
#    model_bnb_4bit.pt checkpoint). The repo's own personaplex/moshi is
#    intentionally NOT used: it cannot load that checkpoint (every quantized
#    weight is stored as flat uint8 [numel//2, 1] and its loader raises
#    size-mismatch errors). On a fresh pod the snapshot is not downloaded
#    yet -- cell 4 downloads + installs it.
moshi_candidates = [
    Path(PERSONAPLEX_DIR) / "moshi",
    Path(CHECKPOINT_DIR) / "personaplex_bnb4" / "moshi",
]
moshi_dir = next((d for d in moshi_candidates if d.exists()), None)
if moshi_dir:
    print(f"[install] installing bundled Moshi from {moshi_dir}...")
    run(f"{VENV_ACTIVATE} && pip install --quiet --no-deps -e {moshi_dir}", check=False)
else:
    print("[info] bundled Moshi not present yet -- cell 4 downloads + installs it from the bnb4 snapshot")''',
    "cell 2 step 9 (moshi install candidates)",
)

# ---------------------------------------------------------------------------
# EDIT 2 — cell 4 (download): also fetch the bundled moshi/ package from the
#          bnb4 snapshot, replace any stale venv install, and verify the
#          loader supports pre-quantized 4-bit checkpoints.
# ---------------------------------------------------------------------------
c4 = cells[find_cell("# --- 4. PersonaPlex 4-bit weights & Mimi tokenizer")]

replace_block(
    c4,
    '''# Install PersonaPlex Moshi package
moshi_candidates = [
    os.path.join(PERSONAPLEX_DIR, "moshi"),
    os.path.join(CHECKPOINT_DIR, "personaplex_bnb4", "moshi"),
    os.path.join(PROJECT_ROOT, "personaplex", "moshi"),
    "/workspace/speech2avatar/personaplex/moshi",
    "/workspace/personaplex/moshi",
]
moshi_dir = next((d for d in moshi_candidates if os.path.isdir(d)), None)
if moshi_dir:
    print(f"[install] installing bundled Moshi from {moshi_dir}...")
    run(f"{VENV_ACTIVATE} && pip install --quiet -e {moshi_dir} --no-deps", retries=2, check=False)''',
    '''# --- 4b. Bundled Moshi package from the bnb4 snapshot (REQUIRED) ---
# model_bnb_4bit.pt is a PRE-QUANTIZED bitsandbytes NF4 checkpoint: every
# quantized weight is stored as a flat uint8 [numel//2, 1] tensor plus
# quant_state.bitsandbytes__* metadata. Only the moshi package bundled in
# this snapshot can load it (its loaders reconstruct the weights via
# bnb.nn.Params4bit.from_prequantized). The repo's own personaplex/moshi is
# an incompatible older version: its loader raises size-mismatch errors on
# these tensors and its attention keys (self_attn.in_proj_weight) do not
# match this checkpoint. The live server resolves "import moshi" from
# $PERSONAPLEX_DIR/moshi (--moshi_root + PYTHONPATH in start_winner_live.sh),
# so this directory MUST exist.
BUNDLED_MOSHI_DIR = os.path.join(PERSONAPLEX_DIR, "moshi")
BUNDLED_MOSHI_LOADERS = os.path.join(BUNDLED_MOSHI_DIR, "moshi", "models", "loaders.py")
if os.path.isfile(BUNDLED_MOSHI_LOADERS):
    print("[skip] bundled Moshi package already present")
else:
    print("[4b/7] downloading bundled Moshi package (pre-quantized loader)...")
    run(
        f"{VENV_ACTIVATE} && hf download {HF_PERSONAPLEX_REPO} "
        f"--include 'moshi/*' --local-dir '{PERSONAPLEX_DIR}'",
        retries=2,
    )
if not os.path.isfile(BUNDLED_MOSHI_LOADERS):
    raise RuntimeError(
        f"Bundled Moshi package missing after download: {BUNDLED_MOSHI_LOADERS}. "
        "It is required to load model_bnb_4bit.pt."
    )

# Install the bundled Moshi package. Uninstall first: an earlier run may
# have installed the repo's own (incompatible) personaplex/moshi -- both
# dists are named 'moshi-personaplex', so this cleanly replaces it.
run(f"{VENV_ACTIVATE} && pip uninstall -y moshi-personaplex", check=False, quiet=True)
print(f"[install] installing bundled Moshi from {BUNDLED_MOSHI_DIR}...")
run(f"{VENV_ACTIVATE} && pip install --quiet --no-deps -e {BUNDLED_MOSHI_DIR}", retries=2, check=False)

# Verify the resolved loaders supports pre-quantized 4-bit checkpoints
moshi_probe = (
    "import sys; sys.path.insert(0, sys.argv[1]); "
    "from moshi.models import loaders; "
    "print('OK' if hasattr(loaders, '_is_prequantized_state_dict') else 'MISSING')"
)
res = subprocess.run([VENV_PYTHON, "-c", moshi_probe, BUNDLED_MOSHI_DIR], capture_output=True, text=True)
if "OK" not in res.stdout:
    raise RuntimeError(
        "Resolved moshi loaders.py does NOT support pre-quantized 4-bit checkpoints "
        f"(probe: {res.stdout.strip()} {res.stderr.strip()[:300]}). "
        "model_bnb_4bit.pt cannot be loaded without it -- aborting here instead "
        "of failing later with a cryptic size-mismatch inside the live server."
    )
print("[ok] bundled Moshi supports pre-quantized 4-bit checkpoints")''',
    "cell 4 section 4b (bundled moshi download + install + verify)",
)

# ---------------------------------------------------------------------------
# EDIT 3 — cell 8 (launch): hard pre-launch guard that the runtime-resolved
#          moshi loader supports the pre-quantized checkpoint.
# ---------------------------------------------------------------------------
c8 = cells[find_cell("# Ensure PersonaPlex moshi package is installed")]

replace_block(
    c8,
    '''# Ensure PersonaPlex moshi package is installed
chk_moshi = subprocess.run([VENV_PYTHON, "-c", "import moshi"], capture_output=True, text=True)
if chk_moshi.returncode != 0:
    moshi_candidates = [
        Path(PERSONAPLEX_DIR) / "moshi",
        Path(CHECKPOINT_DIR) / "personaplex_bnb4" / "moshi",
        Path(PROJECT_ROOT) / "personaplex" / "moshi",
        Path("/workspace/speech2avatar/personaplex/moshi"),
        Path("/workspace/personaplex/moshi"),
    ]
    moshi_dir = next((d for d in moshi_candidates if d.exists()), None)
    if moshi_dir:
        print(f"[fix] installing bundled Moshi from {moshi_dir}...")
        run(f"{VENV_ACTIVATE} && pip install --quiet --no-deps -e {moshi_dir}", check=False)''',
    '''# Hard guard: the moshi package the server will ACTUALLY resolve must
# support the pre-quantized 4-bit checkpoint. start_winner_live.sh sets
# --moshi_root $PERSONAPLEX_DIR and PYTHONPATH=$PERSONAPLEX_DIR/moshi, and
# liveTry.py inserts $PERSONAPLEX_DIR/moshi at sys.path[0] -- so the server
# imports moshi from that bundled copy regardless of what is installed in the
# venv. If it is missing, model_bnb_4bit.pt fails to load with the cryptic
# "size mismatch ... [numel//2, 1] vs [out, in]" error; fail here with a
# clear message instead.
_bundled_moshi_loaders = Path(PERSONAPLEX_DIR) / "moshi" / "moshi" / "models" / "loaders.py"
if not _bundled_moshi_loaders.is_file():
    raise RuntimeError(
        f"Bundled Moshi package missing at {_bundled_moshi_loaders}. "
        "Re-run cell 4 (Download runtime assets) -- it downloads the moshi package "
        "bundled in the personaplex bnb4 snapshot, which is REQUIRED to load "
        "model_bnb_4bit.pt (the repo's own personaplex/moshi cannot)."
    )
if "_is_prequantized_state_dict" not in _bundled_moshi_loaders.read_text():
    raise RuntimeError(
        f"Bundled Moshi at {_bundled_moshi_loaders} lacks pre-quantized 4-bit support "
        "(_is_prequantized_state_dict not found). Re-run cell 4 to re-download it."
    )
print("[ok] runtime moshi loader supports pre-quantized 4-bit checkpoints")

# Ensure PersonaPlex moshi package is also installed in the venv (side tooling)
chk_moshi = subprocess.run([VENV_PYTHON, "-c", "import moshi"], capture_output=True, text=True)
if chk_moshi.returncode != 0:
    _moshi_dir = Path(PERSONAPLEX_DIR) / "moshi"
    print(f"[fix] installing bundled Moshi from {_moshi_dir}...")
    run(f"{VENV_ACTIVATE} && pip install --quiet --no-deps -e {_moshi_dir}", check=False)''',
    "cell 8 (pre-launch bundled-moshi guard)",
)

# ---------------------------------------------------------------------------
# EDIT 4 — cell 8 (launch): pin PERSONAPLEX_DIR in the launch environment so
#          start_winner_live.sh resolves the bundled moshi + weights from the
#          notebook's own checkpoint dir, not a stale /workspace fallback.
# ---------------------------------------------------------------------------
replace_block(
    c8,
    '''    "PORT":                str(PORT),
    "SPEECH2AVATAR_ROOT":  PROJECT_ROOT,
    "IMTALKER_DIR":        IMTALKER_DIR,
    "VENV_DIR":            VENV_DIR,
    "LORA_GENERATOR_PATH": LORA_CKPT_LOCAL,''',
    '''    "PORT":                str(PORT),
    "SPEECH2AVATAR_ROOT":  PROJECT_ROOT,
    "IMTALKER_DIR":        IMTALKER_DIR,
    "VENV_DIR":            VENV_DIR,
    "LORA_GENERATOR_PATH": LORA_CKPT_LOCAL,
    # Pin the bnb4 snapshot dir so start_winner_live.sh always resolves the
    # bundled moshi/ package + model_bnb_4bit.pt from THIS dir (not a stale
    # /workspace/personaplex_bnb4 left over from another setup).
    "PERSONAPLEX_DIR":     PERSONAPLEX_DIR,''',
    "cell 8 (launch_env PERSONAPLEX_DIR pin)",
)

# ---------------------------------------------------------------------------
# Write back (backup first), preserving the notebook's indent=1 formatting.
# ---------------------------------------------------------------------------
BAK.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(NB, BAK)
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"[ok] backup written to {BAK}")
print(f"[ok] notebook updated: {NB}")
