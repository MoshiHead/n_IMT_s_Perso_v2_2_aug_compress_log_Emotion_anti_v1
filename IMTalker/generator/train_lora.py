import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base, rank=16, alpha=32, dropout=0.05):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(base)}")
        self.base = base
        for param in self.base.parameters():
            param.requires_grad = False
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / max(1, self.rank)
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.lora_A = nn.Linear(base.in_features, self.rank, bias=False)
        self.lora_B = nn.Linear(self.rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scale


def _get_child(root, path):
    obj = root
    for part in path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def _set_child(root, path, value):
    parts = path.split(".")
    parent = _get_child(root, ".".join(parts[:-1])) if len(parts) > 1 else root
    key = parts[-1]
    if key.isdigit():
        parent[int(key)] = value
    else:
        setattr(parent, key, value)


def lora_target_names(model, include_pose_lora=True, include_audio_lora=True, only_pose_lora=False):
    names = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if only_pose_lora:
            if name == "pose_projection.0":
                names.append(name)
            continue
        tier1 = name.startswith("fmt.blocks.") and (
            name.endswith(".attn.qkv")
            or name.endswith(".attn.proj")
            or name.endswith(".mlp.fc1")
            or name.endswith(".mlp.fc2")
        )
        tier2 = name.startswith("fmt.blocks.") and name.endswith(".adaLN_modulation.1")
        tier3_names = {"gaze_projection.0", "cam_projection.0"}
        if include_audio_lora:
            tier3_names.add("audio_projection.0")
        tier3 = name in tier3_names or (include_pose_lora and name == "pose_projection.0")
        tier4 = name in {
            "fmt.decoder.linear",
            "fmt.decoder.adaLN_modulation.1",
        }
        if tier1 or tier2 or tier3 or tier4:
            names.append(name)
    return names


def apply_lora_to_model(
    model,
    rank=16,
    alpha=None,
    dropout=0.05,
    include_pose_lora=True,
    include_audio_lora=True,
    only_pose_lora=False,
):
    if alpha is None or float(alpha) <= 0:
        alpha = 2 * int(rank)
    for param in model.parameters():
        param.requires_grad = False
    targets = lora_target_names(
        model,
        include_pose_lora=include_pose_lora,
        include_audio_lora=include_audio_lora,
        only_pose_lora=only_pose_lora,
    )
    for name in targets:
        base = _get_child(model, name)
        _set_child(model, name, LoRALinear(base, rank=rank, alpha=alpha, dropout=dropout))
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    print(
        f"[LoRA] targets={len(targets)} rank={rank} alpha={alpha} dropout={dropout} "
        f"include_pose_lora={include_pose_lora} include_audio_lora={include_audio_lora} "
        f"only_pose_lora={only_pose_lora}",
        flush=True,
    )
    print(f"[LoRA] trainable={trainable:,} total={total:,} ratio={trainable / max(1, total):.6f}", flush=True)
    return targets


def apply_emotion_lora_to_model(
    model,
    rank: int = 32,
    alpha: float = None,
    dropout: float = 0.05,
):
    """
    Emotion-specific LoRA fine-tuning setup.

    Freezes ALL base model parameters, then:
    1. Unfreezes the emotion_conditioner entirely (new, small module).
    2. Applies LoRA to: fmt.c_embedder, all fmt.blocks.*.adaLN_modulation.1,
       fmt.decoder.linear, fmt.decoder.adaLN_modulation.1.

    These are exactly the layers that control how conditioning signals
    (now including emotion) modulate each transformer block.
    Audio/pose/gaze/cam projections are frozen to preserve lip-sync quality.
    """
    if alpha is None or float(alpha) <= 0:
        alpha = float(2 * int(rank))

    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze emotion_conditioner (new module, not in pretrained ckpt)
    for param in model.emotion_conditioner.parameters():
        param.requires_grad = True

    # Collect LoRA targets
    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        is_adaLN = (
            name.startswith("fmt.blocks.")
            and name.endswith(".adaLN_modulation.1")
        )
        is_decoder = name in {
            "fmt.decoder.linear",
            "fmt.decoder.adaLN_modulation.1",
        }
        is_c_embedder = name == "fmt.c_embedder"
        if is_adaLN or is_decoder or is_c_embedder:
            targets.append(name)

    for name in targets:
        base = _get_child(model, name)
        _set_child(
            model, name,
            LoRALinear(base, rank=int(rank), alpha=float(alpha), dropout=float(dropout))
        )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"[EmotionLoRA] targets={len(targets)} rank={rank} alpha={alpha} dropout={dropout}",
        flush=True,
    )
    print(
        f"[EmotionLoRA] trainable={trainable:,} total={total:,} "
        f"ratio={trainable / max(1, total):.6f}",
        flush=True,
    )
    return targets
