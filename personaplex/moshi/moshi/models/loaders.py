# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Retrieves the pretrained models for Moshi and Mimi."""
from pathlib import Path
import logging

from safetensors.torch import load_model, load_file
import torch

logger = logging.getLogger(__name__)

from .compression import MimiModel
from .lm import LMModel
from ..modules import SEANetEncoder, SEANetDecoder, transformer
from ..quantization import SplitResidualVectorQuantizer

SAMPLE_RATE = 24000
FRAME_RATE = 12.5

TEXT_TOKENIZER_NAME = 'tokenizer_spm_32k_3.model'
MOSHI_NAME = 'model.safetensors'
MIMI_NAME = 'tokenizer-e351c8d8-checkpoint125.safetensors'
DEFAULT_REPO = 'nvidia/personaplex-7b-v1'


_seanet_kwargs = {
    "channels": 1,
    "dimension": 512,
    "causal": True,
    "n_filters": 64,
    "n_residual_layers": 1,
    "activation": "ELU",
    "compress": 2,
    "dilation_base": 2,
    "disable_norm_outer_blocks": 0,
    "kernel_size": 7,
    "residual_kernel_size": 3,
    "last_kernel_size": 3,
    # We train using weight_norm but then the weights are pre-processed for inference so
    # that we can use a normal convolution.
    "norm": "none",
    "pad_mode": "constant",
    "ratios": [8, 6, 5, 4],
    "true_skip": True,
}
_quantizer_kwargs = {
    "dimension": 256,
    "n_q": 32,
    "bins": 2048,
    "input_dimension": _seanet_kwargs["dimension"],
    "output_dimension": _seanet_kwargs["dimension"],
}
_transformer_kwargs = {
    "d_model": _seanet_kwargs["dimension"],
    "num_heads": 8,
    "num_layers": 8,
    "causal": True,
    "layer_scale": 0.01,
    "context": 250,
    "conv_layout": True,
    "max_period": 10000,
    "gating": "none",
    "norm": "layer_norm",
    "positional_embedding": "rope",
    "dim_feedforward": 2048,
    "input_dimension": _seanet_kwargs["dimension"],
    "output_dimensions": [_seanet_kwargs["dimension"]],
}

_lm_kwargs = {
    "dim": 4096,
    "text_card": 32000,
    "existing_text_padding_id": 3,
    "n_q": 16,
    "dep_q": 8,
    "card": _quantizer_kwargs["bins"],
    "num_heads": 32,
    "num_layers": 32,
    "hidden_scale": 4.125,
    "causal": True,
    "layer_scale": None,
    "context": 3000,
    "max_period": 10000,
    "gating": "silu",
    "norm": "rms_norm_f32",
    "positional_embedding": "rope",
    "depformer_dim": 1024,
    "depformer_dim_feedforward": int(4.125 * 1024),
    "depformer_num_heads": 16,
    "depformer_num_layers": 6,
    "depformer_causal": True,
    "depformer_layer_scale": None,
    "depformer_multi_linear": True,
    "depformer_context": 8,
    "depformer_max_period": 10000,
    "depformer_gating": "silu",
    "depformer_pos_emb": "none",
    "depformer_weights_per_step": True,
    "delays": [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
}


def _is_safetensors(path: Path | str) -> bool:
    return Path(path).suffix in (".safetensors", ".sft", ".sfts")


def get_mimi(filename: str | Path,
             device: torch.device | str = 'cpu') -> MimiModel:
    """Return a pretrained Mimi model."""
    encoder = SEANetEncoder(**_seanet_kwargs)
    decoder = SEANetDecoder(**_seanet_kwargs)
    encoder_transformer = transformer.ProjectedTransformer(
        device=device, **_transformer_kwargs
    )
    decoder_transformer = transformer.ProjectedTransformer(
        device=device, **_transformer_kwargs
    )
    quantizer = SplitResidualVectorQuantizer(
        **_quantizer_kwargs,
    )
    model = MimiModel(
        encoder,
        decoder,
        quantizer,
        channels=1,
        sample_rate=SAMPLE_RATE,
        frame_rate=FRAME_RATE,
        encoder_frame_rate=SAMPLE_RATE / encoder.hop_length,
        causal=True,
        resample_method="conv",
        encoder_transformer=encoder_transformer,
        decoder_transformer=decoder_transformer,
    ).to(device=device)
    model.eval()
    if _is_safetensors(filename):
        load_model(model, filename)
    else:
        pkg = torch.load(filename, "cpu")
        model.load_state_dict(pkg["model"])
    model.set_num_codebooks(8)
    return model


def _replace_linear_with_bnb4(model: "LMModel", compute_dtype: torch.dtype) -> None:
    """Replace all nn.Linear layers in-place with bitsandbytes NF4 Linear4bit layers.

    This must be called *after* the model is fully constructed (so the architecture
    matches) but *before* load_state_dict, so that PyTorch assigns the packed uint8
    weights to the matching Linear4bit parameter rather than to a float tensor with
    a different shape.

    Args:
        model: The LMModel whose Linear layers will be converted.
        compute_dtype: The dtype used for the de-quantised matrix-multiply (usually bf16).
    """
    try:
        import bitsandbytes as bnb  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "bitsandbytes is required to load 4-bit quantised weights. "
            "Install it with:  pip install bitsandbytes"
        ) from exc

    # Walk the module tree and swap every nn.Linear for a matching Linear4bit.
    # We collect (parent, attr_name, module) first to avoid mutating while iterating.
    targets = []
    for name, module in model.named_modules():
        if type(module) is torch.nn.Linear:  # exact type, not subclass
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent_name, child_name = parts
                parent = model
                for part in parent_name.split("."):
                    parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
            else:
                parent = model
                child_name = name
            targets.append((parent, child_name, module))

    swapped = 0
    for parent, child_name, linear in targets:
        bnb_layer = bnb.nn.Linear4bit(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            compute_dtype=compute_dtype,
            compress_statistics=True,
            quant_type="nf4",
        )
        setattr(parent, child_name, bnb_layer)
        swapped += 1

    print(
        f"[loaders] replaced {swapped} nn.Linear → bnb.Linear4bit(nf4, "
        f"compute_dtype={compute_dtype})",
        flush=True,
    )


def get_moshi_lm(
    filename: str | Path | None,
    copy_missing_weights: bool = True,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    delays=None,
    cpu_offload: bool = False,
    quantize_4bit: bool = False,
) -> LMModel:
    """Return a pretrained Moshi LM model.

    Args:
        filename: Path to model weights.
        copy_missing_weights: Whether to copy missing weights from existing layers.
        device: Target device for the model.
        dtype: Data type for model weights.
        delays: Optional custom delays configuration.
        cpu_offload: If True, offload model layers to CPU when GPU memory is
                     insufficient. Uses accelerate's device_map="auto".
        quantize_4bit: If True, replace nn.Linear layers with bitsandbytes NF4
                       Linear4bit layers before loading the state-dict.  This is
                       required when ``filename`` was saved from a bnb-quantised
                       model (weights stored as uint8 [N*M//2, 1]).
    """
    # Copy to avoid mutating a shared/global dict
    lm_kwargs = dict(_lm_kwargs)
    lm_kwargs["dep_q"] = 16
    if delays is not None:
        lm_kwargs["delays"] = delays

    if cpu_offload and filename is not None:
        return _get_moshi_lm_with_offload(
            filename, copy_missing_weights, device, dtype, lm_kwargs
        )

    dev = torch.device(device) if isinstance(device, str) else device

    if quantize_4bit:
        # bnb-quantised checkpoint: weights are stored as uint8 [rows*cols//2, 1].
        # The model must be initialised with bnb.Linear4bit layers (not nn.Linear)
        # BEFORE load_state_dict is called, otherwise PyTorch raises a size mismatch.
        # We cannot use the "meta" device trick with bnb, so initialise on CPU first.
        model = LMModel(device="cpu", dtype=dtype, **lm_kwargs)
        if filename is None:
            model.eval()
            return model.to(device=device, dtype=dtype)
        _replace_linear_with_bnb4(model, compute_dtype=dtype)
    else:
        # Standard (non-quantised) path: init on meta device to skip dummy alloc.
        init_device = "meta" if filename is not None else device
        model = LMModel(device=init_device, dtype=dtype, **lm_kwargs)
        if filename is None:
            model.to(device=device, dtype=dtype)
            model.eval()
            return model

    filename = str(filename)

    # Load state_dict
    if filename.endswith(".safetensors"):
        # safetensors does not support mps directly
        if dev.type == "mps":
            state_dict = load_file(filename, device="cpu")
        else:
            state_dict = load_file(filename, device=dev.type)
    else:
        # torch checkpoint — always load to CPU first; bnb layers handle device placement
        with open(filename, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")

    # Patch 1: expand depformer self_attn weights if needed.
    # For 4-bit checkpoints the model_sd comes from bnb.Linear4bit which uses
    # a different weight shape, so only run this patch on non-quantised paths.
    model_sd = model.state_dict()
    if not quantize_4bit:
        for name, tensor in list(state_dict.items()):
            if "depformer" in name and "self_attn" in name and name in model_sd:
                if tensor.shape != model_sd[name].shape:
                    print("Expanding %s", name)
                    missing = (
                        tensor
                        if copy_missing_weights
                        else model_sd[name][tensor.shape[0] :]
                    )
                    state_dict[name] = torch.concat([tensor, missing], dim=0)

    # Patch 2: fill missing keys by copying 0..7 -> 8..15 for certain groups
    if copy_missing_weights:
        to_replace = ["gating", "linears", "depformer_in", "depformer_emb"]
        for name in model_sd.keys():
            if name in state_dict:
                continue
            replaced = False
            for old, new in zip(range(8), range(8, 16)):
                for rep in to_replace:
                    needle = f"{rep}.{new}."
                    if needle in name:
                        src = name.replace(needle, f"{rep}.{old}.")
                        if src in state_dict:
                            print("Replacing %s <- %s", name, src)
                            state_dict[name] = state_dict[src]
                            replaced = True
                        break
                if replaced:
                    break
            if not replaced:
                print("Missing %s", name)

    if quantize_4bit:
        # For bnb Linear4bit layers, do NOT cast the weight/quant_state tensors to
        # a float dtype: the weight stays uint8 and quant_state tensors must keep
        # their own dtypes.  Only cast tensors that are NOT part of a bnb layer.
        import bitsandbytes as bnb  # already imported above; kept for clarity

        bnb_weight_keys: set[str] = set()
        for name, module in model.named_modules():
            if isinstance(module, bnb.nn.Linear4bit):
                bnb_weight_keys.add(f"{name}.weight")

        for key in state_dict:
            base_key = key.split(".quant_state")[0]  # strip quant_state suffixes
            if base_key not in bnb_weight_keys:
                # Non-bnb tensor (embeddings, norms, biases …) — cast normally.
                try:
                    state_dict[key] = state_dict[key].to(device="cpu", dtype=dtype)
                except RuntimeError:
                    pass  # integer / uint8 tensors cannot be cast to bf16 — leave as-is
        # Load with assign=True so bnb.Linear4bit can take ownership of the uint8 weight.
        model.load_state_dict(state_dict, strict=False, assign=True)
        model.eval()
        # Move non-bnb parts (embeddings, norms …) to target device; bnb handles its own.
        model.to(device=device)
        return model
    else:
        # Assign weights to target device
        for key in state_dict:
            state_dict[key] = state_dict[key].to(device=dev, dtype=dtype)
        model.load_state_dict(state_dict, strict=False, assign=True)
        model.eval()
        return model.to(device=device, dtype=dtype)


def _get_moshi_lm_with_offload(
    filename: str | Path,
    copy_missing_weights: bool,
    device: torch.device | str,
    dtype: torch.dtype,
    lm_kwargs: dict,
) -> LMModel:
    """Load Moshi LM with CPU offloading using accelerate.

    This function distributes model layers across GPU and CPU based on
    available GPU memory. Layers that don't fit on GPU are kept on CPU
    and moved to GPU only during forward pass.
    """
    try:
        from accelerate import infer_auto_device_map, dispatch_model
    except ImportError:
        raise ImportError(
            "CPU offloading requires the 'accelerate' package. "
            "Install it with: pip install accelerate"
        )

    filename = str(filename)
    logger.info("Loading model with CPU offloading enabled")

    # First, create model on CPU to get the architecture
    model = LMModel(device="cpu", dtype=dtype, **lm_kwargs)

    # Load state_dict to CPU
    if filename.endswith(".safetensors"):
        state_dict = load_file(filename, device="cpu")
    else:
        with open(filename, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")

    # Apply weight patches (same as non-offload path)
    model_sd = model.state_dict()
    for name, tensor in list(state_dict.items()):
        if "depformer" in name and "self_attn" in name and name in model_sd:
            if tensor.shape != model_sd[name].shape:
                logger.info(f"Expanding {name}")
                missing = (
                    tensor
                    if copy_missing_weights
                    else model_sd[name][tensor.shape[0]:]
                )
                state_dict[name] = torch.concat([tensor, missing], dim=0)

    if copy_missing_weights:
        to_replace = ["gating", "linears", "depformer_in", "depformer_emb"]
        for name in model_sd.keys():
            if name in state_dict:
                continue
            replaced = False
            for old, new in zip(range(8), range(8, 16)):
                for rep in to_replace:
                    needle = f"{rep}.{new}."
                    if needle in name:
                        src = name.replace(needle, f"{rep}.{old}.")
                        if src in state_dict:
                            logger.info(f"Replacing {name} <- {src}")
                            state_dict[name] = state_dict[src]
                            replaced = True
                        break
                if replaced:
                    break
            if not replaced:
                logger.warning(f"Missing {name}")

    model.load_state_dict(state_dict, strict=False, assign=True)

    # Determine target device
    dev = torch.device(device) if isinstance(device, str) else device

    if dev.type != "cuda":
        # If not using CUDA, just move to the target device without offloading
        logger.info(f"CPU offload requested but device is {dev}, skipping offload")
        model.to(dev)
        model.eval()
        return model

    # Infer device map based on available GPU memory
    device_map = infer_auto_device_map(
        model,
        max_memory=None,  # Let accelerate auto-detect available memory
        no_split_module_classes=["StreamingTransformerLayer"],
        dtype=dtype,
    )

    # Log the device distribution
    gpu_layers = sum(1 for v in device_map.values() if v == 0 or v == "cuda:0")
    cpu_layers = sum(1 for v in device_map.values() if v == "cpu")
    logger.info(f"Device map: {gpu_layers} modules on GPU, {cpu_layers} modules on CPU")

    # Dispatch model across devices
    model = dispatch_model(
        model,
        device_map=device_map,
        offload_dir="offload_weights",  # Directory for disk offload if needed
    )

    model.eval()
    return model
