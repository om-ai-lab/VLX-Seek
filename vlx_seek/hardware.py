"""Central hardware selection for the CLI and Python inference API."""

import importlib
import os

import torch


_BACKENDS = {
    "auto": "auto", "nvidia": "cuda", "cuda": "cuda",
    "ascend": "npu", "910b": "npu", "310p": "npu", "npu": "npu",
    "cpu": "cpu", "apple": "mps", "mps": "mps",
}


def _npu_available(required=False):
    try:
        importlib.import_module("torch_npu")
        torch.npu.set_compile_mode(jit_compile=False)
    except ImportError as exc:
        if required:
            raise RuntimeError(
                "Ascend backend requires torch_npu. Use a matching Ascend Docker image "
                "or install a torch_npu build matching PyTorch."
            ) from exc
        return False
    return hasattr(torch, "npu") and torch.npu.is_available()


def resolve_device(device=None):
    """Resolve --device/API overrides, otherwise VLX_SEEK_BACKEND (auto).

    Auto prefers CUDA, then Ascend, then Apple/MPS, then CPU.
    """
    if device is None:
        backend = os.environ.get("VLX_SEEK_BACKEND", "auto").strip().lower()
        if backend not in _BACKENDS:
            raise ValueError(
                "VLX_SEEK_BACKEND must be auto, nvidia/cuda, ascend/npu/910b/310p, "
                "cpu, or apple/mps."
            )
        device = _BACKENDS[backend]
    else:
        device = str(device).strip().lower()
        device = _BACKENDS.get(device, device)

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif _npu_available():
            device = "npu"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    kind = device.split(":", 1)[0]
    if kind == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise RuntimeError("Apple backend selected, but MPS is unavailable.")
    elif kind == "npu":
        if not _npu_available(required=True):
            raise RuntimeError("Ascend backend selected, but no NPU is available.")
    elif kind == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("NVIDIA backend selected, but CUDA is unavailable.")
    elif kind != "cpu":
        raise ValueError(f"Unsupported inference device: {device}")
    return torch.device(device)


def is_ascend_310p(device):
    """Identify chips without the FlashAttentionScore/910B FLA kernels."""
    return device.type == "npu" and "310p" in torch.npu.get_device_name(device).lower()


def attention_implementation(device):
    """Use portable attention on 310P, CPU and MPS."""
    if is_ascend_310p(device):
        return "sdpa"
    return "flash_attention_2" if device.type in ("cuda", "npu") else "sdpa"


def inference_dtype(device):
    """Use FP16 on MPS and Ascend chips without BF16 support."""
    if device.type == "mps":
        return torch.float16
    if device.type == "npu":
        with torch.npu.device(device):
            return torch.bfloat16 if torch.npu.is_bf16_supported() else torch.float16
    return torch.bfloat16
