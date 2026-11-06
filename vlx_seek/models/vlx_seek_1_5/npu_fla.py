"""Optional FLA inference kernels for Qwen3.5 on Ascend.

Install an Ascend-compatible FLA/Triton pair in the runtime environment.
This adapter bypasses Transformers' CUDA-only FLA detection without changing
its package files or the CUDA model-loading path.
"""

import logging
import os
import time

import torch

from vlx_seek.hardware import is_ascend_310p

logger = logging.getLogger(__name__)
_ready_signatures = set()


class NPUFLARule:
    """Use FLA for NPU inference and retain the original training/CPU path."""

    def __init__(self, kernel, fallback, backend="Triton", supports=None):
        self.kernel = kernel
        self.fallback = fallback
        self.backend = backend
        self.supports = supports

    def __call__(
        self, query, key, value, g, beta, initial_state=None,
        output_final_state=False, use_qk_l2norm_in_kernel=False, **kwargs,
    ):
        options = dict(
            g=g, beta=beta, initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            **kwargs,
        )
        if (
            query.device.type != "npu"
            or query.dtype not in (torch.float16, torch.bfloat16)
            or torch.is_grad_enabled()
            or (self.supports is not None and not self.supports(query, key, value, **options))
        ):
            return self.fallback(query, key, value, **options)
        signature = (
            self.kernel, query.device, query.dtype, tuple(query.shape),
            initial_state is not None, output_final_state, use_qk_l2norm_in_kernel,
        )
        if signature in _ready_signatures:
            return self.kernel(query, key, value, **options)
        name = self.kernel.__name__
        logger.warning(
            "NPU FLA first call starting: %s, shape=%s. "
            "Backend: %s. Cache: %s",
            name, tuple(query.shape), self.backend,
            os.environ.get("TRITON_CACHE_DIR", "~/.triton/cache") if self.backend == "Triton" else "310P AscendC operator package",
        )
        started = time.perf_counter()
        result = self.kernel(query, key, value, **options)
        torch.npu.synchronize(query.device)
        _ready_signatures.add(signature)
        logger.warning(
            "NPU FLA first call completed: %s in %.2f seconds (backend=%s).",
            name, time.perf_counter() - started, self.backend,
        )
        return result


def enable_npu_fla(model):
    """Attach FLA kernels to NPU layers; return the number of enabled layers.

    VLX_SEEK_NPU_FLA=off disables the adapter for performance comparisons.
    The environment can also select decode-only or full (prefill + decode).
    """
    mode = os.environ.get("VLX_SEEK_NPU_FLA", "full").lower()
    if mode not in ("off", "decode", "full"):
        raise ValueError("VLX_SEEK_NPU_FLA must be off, decode, or full.")
    if mode == "off":
        for layer in model.modules():
            for name in ("chunk_gated_delta_rule", "recurrent_gated_delta_rule"):
                rule = getattr(layer, name, None)
                if isinstance(rule, NPUFLARule):
                    setattr(layer, name, rule.fallback)
        return 0

    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

    layers = [
        layer for layer in model.modules()
        if isinstance(layer, Qwen3_5GatedDeltaNet)
        and layer.in_proj_qkv.weight.device.type == "npu"
    ]
    if not layers:
        return 0

    if any(is_ascend_310p(layer.in_proj_qkv.weight.device) for layer in layers):
        from functools import partial
        from .npu_fla_310p import (
            chunk_gated_delta_rule as chunk_310p,
            recurrent_gated_delta_rule as recurrent_310p,
            load_ascendc_310p, supports_310p,
        )
        try:
            available = load_ascendc_310p()
        except (OSError, RuntimeError) as exc:
            logger.warning("310P AscendC bridge unavailable; using PyTorch linear attention: %s", exc)
            return 0
        if not available:
            logger.warning("310P AscendC GDN package not configured; using PyTorch linear attention. Set ASCEND_CUSTOM_OPP_PATH to the installed custom-op package before starting Python to enable acceleration.")
            return 0
        for layer in layers:
            if mode == "decode" and isinstance(layer.chunk_gated_delta_rule, NPUFLARule):
                layer.chunk_gated_delta_rule = layer.chunk_gated_delta_rule.fallback
            if mode == "full" and not isinstance(layer.chunk_gated_delta_rule, NPUFLARule):
                layer.chunk_gated_delta_rule = NPUFLARule(
                    chunk_310p, layer.chunk_gated_delta_rule, backend="AscendC",
                    supports=partial(supports_310p, prefill=True),
                )
            if not isinstance(layer.recurrent_gated_delta_rule, NPUFLARule):
                layer.recurrent_gated_delta_rule = NPUFLARule(
                    recurrent_310p, layer.recurrent_gated_delta_rule, backend="AscendC",
                    supports=supports_310p,
                )
        logger.warning(
            "NPU FLA enabled on %d linear-attention layers: backend=310P AscendC, prefill=%s, decode=AscendC. Causal convolution uses PyTorch; unsupported shapes use PyTorch.",
            len(layers), "PyTorch WY + AscendC" if mode == "full" else "PyTorch",
        )
        return len(layers)

    try:
        import triton
        from fla.ops.gated_delta_rule import (
            fused_recurrent_gated_delta_rule,
        )
    except ImportError as exc:
        logger.warning("NPU FLA unavailable; using PyTorch: %s", exc)
        return 0
    if triton.runtime.driver.active.get_current_target().backend != "npu":
        logger.warning("NPU FLA requires an Ascend Triton backend; using PyTorch.")
        return 0

    if mode == "full":
        from .npu_fla_output import chunk_gated_delta_rule

    for layer in layers:
        if mode == "decode" and isinstance(layer.chunk_gated_delta_rule, NPUFLARule):
            layer.chunk_gated_delta_rule = layer.chunk_gated_delta_rule.fallback
        if mode == "full" and not isinstance(layer.chunk_gated_delta_rule, NPUFLARule):
            layer.chunk_gated_delta_rule = NPUFLARule(
                chunk_gated_delta_rule, layer.chunk_gated_delta_rule
            )
        if not isinstance(layer.recurrent_gated_delta_rule, NPUFLARule):
            layer.recurrent_gated_delta_rule = NPUFLARule(
                fused_recurrent_gated_delta_rule, layer.recurrent_gated_delta_rule
            )
    logger.warning(
        "NPU FLA enabled on %d linear-attention layers: prefill=%s, decode=FLA. "
        "Causal convolution uses PyTorch.",
        len(layers), "FLA" if mode == "full" else "PyTorch",
    )
    return len(layers)
