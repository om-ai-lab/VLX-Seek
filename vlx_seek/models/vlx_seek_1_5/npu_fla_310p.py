"""Optional 310P GDN inference via the vLLM AscendC ACLNN operator package.

The package path is provided by ASCEND_CUSTOM_OPP_PATH, for example by the
310P Docker image. The small bridge uses torch_npu's headers and does not
import vLLM or Triton.
Prefill keeps the torch WY preparation and uses AscendC state/output kernels.
"""

import ctypes
from functools import lru_cache
import logging
import os
from pathlib import Path
import re
import sys

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)
CHUNK_SIZE = 64


@lru_cache(maxsize=1)
def load_ascendc_310p():
    """Load once per process; torch's build lock coordinates concurrent TP ranks."""
    roots = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep)
    required = ("aclnnRecurrentGatedDeltaRuleV310", "aclnnChunkGatedDeltaRuleFwdHVllm", "aclnnChunkFwdOVllm")
    library = None
    for root in filter(None, roots):
        candidate = Path(root) / "op_api/lib/libcust_opapi.so"
        if not candidate.is_file():
            continue
        handle = ctypes.CDLL(str(candidate))
        if all(hasattr(handle, symbol) for symbol in required):
            library = candidate
            break
    if library is None:
        return False

    import torch_npu
    from torch.utils.cpp_extension import get_default_build_root, load

    npu = Path(torch_npu.__file__).parent
    toolkit = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest"))
    version = re.sub(r"\W", "_", f"py{sys.version_info.major}{sys.version_info.minor}_{torch.__version__}_{torch_npu.__version__}")
    cache_root = os.environ.get("TORCH_EXTENSIONS_DIR") or get_default_build_root()
    cache = Path(cache_root) / f"vlx_seek_310p_{version}"
    cache.mkdir(parents=True, exist_ok=True)
    logger.warning("Loading 310P AscendC bridge from %s; first build uses a C++ compiler. Cache: %s", library, cache)
    load(
        name="vlx_seek_310p_bridge",
        sources=[str(Path(__file__).with_name("csrc") / "ascendc_310p.cpp")],
        extra_include_paths=[str(npu / "include"), str(npu / "include/third_party/op-plugin"), str(toolkit / "include")],
        extra_cflags=["-O1"],
        extra_ldflags=[f"-L{npu / 'lib'}", "-ltorch_npu"],
        build_directory=str(cache), is_python_module=False, verbose=False,
    )
    return True


def supports_310p(query, key, value, g, beta, *, initial_state=None, prefill=False, output_final_state=False,
                  use_qk_l2norm_in_kernel=False, **kwargs):
    """Limit acceleration to validated dense FP16, 128-dimensional GDN calls."""
    if any(v is not None for v in kwargs.values()):
        return False
    if any(not isinstance(x, torch.Tensor) for x in (query, key, value, g, beta)):
        return False
    if query.ndim != 4 or key.shape != query.shape or value.shape != query.shape:
        return False
    batch, length, heads, dim = query.shape
    if batch < 1 or length < 1 or not (1 <= heads <= 256) or dim != 128:
        return False
    if not prefill and length != 1:
        return False
    if g.shape != query.shape[:3] or beta.shape != g.shape:
        return False
    if any(x.device != query.device for x in (key, value, g, beta)):
        return False
    if any(x.dtype != torch.float16 for x in (query, key, value, beta)):
        return False
    return initial_state is None or (
        initial_state.shape == (batch, heads, dim, dim)
        and initial_state.device == query.device
        and initial_state.dtype in (torch.float16, torch.float32)
    )


def _l2norm(x):
    # Match Transformers: normalization occurs in the input dtype.
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + 1e-6)


def recurrent_gated_delta_rule(query, key, value, g, beta, initial_state=None,
                              output_final_state=False, use_qk_l2norm_in_kernel=False, **kwargs):
    batch, _, heads, dim = query.shape
    if use_qk_l2norm_in_kernel:
        query, key = _l2norm(query), _l2norm(key)
    # ACLNN owns an FP16 [B,H,V,K] cache; never mutate the caller's [B,H,K,V].
    state = torch.zeros(batch, heads, dim, dim, dtype=query.dtype, device=query.device) if initial_state is None else initial_state.transpose(-1, -2).to(query.dtype).contiguous().clone()
    lengths = torch.ones(batch, dtype=torch.int32, device=query.device)
    indices = torch.arange(batch, dtype=torch.int32, device=query.device)
    out = torch.ops.vlx_seek_310p.recurrent(
        query.flatten(0, 1).contiguous(), key.flatten(0, 1).contiguous(),
        value.flatten(0, 1).contiguous(), beta.flatten(0, 1).contiguous(), state,
        lengths, indices, g.flatten(0, 1).float().contiguous(), dim ** -0.5,
    )
    final_state = state.transpose(-1, -2).contiguous().float() if output_final_state else None
    return out.reshape(value.shape), final_state


def chunk_gated_delta_rule(query, key, value, g, beta, initial_state=None,
                          output_final_state=False, use_qk_l2norm_in_kernel=False, **kwargs):
    if use_qk_l2norm_in_kernel:
        query, key = _l2norm(query), _l2norm(key)
    batch, length, heads, dim = query.shape
    pad = (-length) % CHUNK_SIZE
    q_kernel = F.pad(query.transpose(1, 2).contiguous(), (0, 0, 0, pad))
    k_kernel = F.pad(key.transpose(1, 2).contiguous(), (0, 0, 0, pad))
    k = k_kernel.float().reshape(batch, heads, -1, CHUNK_SIZE, dim)
    v = F.pad(value.transpose(1, 2).contiguous().float(), (0, 0, 0, pad)).reshape_as(k)
    gates = F.pad(g.transpose(1, 2).contiguous().float(), (0, pad)).reshape(batch, heads, -1, CHUNK_SIZE).cumsum(-1)
    betas = F.pad(beta.transpose(1, 2).contiguous().float(), (0, pad)).reshape_as(gates)
    decay = (gates.unsqueeze(-1) - gates.unsqueeze(-2)).tril().exp().tril()
    mask = torch.triu(torch.ones(CHUNK_SIZE, CHUNK_SIZE, dtype=torch.bool, device=query.device))
    k_beta = k * betas.unsqueeze(-1)
    attn = -(k_beta @ k.transpose(-1, -2) * decay).masked_fill(mask, 0)
    for i in range(1, CHUNK_SIZE):
        row, sub = attn[..., i, :i].clone(), attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(CHUNK_SIZE, dtype=attn.dtype, device=query.device)
    u = (attn @ (v * betas.unsqueeze(-1))).reshape(batch, heads, -1, dim).half().contiguous()
    w = (attn @ (k_beta * gates.exp().unsqueeze(-1))).reshape_as(u).half().contiguous()
    gates = gates.reshape(batch, heads, -1).contiguous()
    state = torch.zeros(batch, heads, dim, dim, dtype=torch.float32, device=query.device) if initial_state is None else initial_state.float().contiguous()
    h, v_new, final_state = torch.ops.vlx_seek_310p.chunk_h(k_kernel, w, u, gates, state)
    # The upstream 310P output scheduler corrupts earlier tiles on long/batched
    # inputs. Invoke it on one dense 64-token tile at a time with its matching
    # inter-chunk state. This avoids the faulty multi-task scheduling path.
    pieces = []
    for b in range(batch):
        chunks = []
        for c, start in enumerate(range(0, length + pad, CHUNK_SIZE)):
            tile = slice(start, start + CHUNK_SIZE)
            chunks.append(torch.ops.vlx_seek_310p.chunk_o(
                q_kernel[b:b+1, :, tile].contiguous(),
                k_kernel[b:b+1, :, tile].contiguous(),
                v_new[b:b+1, :, tile].contiguous(),
                h[b:b+1, :, c:c+1].contiguous(),
                gates[b:b+1, :, tile].contiguous(), dim ** -0.5,
            ))
        pieces.append(torch.cat(chunks, dim=2))
    out = torch.cat(pieces, dim=0)
    return out[:, :, :length].transpose(1, 2).contiguous(), final_state if output_final_state else None
