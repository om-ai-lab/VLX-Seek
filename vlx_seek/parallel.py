"""Ascend tensor-parallel setup and CPU control messages for inference."""

import os
from datetime import timedelta

import torch
import torch.distributed as dist

_control_group = None


def initialize_tensor_parallel(device, tp_size=1):
    global _control_group

    if tp_size not in (1, 2):
        raise ValueError("tp_size must be 1 or 2.")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if tp_size == 1:
        if world_size != 1:
            raise ValueError("Pass --tp-size 2 when launching two processes with torchrun.")
        return device
    if device.type != "npu":
        raise ValueError("This tensor-parallel inference path requires Ascend NPUs.")
    if world_size != tp_size or "LOCAL_RANK" not in os.environ:
        raise ValueError(
            "TP=2 requires two processes: torchrun --standalone --nproc-per-node=2 "
            "inference.py --tp-size 2 ..."
        )
    local_rank = int(os.environ["LOCAL_RANK"])
    if device.index is not None and device.index != local_rank:
        raise ValueError("For TP, omit the NPU index; torchrun assigns one NPU per process.")
    if torch.npu.device_count() < tp_size:
        raise RuntimeError("TP=2 requires two visible Ascend NPUs.")
    torch.npu.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("hccl", timeout=timedelta(minutes=30))
    if dist.get_world_size() != tp_size or dist.get_backend() != "hccl":
        raise ValueError("TP=2 requires a two-rank HCCL process group.")
    if _control_group is None:
        # 310P collective support differs from 910B; keep Python objects on Gloo.
        _control_group = dist.new_group(backend="gloo", timeout=timedelta(minutes=30))
    return torch.device("npu", local_rank)


def is_main_process(tp_size=1):
    return tp_size == 1 or dist.get_rank() == 0


def run_on_rank_zero(function, tp_size=1):
    """Run CPU/control work once and propagate either its value or its error."""
    if tp_size == 1:
        return function()
    payload = [None]
    if dist.get_rank() == 0:
        try:
            payload[0] = (function(), None)
        except Exception as exc:
            payload[0] = (None, f"{type(exc).__name__}: {exc}")
    dist.broadcast_object_list(payload, src=0, group=_control_group)
    value, error = payload[0]
    if error is not None:
        raise RuntimeError(f"TP rank 0 failed: {error}")
    return value


def shutdown_tensor_parallel():
    global _control_group

    if dist.is_initialized():
        dist.destroy_process_group()
    _control_group = None
