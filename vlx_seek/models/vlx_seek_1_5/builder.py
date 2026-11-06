from transformers import AutoTokenizer
from vlx_seek.hardware import attention_implementation, inference_dtype, resolve_device
from vlx_seek.parallel import initialize_tensor_parallel


def load_pretrained_model(model_path, device=None, dtype=None, tp_size=1):
    """Load a VLX-Seek model and its primary/auxiliary image processors.

    Args:
        model_path: Local checkpoint directory or Hugging Face model identifier.
        device: Explicit device override; otherwise use VLX_SEEK_BACKEND.
        dtype: Model/vision precision; FP16 on MPS and NPUs without BF16 support,
            otherwise BF16.
        tp_size: Ascend tensor-parallel degree (1 or 2); launch all ranks with torchrun.

    Returns:
        The tokenizer, loaded model, and a tuple containing primary and
        optional auxiliary image processors.
    """
    device = initialize_tensor_parallel(resolve_device(device), tp_size)
    if dtype is None:
        dtype = inference_dtype(device)
    from vlx_seek.models.vlx_seek_1_5 import VLXSeek1_5ForCausalLM

    # Keep the model placement configuration consistent with Transformers.
    kwargs = {"dtype": dtype}
    if tp_size > 1:
        kwargs.update(tp_plan="auto", tp_size=tp_size)
    else:
        kwargs["device_map"] = device

    if 'vlx-seek' in model_path.lower(): 
        # Load the language model and checkpoint-provided multimodal modules.
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        model = VLXSeek1_5ForCausalLM.from_pretrained(
                model_path,
                low_cpu_mem_usage=True,
                attn_implementation=attention_implementation(device),
                **kwargs
            )
    else:
        raise ValueError(f"Unsupported model: {model_path}")

    if device.type == "npu":
        from .npu_fla import enable_npu_fla
        enable_npu_fla(model)
    
    vision_tower = model.get_vision_tower()
    if vision_tower and not vision_tower.is_loaded:
        vision_tower.load_model(is_train=False)            
    vision_tower.to(device=device, dtype=dtype)          
    primary_image_processor = vision_tower.image_processor

    vision_tower_aux = model.get_vision_tower_aux()
    aux_image_processor = None
    if vision_tower_aux is not None:
        if not vision_tower_aux.is_loaded:
            vision_tower_aux.load_model(is_train=False)
        vision_tower_aux.to(device=device, dtype=dtype)
        aux_image_processor = vision_tower_aux.image_processor
    
    # Keep the public return format stable for downstream callers.
    dual_image_processor = (primary_image_processor, aux_image_processor)
    
    return tokenizer, model, dual_image_processor
