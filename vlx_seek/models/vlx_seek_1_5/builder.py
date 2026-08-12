from transformers import AutoTokenizer
import torch
from vlx_seek.models.vlx_seek_1_5 import VLXSeek1_5ForCausalLM


def load_pretrained_model(model_path, device="cuda", dtype=torch.bfloat16):
    """Load a VLX-Seek model and its primary/auxiliary image processors.

    Args:
        model_path: Local checkpoint directory or Hugging Face model identifier.
        device: Device or device-map target used to place model components.
        dtype: Floating-point precision used for vision towers.

    Returns:
        The tokenizer, loaded model, and a tuple containing primary and
        optional auxiliary image processors.
    """
    # Keep the model placement configuration consistent with Transformers.
    kwargs = {"device_map": device, "dtype": dtype}

    if 'vlx-seek' in model_path.lower(): 
        # Load the language model and checkpoint-provided multimodal modules.
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        model = VLXSeek1_5ForCausalLM.from_pretrained(
                model_path,
                low_cpu_mem_usage=True,
                attn_implementation="flash_attention_2",
                **kwargs
            )
    else:
        raise ValueError(f"Unsupported model: {model_path}")
    
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
