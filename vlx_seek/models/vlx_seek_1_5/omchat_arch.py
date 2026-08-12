from abc import ABC, abstractmethod

from vlx_seek.models.vlx_seek_1_5.multimodal_encoder.builder import build_vision_tower, build_vision_tower_aux

from vlx_seek.models.vlx_seek_1_5.multimodal_projector.builder import build_vision_projector, build_vision_projector_aux

from vlx_seek.models.vlx_seek_1_5.multimodal_visual_prompt_encoder.hybrid_finegrained_region_encoder import HybridFineGrainedRegionEncoder


def _infer_aux_spatial_scale(aux_tower, default: float = 0.25) -> float:
    """Infer the RoI Align scale for the non-FPN auxiliary-feature path.

    ViT-style backbones expose an integer ``patch_size``, so their feature
    stride is inferred as ``1 / patch_size``. Hierarchical backbones usually
    expose a non-integer patch size and fall back to the shallow-stage scale.

    This scale is used only when the auxiliary SimpleFPN path is disabled;
    SimpleFPN computes the scale of each pyramid level independently.
    """
    if aux_tower is None:
        return default
    # Prefer an explicit scale supplied by a tower implementation.
    s = getattr(aux_tower, "spatial_scale", None)
    if isinstance(s, (int, float)):
        return float(s)
    cfg = getattr(aux_tower, "config", None)
    if cfg is not None:
        ps = getattr(cfg, "patch_size", None)
        if isinstance(ps, int) and ps > 0:
            return 1.0 / ps
    return default


class OmChatMetaModel:
    """Mixin that initializes VLX-Seek multimodal towers and projectors."""

    def __init__(self, config):
        super().__init__(config)
        has_vision_tower = getattr(config, "mm_vision_tower", None) is not None
        has_video_tower = getattr(config, "mm_video_tower", None) is not None
        has_aux_tower = getattr(config, "mm_vision_tower_aux", None) is not None

        # Build the primary visual encoder and its language-model projector.
        if has_vision_tower:
            self.vision_tower = build_vision_tower(config, delay_load=getattr(config, 'delay_load', True))
        if has_video_tower:
            # Video inputs use a separate tower attribute so they cannot
            # overwrite the primary image tower.
            self.video_tower = build_vision_tower(config, delay_load=getattr(config, 'delay_load', True))
        if has_vision_tower or has_video_tower:
            self.mm_projector = build_vision_projector(config)

        # The auxiliary tower supplies fine-grained region features for HFRE.
        if has_aux_tower:
            self.vision_tower_aux = build_vision_tower_aux(config, delay_load=getattr(config, 'delay_load', True))
            self.object_vp_extractor = HybridFineGrainedRegionEncoder(
                output_size=getattr(config, "mm_roi_output_size", 7),
                spatial_scale=_infer_aux_spatial_scale(self.get_vision_tower_aux()),
                add_pos_embedding=getattr(config, "mm_add_pos_embed", True),
                pos_embedding_dim=config.mm_object_hidden_size,
                use_vision_tower_object_feature=getattr(config, "mm_use_vision_tower_object_feature", False),
                vision_tower_object_feature_dim=self.get_vision_tower().config.hidden_size * 4 if not getattr(config, "mm_use_simpleFPN_for_vt", False) else self.get_vision_tower().config.out_hidden_size,
                vision_tower_spatial_scale=1/(self.get_vision_tower().config.patch_size*self.get_vision_tower().config.spatial_merge_size),
                object_feature_combination=getattr(config, "mm_object_feature_combination", "mean"),
                use_vt_object_feature_only=getattr(config, "mm_use_vt_object_feature_only", False),
                use_simpleFPN_for_vt=getattr(config, "mm_use_simpleFPN_for_vt", False),
                use_separate_mlp_for_object=getattr(config, "mm_use_separate_mlp_for_object", False),
                obj_pooling_type=getattr(config, "mm_obj_pooling_type", "mean"),
                use_multi_scale_roi_align=getattr(config, "mm_use_multi_scale_roi_align", False),
                apply_object_layer_norm=getattr(config, "mm_apply_object_layer_norm", False),
                roi_algined=getattr(config, "mm_roi_algined", False),
                use_simpleFPN_for_vt_aux=getattr(config, "mm_use_simpleFPN_for_vt_aux", False),
                aux_feature_dims=[getattr(self.get_vision_tower_aux().config, "hidden_size", None)],
                simpleFPN_out_channels_for_vt=getattr(config, "mm_simpleFPN_out_channels_for_vt", 512),
            )
            # Project HFRE object prompts into the language-model hidden size.
            self.mm_projector_aux = build_vision_projector_aux(config)

    def get_vision_tower(self):
        """Return the primary image tower, unwrapping legacy list containers."""
        vision_tower = getattr(self, 'vision_tower', None)
        if isinstance(vision_tower, list):
            vision_tower = vision_tower[0]
        return vision_tower

    def get_vision_tower_aux(self):
        """Return the auxiliary image tower, when configured."""
        vision_tower_aux = getattr(self, 'vision_tower_aux', None)
        if isinstance(vision_tower_aux, list):
            vision_tower_aux = vision_tower_aux[0]
        return vision_tower_aux

    def get_video_tower(self):
        """Return the video tower, when configured."""
        video_tower = getattr(self, 'video_tower', None)
        if isinstance(video_tower, list):
            video_tower = video_tower[0]
        return video_tower


class OmChatMetaForCausalLM(ABC):
    """Interface exposing multimodal towers from a causal language model."""

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()
    
    def get_vision_tower_aux(self):
        return self.get_model().get_vision_tower_aux()

    def get_video_tower(self):
        return self.get_model().get_video_tower()
