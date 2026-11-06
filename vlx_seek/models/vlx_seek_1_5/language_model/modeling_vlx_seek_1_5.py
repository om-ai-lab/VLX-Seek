
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model, Qwen3_5ForConditionalGeneration, Qwen3_5CausalLMOutputWithPast, Qwen3_5TextModel, Cache, Qwen3_5ModelOutputWithPast
from ..multimodal_encoder.qwen3_5_vl_encoder import Qwen3_5_VlVisionTower
from ..constants import (
    DEFAULT_OBJECT_INDEX,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    VLX_SEEK_1_5_IMAGE_TOKEN_INDEX,
    VLX_SEEK_1_5_OBJECT_FEATURE_TOKEN_INDEX,
)
from transformers.modeling_outputs import BaseModelOutputWithPooling
from vlx_seek.models.vlx_seek_1_5.omchat_arch import OmChatMetaModel, OmChatMetaForCausalLM



class VLXSeek1_5Config(Qwen3_5Config):
    model_type = "vlx_seek_1_5"
    rotary_type = "normal_rotary"
    multi_scale_im = None
    vision_tower_aux = None


class VLXSeek1_5Model(OmChatMetaModel, Qwen3_5Model):
    config: VLXSeek1_5Config

    def __init__(self, config: VLXSeek1_5Config):
        super(VLXSeek1_5Model, self).__init__(config)
        self.visual = None
        self.language_model = Qwen3_5TextModel._from_config(config.text_config)
        self.rope_deltas = None

        self.post_init()
    
    def encode_images(self, images, images_grid_thw=None):
        if isinstance(self.get_vision_tower(), Qwen3_5_VlVisionTower):
            image_features, image_grid_thws, multi_level_features_list = self.get_vision_tower()(images, images_grid_thw)
            if type(image_features) is list:
                token_length_list = [i.shape[1] for i in image_features]
                image_features = torch.cat(image_features, dim=1)
        else:
            image_features = self.get_vision_tower()(images)
            image_grid_thws = None
            multi_level_features = None

        image_features = self.mm_projector(image_features)

        if isinstance(self.get_vision_tower(), Qwen3_5_VlVisionTower):
            start = 0
            new_image_features = []
            for length in token_length_list:
                end = start + length
                new_image_features.append(image_features[:, start:end, :].squeeze(0))
                start = end
            image_features = new_image_features
        
        return image_features, image_grid_thws, multi_level_features_list
    
    def encode_objects(self, images, bbox_list, vt_multi_level_features=None, vt_images_size=None, add_pos_embed=True):
        aux_image_features_list = self.get_vision_tower_aux()(images)
        object_features = []
        
        if getattr(self.config, "mm_use_vision_tower_object_feature", False):
            image_features_list = vt_multi_level_features
            for batch_idx, (image_features, aux_image_features, boxes) in enumerate(zip(image_features_list, aux_image_features_list, bbox_list)):
                
                if boxes is None or len(boxes) == 0:
                    continue

                if getattr(self.config, "mm_use_simpleFPN_for_vt", False):
                    multilevel_visual_feats = image_features[-1]
                else:
                    multilevel_visual_feats = image_features
                
                if getattr(self.config, "mm_use_simpleFPN_for_vt_aux", False):
                    multilevel_aux_visual_feats = [aux_image_features["image_features"][-1]]
                else:
                    multilevel_aux_visual_feats = aux_image_features["image_features"]

                if boxes is None or len(boxes) == 0:
                    boxes = torch.tensor([[0, 10, 0, 10]], device=multilevel_aux_visual_feats[0].device, dtype=torch.float32)

                current_image_height, current_image_width = images[batch_idx].shape[-2:]
                boxes = boxes.to(torch.float32).to(multilevel_aux_visual_feats[0].device)

                original_height, original_width = vt_images_size[batch_idx]
                scale_height = original_height / current_image_height
                scale_width = original_width / current_image_width

                scale_tensor = torch.tensor([scale_width, scale_height, scale_width, scale_height], device=boxes.device)

                vt_boxes = boxes * scale_tensor
                extracted_object_feat = self.object_vp_extractor(
                    multi_level_features=multilevel_aux_visual_feats,
                    vt_multi_level_features=multilevel_visual_feats,
                    boxes=[boxes],
                    vt_boxes=[vt_boxes],
                    add_pos_embed=add_pos_embed
                ).squeeze(0).to(
                    dtype=next(self.mm_projector_aux.parameters()).dtype
                )

                object_feat = self.mm_projector_aux(extracted_object_feat)
                object_features.append(object_feat)
        else:
            for batch_idx, image_features in enumerate(aux_image_features_list):
                multilevel_visual_feats = image_features["image_features"]
                last_feat = image_features["last_feat"]
                boxes = bbox_list[batch_idx]

                if boxes is None or len(boxes) == 0:
                    boxes = torch.tensor([[0, 10, 0, 10]], device=multilevel_visual_feats[0].device, dtype=torch.float32)
                multi_level_aux_features = multilevel_visual_feats
                current_image_height, current_image_width = images[batch_idx].shape[-2:]
                boxes = boxes.to(torch.float32).to(multi_level_aux_features[0].device)

                extracted_object_feat = self.object_vp_extractor(
                    multi_level_aux_features,
                    [boxes],
                    add_pos_embed=add_pos_embed
                ).squeeze(0).to(
                    dtype=next(self.mm_projector_aux.parameters()).dtype
                )
                object_feat = self.mm_projector_aux(extracted_object_feat)
                object_features.append(object_feat)

        return object_features
            
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        images: Optional[torch.Tensor] = None,
        images_aux: Optional[torch.Tensor] = None,
        bbox_list: Optional[torch.Tensor] = None,
        image_grid_thws: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[tuple, Qwen3_5ModelOutputWithPast]:

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)


        if images is not None and len(images) > 0:
            vision_tower = self.get_vision_tower()
            vision_tower_aux = self.get_vision_tower_aux()
            image_embeds, image_grid_thws, vt_multi_level_features_list = self.encode_images(images, image_grid_thws)
                
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)

            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            has_bbox = False
            if bbox_list is not None:
                for bbox in bbox_list:
                    if bbox is not None and len(bbox) > 0:
                        has_bbox = True
                        break

            run_object_encoder = False
            if images_aux is not None and vision_tower_aux is not None and has_bbox:
                run_object_encoder = True
                patch_size = vision_tower.config.patch_size
                vt_images_size_minibatch = [im_grid_thw[0][-2:]*patch_size for im_grid_thw in image_grid_thws]
                tmp_images_aux = [images_aux[i].unsqueeze(0) for i in range(len(images_aux))]
                object_features = self.encode_objects(tmp_images_aux, bbox_list, vt_multi_level_features_list, vt_images_size_minibatch)

                valid_object_features = []
                obj_feat_idx = 0
                
                for i, input_id in enumerate(input_ids):
                    num_obj_tokens = (input_id == VLX_SEEK_1_5_OBJECT_FEATURE_TOKEN_INDEX).sum().item()
                    
                    if num_obj_tokens == 0:
                        continue
                    
                    if obj_feat_idx >= len(object_features):
                        raise ValueError(f"Not enough object features. Batch {i} needs features but list is exhausted.")
                        
                    feat = object_features[obj_feat_idx]
                    obj_feat_idx += 1
                    
                    if feat is None:
                         raise ValueError(f"Batch {i} has object tokens but feature is None")

                    if feat.shape[0] != num_obj_tokens:
                        print(f"Warning: Batch {i} object token count {num_obj_tokens} != feature count {feat.shape[0]}")
                        if feat.shape[0] > num_obj_tokens:
                             feat = feat[:num_obj_tokens]
                        else:
                             raise ValueError(f"Batch {i} feature count {feat.shape[0]} < token count {num_obj_tokens}")
                    
                    valid_object_features.append(feat)
                
                if len(valid_object_features) > 0:
                    all_object_features = torch.cat(valid_object_features, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
                    object_mask = (input_ids == VLX_SEEK_1_5_OBJECT_FEATURE_TOKEN_INDEX)
                    if object_mask.sum() == all_object_features.shape[0]:
                        inputs_embeds[object_mask] = all_object_features
                    else:
                        raise ValueError(f"Total object tokens {object_mask.sum()} != Total object features {all_object_features.shape[0]}")

            if self.training and not run_object_encoder and getattr(self, "object_vp_extractor", None) is not None and \
                getattr(self, "mm_projector_aux", None) is not None and \
                vision_tower_aux is not None:
                patch_size = vision_tower.config.patch_size
                dummy_idx = 0
                dummy_aux_img = [torch.zeros((1, 3, 224, 224), device=inputs_embeds.device, dtype=inputs_embeds.dtype)]
                dummy_vt_size = [torch.tensor([32, 32], device=inputs_embeds.device, dtype=torch.long)]
                dummy_bbox = [torch.tensor([[0, 0, 10, 10]], device=inputs_embeds.device, dtype=torch.float32)]
                dummy_vt_features = [vt_multi_level_features_list[dummy_idx]]
                
                dummy_obj_feats = self.encode_objects(
                    dummy_aux_img, 
                    dummy_bbox, 
                    dummy_vt_features, 
                    dummy_vt_size,
                )
                if dummy_obj_feats and dummy_obj_feats[0] is not None:
                    inputs_embeds = inputs_embeds + (0.0 * dummy_obj_feats[0].sum())

            image_grid_thw = torch.cat(image_grid_thws, dim=0)
        else:
            vision_tower = self.get_vision_tower()
            vision_tower_aux = self.get_vision_tower_aux()
            
            run_dummy_visual = False
            if self.training and getattr(self, "mm_projector", None) is not None and vision_tower is not None:
                patch_size = vision_tower.config.patch_size
                run_dummy_visual = True

            if run_dummy_visual:
                patch_size = vision_tower.config.patch_size
                dummy_image = torch.zeros((100, 1536), device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                dummy_grid = torch.tensor([[1, 10, 10]], device=inputs_embeds.device, dtype=torch.long)
                dummy_grid_thws = [dummy_grid]
                
                image_embeds,  _, vt_multi_level_features_list = self.encode_images([dummy_image], dummy_grid_thws)
                
                if image_embeds is not None:
                     if isinstance(image_embeds, list):
                          image_embeds = torch.cat(image_embeds, dim=0)
                     inputs_embeds = inputs_embeds + (0.0 * image_embeds.sum())

                if getattr(self, "object_vp_extractor", None) is not None and \
                   getattr(self, "mm_projector_aux", None) is not None and \
                   vision_tower_aux is not None:
                   
                   dummy_aux_img = [torch.zeros((1, 3, 224, 224), device=inputs_embeds.device, dtype=inputs_embeds.dtype)]
                   dummy_vt_size = [im_grid_thw[0][-2:]*patch_size for im_grid_thw in dummy_grid_thws]
                   dummy_bbox = [torch.tensor([[0, 0, 10, 10]], device=inputs_embeds.device, dtype=torch.float32)]
                   
                   dummy_obj_feats = self.encode_objects(
                        dummy_aux_img,
                        dummy_bbox,
                        vt_multi_level_features_list,
                        dummy_vt_size,
                   )
                   if dummy_obj_feats and dummy_obj_feats[0] is not None:
                        inputs_embeds = inputs_embeds + (0.0 * dummy_obj_feats[0].sum())
                        
        if pixel_values_videos is not None:
            video_outputs: BaseModelOutputWithPooling = self.get_video_features(
                pixel_values_videos, video_grid_thw, return_dict=True
            )
            video_embeds = video_outputs.pooler_output
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
            
        if position_ids is None:
            position_ids = self.compute_3d_position_ids(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )

        return Qwen3_5ModelOutputWithPast(
            **outputs,
            rope_deltas=self.rope_deltas,
        )



class VLXSeek1_5ForCausalLM(Qwen3_5ForConditionalGeneration, OmChatMetaForCausalLM):
    config: VLXSeek1_5Config

    def __init__(self, config, delay_load=True):
        if not hasattr(config, 'delay_load'):
            config.delay_load = delay_load
        super(Qwen3_5ForConditionalGeneration, self).__init__(config)
        self.model = VLXSeek1_5Model(config)

        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        self.post_init()
        # The conditional-generation base omits the language head from its TP plan.
        self._tp_plan["lm_head"] = "colwise_gather_output"
    
    def get_model(self):
        return self.model
    
    def convert_input_ids(self, input_ids, image_grid_thws, attention_mask=None, labels=None):
        if image_grid_thws is None or len(image_grid_thws) == 0:
            return input_ids, attention_mask, labels
            
        new_input_ids_list = []
        new_labels_list = [] if labels is not None else None
        cur_image_idx = 0 

        for i, input_id in enumerate(input_ids):
            if attention_mask is not None:
                valid_len = attention_mask[i].sum()
                cur_input_id = input_id[:valid_len]
                cur_labels = labels[i][:valid_len] if labels is not None else None
            else:
                cur_input_id = input_id
                cur_labels = labels[i] if labels is not None else None

            image_indices = torch.where(cur_input_id == IMAGE_TOKEN_INDEX)[0]
            object_indices = torch.where(cur_input_id == DEFAULT_OBJECT_INDEX)[0]

            all_indices = []
            for idx in image_indices:
                all_indices.append((idx.item(), 'image'))
            for idx in object_indices:
                all_indices.append((idx.item(), 'object'))
            all_indices.sort()

            if not all_indices:
                new_input_ids_list.append(cur_input_id)
                if new_labels_list is not None:
                    new_labels_list.append(cur_labels)
                continue

            pieces = []
            labels_pieces = [] if cur_labels is not None else None
            last_idx = 0
            
            for idx, token_type in all_indices:
                pieces.append(cur_input_id[last_idx:idx])
                if labels_pieces is not None:
                    labels_pieces.append(cur_labels[last_idx:idx])
                
                if token_type == 'image':
                    T, H, W = image_grid_thws[cur_image_idx][0]
                    cur_image_idx += 1
                    num_patches = (T * H * W) // 4
                    pieces.append(torch.full((num_patches,), VLX_SEEK_1_5_IMAGE_TOKEN_INDEX, 
                                           dtype=input_id.dtype, device=input_id.device))
                    if labels_pieces is not None:
                        labels_pieces.append(torch.full((num_patches,), IGNORE_INDEX, 
                                                      dtype=cur_labels.dtype, device=cur_labels.device))
                else:
                    pieces.append(torch.full((1,), VLX_SEEK_1_5_OBJECT_FEATURE_TOKEN_INDEX, 
                                           dtype=input_id.dtype, device=input_id.device))
                    if labels_pieces is not None:
                        labels_pieces.append(torch.full((1,), IGNORE_INDEX, 
                                                      dtype=cur_labels.dtype, device=cur_labels.device))
                last_idx = idx + 1
            
            pieces.append(cur_input_id[last_idx:])
            new_input_ids_list.append(torch.cat(pieces))
            
            if labels_pieces is not None:
                labels_pieces.append(cur_labels[last_idx:])
                new_labels_list.append(torch.cat(labels_pieces))
 
        max_len = max(x.shape[0] for x in new_input_ids_list)
        if self.training:
            max_len = (max_len + 7) // 8 * 8
        else:
            max_len = max_len
        
        batch_size = len(new_input_ids_list)
        device = input_ids.device
        
        pad_token_id = getattr(self.config, 'pad_token_id', 0)

        final_input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=input_ids.dtype, device=device)
        final_attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
        final_labels = None
        if new_labels_list is not None:
            final_labels = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=labels.dtype, device=device)
        
        for i, cur_ids in enumerate(new_input_ids_list):
            length = cur_ids.shape[0]
            final_input_ids[i, :length] = cur_ids
            final_attention_mask[i, :length] = 1
            if final_labels is not None:
                final_labels[i, :length] = new_labels_list[i]
            
        return final_input_ids, final_attention_mask, final_labels

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        images: Optional[torch.FloatTensor] = None,
        images_aux: Optional[torch.FloatTensor] = None,
        bbox_list: Optional[torch.FloatTensor] = None,
        image_grid_thws: Optional[torch.FloatTensor] = None,
        **kwargs
    ) -> Union[Tuple, Qwen3_5CausalLMOutputWithPast]:
        if input_ids is not None and (input_ids == IMAGE_TOKEN_INDEX).any():
            input_ids, attention_mask, labels = self.convert_input_ids(input_ids, image_grid_thws, attention_mask, labels)

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            images=images,
            images_aux=images_aux,
            bbox_list=bbox_list,
            image_grid_thws=image_grid_thws,
            **kwargs,
        )

        hidden_states = outputs[0]

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])


        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

        return Qwen3_5CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )
    
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        is_first_iteration=False,
        images: Optional[torch.FloatTensor] = None,
        images_aux: Optional[torch.FloatTensor] = None,
        bbox_list: Optional[torch.FloatTensor] = None,
        image_grid_thws: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_cache=use_cache,
            is_first_iteration=is_first_iteration,
            images=images,
            images_aux=images_aux,
            bbox_list=bbox_list,
            image_grid_thws=image_grid_thws,
            **kwargs,
        )   

        if not is_first_iteration and use_cache:
            model_inputs["images"] = None
            model_inputs["images_aux"] = None
        return model_inputs
    
AutoConfig.register("vlx_seek_1_5", VLXSeek1_5Config)
AutoModelForCausalLM.register(VLXSeek1_5Config, VLXSeek1_5ForCausalLM)
