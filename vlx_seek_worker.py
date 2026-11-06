"""Reusable inference worker for VLX-Seek 1.5.

The worker mirrors the Qwen3.5 inference path used during training, including
pre-expansion of image placeholders and optional object-feature placeholders
for pixel-coordinate bounding boxes.
"""
from __future__ import annotations

import random
import re
from typing import Optional, Sequence, Union

import torch
from PIL import Image

from vlx_seek.hardware import resolve_device
from vlx_seek.parallel import initialize_tensor_parallel, run_on_rank_zero
from vlx_seek.models.vlx_seek_1_5.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_OBJECT_FEATURE_TOKEN,
    DEFAULT_OBJECT_TOKEN,
    DEFAULT_OBJECT_INDEX,
    IMAGE_TOKEN_INDEX,
    VLX_SEEK_1_5_IMAGE_TOKEN_INDEX,
    VLX_SEEK_1_5_OBJECT_FEATURE_TOKEN_INDEX,
)
from vlx_seek.models.vlx_seek_1_5.mm_utils import (
    KeywordsStoppingCriteria,
    tokenizer_image_object_token,
    tokenizer_image_token,
)
from vlx_seek.task_templates import build_prompt


class VLXSeekWorker:
    """Stateful VLX-Seek 1.5 inference worker.

    ``bbox_list`` coordinates are pixel coordinates in the original input
    image: ``[[x1, y1, x2, y2], ...]``.  Passing boxes enables the model's
    auxiliary visual-prompt encoder and is required by region-level tasks.
    """

    def __init__(self, model_path: str, device: str | torch.device | None = None, tp_size: int = 1):
        self.tp_size = tp_size
        self.device = initialize_tensor_parallel(resolve_device(device), tp_size)
        from vlx_seek.models.vlx_seek_1_5.builder import load_pretrained_model

        self.tokenizer, self.model, image_processors = load_pretrained_model(
            model_path, device=self.device, tp_size=tp_size
        )
        self.image_processor, self.image_processor_aux = image_processors
        self.model.eval()

    @staticmethod
    def _validate_boxes(
        boxes: Optional[Sequence[Sequence[float]]], image: Image.Image
    ) -> Optional[list[list[float]]]:
        if boxes is None:
            return None

        width, height = image.size
        normalized = []
        for box in boxes:
            if len(box) != 4:
                raise ValueError("Each bbox must contain [x1, y1, x2, y2].")
            x1, y1, x2, y2 = (float(value) for value in box)
            x1, x2 = sorted((max(0.0, min(x1, width)), max(0.0, min(x2, width))))
            y1, y2 = sorted((max(0.0, min(y1, height)), max(0.0, min(y2, height))))
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"bbox must have positive area after clipping: {box}")
            normalized.append([x1, y1, x2, y2])
        return normalized

    def _order_boxes(
        self, boxes: Optional[list[list[float]]]
    ) -> tuple[Optional[list[list[float]]], list[int]]:
        """Order boxes internally and retain the mapping to caller indices."""
        if not boxes:
            return boxes, []

        order_mode = getattr(self.model.config, "mm_bbox_order_mode", "none")

        if order_mode == "none":
            return boxes, list(range(len(boxes)))
        if order_mode == "raster":
            sort_key = lambda item: (item[1][0], item[1][1], item[0])
        elif order_mode == "area_asc":
            sort_key = lambda item: (
                (item[1][2] - item[1][0]) * (item[1][3] - item[1][1]),
                item[0],
            )
        elif order_mode == "area_desc":
            sort_key = lambda item: (
                -((item[1][2] - item[1][0]) * (item[1][3] - item[1][1])),
                item[0],
            )
        else:
            raise ValueError(f"Unsupported mm_bbox_order_mode: {order_mode}")

        ordered = sorted(enumerate(boxes), key=sort_key)
        return [box for _, box in ordered], [original_index for original_index, _ in ordered]

    @staticmethod
    def _remap_object_tokens(answer: str, sorted_to_original: list[int]) -> str:
        """Map internal sorted object indices back to caller-provided indices."""
        def replace(match: re.Match) -> str:
            sorted_index = int(match.group(1))
            if sorted_index >= len(sorted_to_original):
                return match.group(0)
            return f"<obj{sorted_to_original[sorted_index]}>"

        return re.sub(r"<obj(\d+)>", replace, answer)

    @staticmethod
    def _build_result_bbox_list(
        answer: str, caller_boxes: Optional[list[list[float]]]
    ) -> list[dict]:
        """Extract categorized object references and their caller-facing boxes."""
        if not caller_boxes:
            return []

        result_bbox_list = []
        matches = re.finditer(
            r"<ground>(.*?)</ground>\s*<objects>(.*?)</objects>",
            answer,
            flags=re.DOTALL,
        )
        for match in matches:
            label = match.group(1).strip()
            for object_match in re.finditer(r"<obj(\d+)>", match.group(2)):
                object_index = int(object_match.group(1))
                if object_index >= len(caller_boxes):
                    continue
                xmin, ymin, xmax, ymax = caller_boxes[object_index]
                result_bbox_list.append(
                    {
                        "object_index": object_match.group(0),
                        "xmin": xmin,
                        "ymin": ymin,
                        "xmax": xmax,
                        "ymax": ymax,
                        "label": label,
                    }
                )

        return result_bbox_list

    def _build_prompt(
        self, question: str, boxes: Optional[Sequence[Sequence[float]]]
    ) -> str:
        image_part = f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}"
        if boxes:
            object_tokens = "".join(
                DEFAULT_OBJECT_TOKEN.replace("<i>", str(index))
                + DEFAULT_OBJECT_FEATURE_TOKEN
                for index in range(len(boxes))
            )
            image_part = f"{image_part}\n{object_tokens}"
        return (
            f"<|im_start|>user\n{image_part}\n{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    @staticmethod
    def _expand_multimodal_tokens(
        input_ids: torch.Tensor, image_grid_thws: list[torch.Tensor]
    ) -> torch.Tensor:
        """Expand placeholders before ``generate`` creates position encodings."""
        expanded_ids = []
        image_index = 0
        for token in input_ids.tolist():
            if token == IMAGE_TOKEN_INDEX:
                if image_index >= len(image_grid_thws):
                    raise ValueError("Prompt contains more image tokens than input images.")
                grid_thw = image_grid_thws[image_index]
                num_patches = int(torch.prod(grid_thw[0]).item() // 4)
                expanded_ids.extend([VLX_SEEK_1_5_IMAGE_TOKEN_INDEX] * num_patches)
                image_index += 1
            elif token == DEFAULT_OBJECT_INDEX:
                expanded_ids.append(VLX_SEEK_1_5_OBJECT_FEATURE_TOKEN_INDEX)
            else:
                expanded_ids.append(token)
        if image_index != len(image_grid_thws):
            raise ValueError("Every input image must have one <image> placeholder.")
        return torch.tensor(expanded_ids, dtype=torch.long, device=input_ids.device)

    def _prepare_image_inputs(
        self, image: Image.Image, boxes: Optional[list[list[float]]]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], Optional[list[torch.Tensor]]]:
        primary = self.image_processor.preprocess(image, return_tensors="pt")
        image_tensor = primary["pixel_values"]
        image_grid_thw = primary["image_grid_thw"]

        images_aux = None
        if boxes:
            auxiliary = self.image_processor_aux.preprocess(image, return_tensors="pt")
            images_aux = [auxiliary["pixel_values"][0].to(self.device)]

        return [image_tensor], [image_grid_thw], images_aux

    @torch.inference_mode()
    def predict(
        self,
        image: Image.Image,
        question: str,
        bbox_list: Optional[Sequence[Sequence[float]]] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
    ) -> dict:
        """Answer one image question, optionally using object-region prompts."""
        image = image.convert("RGB")
        caller_boxes = self._validate_boxes(bbox_list, image)
        boxes, sorted_to_original = self._order_boxes(caller_boxes)
        question = self._remap_prompt_object_tokens(question, sorted_to_original)
        prompt = self._build_prompt(question, boxes)

        if boxes:
            raw_input_ids = tokenizer_image_object_token(
                prompt, self.tokenizer, return_tensors="pt"
            )
        else:
            raw_input_ids = tokenizer_image_token(
                prompt, self.tokenizer, return_tensors="pt"
            )
        images, image_grid_thws, images_aux = self._prepare_image_inputs(image, boxes)
        input_ids = self._expand_multimodal_tokens(
            raw_input_ids, image_grid_thws
        ).unsqueeze(0).to(self.device)
        attention_mask = torch.ones_like(input_ids)

        do_sample = temperature > 0
        generate_kwargs = {
            "inputs": input_ids,
            "attention_mask": attention_mask,
            "images": images,
            "images_aux": images_aux,
            "image_grid_thws": image_grid_thws,
            "bbox_list": [torch.tensor(boxes)] if boxes else None,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "repetition_penalty": repetition_penalty,
            "pad_token_id": self.tokenizer.pad_token_id,
            "use_cache": True,
            "stopping_criteria": [
                KeywordsStoppingCriteria(["<|im_end|>"], self.tokenizer, input_ids)
            ],
        }
        if do_sample:
            if self.tp_size > 1:
                seed = run_on_rank_zero(lambda: random.getrandbits(63), self.tp_size)
                torch.manual_seed(seed)
            generate_kwargs.update(temperature=temperature, top_p=top_p)
        output_ids = self.model.generate(**generate_kwargs)
        completion_ids = output_ids[0, input_ids.shape[1] :]
        answer = self.tokenizer.decode(completion_ids, skip_special_tokens=False)
        answer = answer.replace("<|im_end|>", "").strip()
        answer = self._remap_object_tokens(answer, sorted_to_original)
        result_bbox_list = self._build_result_bbox_list(answer, caller_boxes)
        return {
            "answer": answer,
            "result_bbox_list": result_bbox_list,
            "prompt_tokens": input_ids.shape[1],
            "completion_tokens": completion_ids.shape[0],
        }

    def predict_batch(
        self,
        requests: list[Union[tuple[Image.Image, str], dict]],
        **kwargs,
    ) -> list[dict]:
        """Run requests serially while keeping the loaded model resident."""
        results = []
        for request in requests:
            if isinstance(request, dict):
                results.append(
                    self.predict(
                        request["image"],
                        request.get("question", request.get("prompt", "")),
                        bbox_list=request.get("bbox_list"),
                        **kwargs,
                    )
                )
            else:
                image, question = request
                results.append(self.predict(image, question, **kwargs))
        return results

    def run_task(
        self,
        image: Image.Image,
        task: str,
        text: str | list[str],
        lang: str = "en",
        bbox_list: Optional[Sequence[Sequence[float]]] = None,
        **kwargs,
    ) -> dict:
        """Build a VLX-Seek 1.5 task-template prompt and run inference."""
        prompt = build_prompt("vlx_seek_1_5", task, text, lang=lang)
        return self.predict(image, prompt, bbox_list=bbox_list, **kwargs)

    def detect(
        self,
        image: Image.Image,
        bbox_list: Sequence[Sequence[float]],
        categories: str | list[str],
        **kwargs,
    ) -> dict:
        """Detect categories among the candidate regions in ``bbox_list``."""
        return self.run_task(
            image, "detection", categories, bbox_list=bbox_list, **kwargs
        )

    def ground(
        self,
        image: Image.Image,
        bbox_list: Sequence[Sequence[float]],
        description: str,
        **kwargs,
    ) -> dict:
        """Locate one matching instance among the candidate ``bbox_list``."""
        return self.run_task(
            image, "grounding_single", description, bbox_list=bbox_list, **kwargs
        )

    def count(
        self,
        image: Image.Image,
        bbox_list: Sequence[Sequence[float]],
        description: str,
        **kwargs,
    ) -> dict:
        """Count matching objects among the candidate regions in ``bbox_list``."""
        return self.run_task(
            image, "counting", description, bbox_list=bbox_list, **kwargs
        )

    def reasoning_detect(
        self,
        image: Image.Image,
        bbox_list: Sequence[Sequence[float]],
        categories: str | list[str],
        **kwargs,
    ) -> dict:
        """Detect categories among candidate regions using explicit reasoning."""
        return self.run_task(
            image,
            "reasoning_detection",
            categories,
            bbox_list=bbox_list,
            **kwargs,
        )

    @staticmethod
    def _remap_prompt_object_tokens(
        question: str, sorted_to_original: list[int]
    ) -> str:
        """Map caller-facing object references to the model's sorted indices."""
        original_to_sorted = {
            original_index: sorted_index
            for sorted_index, original_index in enumerate(sorted_to_original)
        }

        def replace(match: re.Match) -> str:
            original_index = int(match.group(1))
            sorted_index = original_to_sorted.get(original_index)
            if sorted_index is None:
                return match.group(0)
            return f"<obj{sorted_index}>"

        return re.sub(r"<obj(\d+)>", replace, question)

    @staticmethod
    def _target_region_references(
        bbox_list: Sequence[Sequence[float]], target_region_indexes: Sequence[int]
    ) -> str:
        if not bbox_list:
            raise ValueError("A region task requires at least one bbox.")
        if not target_region_indexes:
            raise ValueError("A region task requires at least one target region index.")

        references = []
        for index in target_region_indexes:
            if isinstance(index, bool) or not isinstance(index, int):
                raise ValueError("Target region indexes must be integers.")
            if not 0 <= index < len(bbox_list):
                raise ValueError(
                    f"Target region index {index} is outside bbox_list "
                    f"(expected 0 to {len(bbox_list) - 1})."
                )
            references.append(f"<obj{index}>")
        return "".join(references)

    def describe_region(
        self,
        image: Image.Image,
        bbox_list: Sequence[Sequence[float]],
        target_region_indexes: Sequence[int],
        **kwargs,
    ) -> dict:
        """Briefly describe the requested regions in ``bbox_list``."""
        text = self._target_region_references(bbox_list, target_region_indexes)
        return self.run_task(
            image, "brief_region_caption", text, bbox_list=bbox_list, **kwargs
        )

    def read_region_text(
        self,
        image: Image.Image,
        bbox_list: Sequence[Sequence[float]],
        target_region_indexes: Sequence[int],
        **kwargs,
    ) -> dict:
        """Read text from the requested regions in ``bbox_list``."""
        text = self._target_region_references(bbox_list, target_region_indexes)
        return self.run_task(image, "region_ocr", text, bbox_list=bbox_list, **kwargs)
