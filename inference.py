"""Command-line inference demo for VLX-Seek 1.5.

Examples:
    python inference.py --task detection --text "person; apple"
    python inference.py --task brief_region_caption --target-region-indexes "[2, 3]"
    python inference.py --task vqa --text "What is happening in this image?"
"""
import argparse
import importlib
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw

DEFAULT_DETECTOR_CHECKPOINT = "resources/wedetect_base_uni.pth"
WEDETECT_HF_REPO_ID = "fushh7/WeDetect"
WEDETECT_HF_FILENAME = "wedetect_base_uni.pth"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VLX-Seek 1.5 inference.")
    parser.add_argument("--model-path", default="resources/VLX-Seek-1.5-10B")
    parser.add_argument("--image-path", default="demo/demo_image.jpg")
    parser.add_argument(
        "--task",
        choices=(
            "vqa",
            "detection",
            "grounding_single",
            "counting",
            "reasoning_detection",
            "brief_region_caption",
            "region_ocr",
        ),
        default="detection",
    )
    parser.add_argument(
        "--text",
        default="orange; apple",
        help="Question, category, or description used to fill the task template.",
    )
    parser.add_argument("--lang", choices=("en", "zh"), default="en")
    parser.add_argument(
        "--bbox-list",
        default=None,
        help=(
            "JSON pixel boxes, e.g. '[[10, 20, 100, 200]]'. When omitted, "
            "WeDetect-Uni generates 100 proposals."
        ),
    )
    parser.add_argument(
        "--detector-checkpoint",
        default=DEFAULT_DETECTOR_CHECKPOINT,
        help=(
            "Path to the WeDetect-Base-Uni checkpoint used when --bbox-list "
            "is omitted. If the default path is missing, it is downloaded "
            f"from Hugging Face ({WEDETECT_HF_REPO_ID})."
        ),
    )
    parser.add_argument(
        "--target-region-indexes",
        default=None,
        help=(
            "JSON indexes of regions targeted by brief_region_caption or "
            "region_ocr. Required for those tasks, e.g. '[2, 3]'."
        ),
    )
    parser.add_argument(
        "--device", default=None,
        help="Override VLX_SEEK_BACKEND, e.g. cuda:0, npu:0, mps, or cpu; default: auto.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--tp-size", "--tp", type=int, choices=(1, 2), default=1,
                        help="Ascend tensor parallelism; TP=2 requires torchrun with two processes.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--visualize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw result_bbox_list on the input image and save it.",
    )
    parser.add_argument(
        "--visualization-output",
        default=None,
        help="Output path for the visualization. Defaults to <image>_result.png.",
    )
    args = parser.parse_args()
    if args.task in {"brief_region_caption", "region_ocr"} and args.target_region_indexes is None:
        parser.error(
            "--target-region-indexes is required for brief_region_caption and "
            "region_ocr, e.g. --target-region-indexes '[0]'."
        )
    return args


def load_boxes(boxes_json: str | None) -> list[list[float]] | None:
    if boxes_json is None:
        return None

    boxes = json.loads(boxes_json)
    if not isinstance(boxes, list):
        raise ValueError("--bbox-list must be a JSON list of [x1, y1, x2, y2] boxes.")
    return boxes


def ensure_wedetect_checkpoint(checkpoint_path: str) -> Path:
    """Return a local WeDetect checkpoint, downloading it if needed.

    When the default path is missing, download
    ``wedetect_base_uni.pth`` from ``fushh7/WeDetect`` into that path.
    Custom ``--detector-checkpoint`` paths are not auto-downloaded.
    """
    checkpoint = Path(checkpoint_path)
    if checkpoint.is_file():
        return checkpoint

    default_checkpoint = Path(DEFAULT_DETECTOR_CHECKPOINT)
    if checkpoint.resolve() != default_checkpoint.resolve():
        raise FileNotFoundError(
            "WeDetect-Uni checkpoint not found: "
            f"{checkpoint}. Provide --bbox-list or set "
            "--detector-checkpoint to an existing checkpoint file."
        )

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"WeDetect checkpoint not found at {checkpoint}. "
        f"Downloading {WEDETECT_HF_FILENAME}",
        file=sys.stderr,
    )
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to download the WeDetect "
            "checkpoint automatically. Install transformers/huggingface_hub, "
            "or place wedetect_base_uni.pth at "
            f"{DEFAULT_DETECTOR_CHECKPOINT}."
        ) from exc

    downloaded = hf_hub_download(
        repo_id=WEDETECT_HF_REPO_ID,
        filename=WEDETECT_HF_FILENAME,
        local_dir=str(checkpoint.parent),
    )
    downloaded_path = Path(downloaded)
    if downloaded_path.resolve() != checkpoint.resolve():
        downloaded_path.replace(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            "Failed to download WeDetect-Uni checkpoint to "
            f"{checkpoint} from {WEDETECT_HF_REPO_ID}."
        )
    print(f"Saved WeDetect checkpoint to {checkpoint}.", file=sys.stderr)
    return checkpoint


def load_wedetect_proposals(
    image: Image.Image, checkpoint_path: str, device=None
) -> list[list[float]]:
    """Generate the top 100 WeDetect-Uni xyxy proposals in original pixels."""
    checkpoint = ensure_wedetect_checkpoint(checkpoint_path)

    wedetect_dir = Path(__file__).resolve().parent / "detect_tools" / "WeDetect"
    if str(wedetect_dir) not in sys.path:
        sys.path.insert(0, str(wedetect_dir))

    import torch
    from vlx_seek.hardware import resolve_device

    device = resolve_device(device)

    proposal_module = importlib.import_module("generate_proposal")
    SimpleYOLOWorldDetector = proposal_module.SimpleYOLOWorldDetector

    model = SimpleYOLOWorldDetector(
        backbone_size="base",
        prompt_dim=768,
        num_prompts=256,
        num_proposals=100,
    )
    state_dict = torch.load(checkpoint, map_location="cpu")
    state_dict = state_dict.get("state_dict", state_dict)

    for key in list(state_dict.keys()):
        if "backbone" in key:
            new_key = key.replace("backbone.image_model.model.", "backbone.")
            state_dict[new_key] = state_dict.pop(key)
    for key in list(state_dict.keys()):
        if "bbox_head" in key:
            new_key = key.replace("bbox_head.head_module.", "bbox_head.")
            new_key = new_key.replace("0.2.", "0.6.")
            new_key = new_key.replace("1.2.", "1.6.")
            new_key = new_key.replace("2.2.", "2.6.")
            new_key = new_key.replace("1.bn", "4")
            new_key = new_key.replace("1.conv", "3")
            new_key = new_key.replace("0.bn", "1")
            new_key = new_key.replace("0.conv", "0")
            state_dict[new_key] = state_dict.pop(key)

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
    with torch.inference_mode():
        outputs = model([image])
        boxes = outputs[0]["bboxes"].float().cpu().tolist()

    del outputs, model
    if device.type in ("cuda", "npu", "mps"):
        getattr(torch, device.type).empty_cache()
    return boxes


def load_target_region_indexes(indexes_json: str | None) -> list[int]:
    if indexes_json is None:
        raise ValueError(
            "--target-region-indexes is required for brief_region_caption "
            "and region_ocr."
        )
    indexes = json.loads(indexes_json)
    if not isinstance(indexes, list) or any(
        isinstance(index, bool) or not isinstance(index, int) for index in indexes
    ):
        raise ValueError("--target-region-indexes must be a JSON list of integers.")
    return indexes


def visualize_result_boxes(
    image: Image.Image, result_bbox_list: list[dict], output_path: str
) -> str:
    """Draw worker-selected boxes and their model-generated labels."""
    visualized = image.copy()
    draw = ImageDraw.Draw(visualized)

    for result_bbox in result_bbox_list:
        xmin = result_bbox["xmin"]
        ymin = result_bbox["ymin"]
        xmax = result_bbox["xmax"]
        ymax = result_bbox["ymax"]
        label = result_bbox["label"]
        draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=3)
        try:
            draw.text((xmin + 2, ymin + 2), label, fill="red")
        except UnicodeEncodeError:
            draw.text((xmin + 2, ymin + 2), result_bbox["object_index"], fill="red")

    visualized.save(output_path)
    return output_path


def default_visualization_output(image_path: str) -> str:
    path = Path(image_path)
    return str(path.with_name(f"{path.stem}_result.png"))


def _run_inference(args) -> None:
    from vlx_seek.hardware import resolve_device
    from vlx_seek.parallel import initialize_tensor_parallel, is_main_process, run_on_rank_zero

    device = initialize_tensor_parallel(resolve_device(args.device), args.tp_size)
    print(f"VLX-Seek inference device: {device}, TP={args.tp_size}", file=sys.stderr)
    # Keep --help usable in environments that do not have VLX-Seek's required
    # Qwen3.5-enabled transformers build installed.
    from vlx_seek_worker import VLXSeekWorker

    image = Image.open(args.image_path).convert("RGB")
    proposal_tasks = {
        "detection",
        "grounding_single",
        "counting",
        "reasoning_detection",
        "brief_region_caption",
        "region_ocr",
    }
    boxes = None
    if args.task in proposal_tasks:
        boxes = load_boxes(args.bbox_list)
        if boxes is None:
            boxes = run_on_rank_zero(
                lambda: load_wedetect_proposals(image, args.detector_checkpoint, device=device),
                args.tp_size,
            )
        else:
            print(f"Using {len(boxes)} proposals from --bbox-list.", file=sys.stderr)
    target_region_indexes = (
        load_target_region_indexes(args.target_region_indexes)
        if args.task in {"brief_region_caption", "region_ocr"}
        else None
    )
    worker = VLXSeekWorker(args.model_path, device=device, tp_size=args.tp_size)
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }

    if args.task == "vqa":
        result = worker.predict(image, args.text, **generation_kwargs)
    elif args.task == "detection":
        import time
        for _ in range(10):
            s = time.time()
            result = worker.detect(
                image,
                boxes,
                args.text,
                lang=args.lang,
                **generation_kwargs,
            )
            if is_main_process(args.tp_size):
                print("time:", time.time() - s)
    elif args.task == "grounding_single":
        result = worker.ground(
            image,
            boxes,
            args.text,
            lang=args.lang,
            **generation_kwargs,
        )
    elif args.task == "counting":
        result = worker.count(
            image,
            boxes,
            args.text,
            lang=args.lang,
            **generation_kwargs,
        )
    elif args.task == "reasoning_detection":
        result = worker.reasoning_detect(
            image,
            boxes,
            args.text,
            lang=args.lang,
            **generation_kwargs,
        )
    elif args.task == "brief_region_caption":
        result = worker.describe_region(
            image,
            boxes,
            target_region_indexes,
            lang=args.lang,
            **generation_kwargs,
        )
    else:
        result = worker.read_region_text(
            image,
            boxes,
            target_region_indexes,
            lang=args.lang,
            **generation_kwargs,
        )

    if not is_main_process(args.tp_size):
        return
    if args.visualize and result.get("result_bbox_list"):
        output_path = args.visualization_output or default_visualization_output(
            args.image_path
        )
        result["visualization_path"] = visualize_result_boxes(
            image, result.get("result_bbox_list", []), output_path
        )

    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    try:
        _run_inference(args)
    finally:
        if args.tp_size > 1:
            from vlx_seek.parallel import shutdown_tensor_parallel
            shutdown_tensor_parallel()


if __name__ == "__main__":
    main()
