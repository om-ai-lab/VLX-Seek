# VLX-Seek 1.5 Inference Guide

English | [中文](inference_zh.md)

This guide covers command-line and Python inference for VLX-Seek 1.5-10B. For the model overview and installation instructions, see the [project README](../README.md).

## Prerequisites

The provided pipeline currently requires:

- inference dependencies, installed locally with `requirements.txt` or provided by the Docker image described below;
- the VLX-Seek 1.5-10B checkpoint from [omlab/VLX-Seek-1.5-10B](https://huggingface.co/omlab/VLX-Seek-1.5-10B);
- the region detector checkpoint from [WeDetect](https://huggingface.co/fushh7/WeDetect)

You can load the VLX-Seek main model directly from Hugging Face:

```bash
python inference.py --model-path omlab/VLX-Seek-1.5-10B ...
```

When `--bbox-list` is omitted, `wedetect_base_uni.pth` is automatically downloaded to `resources/wedetect_base_uni.pth` if needed.

Alternatively, place local copies in the default directory:

```text
resources/
├── VLX-Seek-1.5-10B/
└── wedetect_base_uni.pth
```

`--model-path` accepts either a Hugging Face model identifier or a local checkpoint directory. The model architecture is provided by the local `vlx_seek` package in this repository.

Alternatively, set paths with `--model-path` and `--detector-checkpoint`.

### Hardware Selection

Use `VLX_SEEK_BACKEND` to select hardware for both the CLI and `VLXSeekWorker`:

| Value | Device |
| --- | --- |
| `auto` (default) | Detect NVIDIA CUDA first, then Ascend NPU, then Apple MPS, otherwise CPU |
| `nvidia` | NVIDIA GPU with CUDA |
| `ascend` / `310p` | Ascend NPU; tested on 910B and 310P3 |
| `cpu` | CPU with SDPA attention |
| `apple` / `mps` | Apple Silicon GPU with MPS and SDPA attention |

```bash
VLX_SEEK_BACKEND=nvidia python inference.py --task detection --text "orange; apple"
VLX_SEEK_BACKEND=ascend python inference.py --task detection --text "orange; apple"
VLX_SEEK_BACKEND=apple python inference.py --task detection --text "orange; apple"
```

An explicit `--device cuda:1`, `--device npu:0`, `--device mps`, or Python API `device=` overrides the environment variable. Selecting an unavailable backend raises an error instead of silently switching hardware. NVIDIA/CPU/MPS selection does not require NPU dependencies or change CUDA APIs.

### Apple Silicon (macOS)

MPS is available only on the macOS host. Docker containers cannot access the
host's MPS drivers; run the following installation and inference commands
directly in a macOS terminal.

We recommend downloading the Apple Silicon installer from the
[Tsinghua Miniforge mirror](https://mirrors-i.tuna.tsinghua.edu.cn/github-release/conda-forge/miniforge/LatestRelease/Miniforge3-26.7.2-0-MacOSX-arm64.sh).
After downloading, run the installer from the directory containing it:

```bash
bash Miniforge3-26.7.2-0-MacOSX-arm64.sh
```

Follow the installer prompts to initialize conda, then open a new terminal.
From the repository root, create the environment and install the macOS
dependencies using the Tsinghua PyPI mirror. These dependencies omit CUDA-only
Flash Attention:

```bash
conda create -n vlx-seek python=3.12
conda activate vlx-seek
pip install -r requirements_mac.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Run inference in the activated environment:

```bash
bash scripts/run_mac.sh
```

MPS uses FP16 model weights, SDPA attention, and the PyTorch Linear Attention implementation. The proposal detector also runs on MPS. ROIAlign runs in FP32 on CPU and returns features to MPS because torchvision has no native MPS ROIAlign kernel. This path does not require Flash Attention, Triton, or CUDA FLA kernels.

### Optional: Faster Inference

The model uses Linear Attention layers. Without the corresponding acceleration kernels, you will see a warning like the following and fall back to a slower PyTorch implementation:

```text
The fast path is not available because one of the required library is not installed. Falling back to torch implementation.
```

On NVIDIA GPUs, installing [flash-linear-attention](https://github.com/fla-org/flash-linear-attention#installation) and [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) enables the fast path. For NPU acceleration, see the [Ascend 310P](#docker-inference-on-ascend-310p-aarch64) and [910B](#docker-inference-on-ascend-910b-aarch64) instructions below.

## Docker Inference on NVIDIA x86 (e.g. A100)

Use the provided Docker environment on a Linux x86_64 host with an NVIDIA GPU such as A100. The host needs an NVIDIA driver, Docker, and NVIDIA Container Toolkit configured for GPU containers. The image includes the inference dependencies, including `flash-linear-attention` and `causal-conv1d`.

Host NVIDIA driver requirements:

| x86 GPU type | Minimum Driver Version |
| --- | --- |
| Data center GPUs such as A100 | 535 or later |
| Consumer GPUs such as RTX 4090 | 580 or later |

### Prepare the Image

Clone the repository if needed, then run all host-side commands from its root directory:

```bash
git clone https://github.com/om-ai-lab/VLX-Seek.git
cd VLX-Seek
```

Choose one of the following:

- Build with [Dockerfile.nvidia_x86](../docker/Dockerfile.nvidia_x86):

  ```bash
  docker build -f docker/Dockerfile.nvidia_x86 -t vlx-seek:py3.12-ubuntu24.04-nvidia-x86_64 .
  ```

- Pull our prebuilt image:

  ```bash
  docker pull registry.linker.cc/vlx/vlx-seek:py3.12-ubuntu24.04-nvidia-x86_64
  ```

### Start and Enter the Container

Run the following command from the repository root. It selects host GPU `0` through `--gpus device=0`; change the index to use another GPU. For a locally built image, replace `registry.linker.cc/vlx/vlx-seek:py3.12-ubuntu24.04-nvidia-x86_64` with `vlx-seek:py3.12-ubuntu24.04-nvidia-x86_64`.

```bash
docker run -d --name seek-dev \
  --gpus device=0 \
  --shm-size=40g \
  -v "$PWD:/workspace" \
  -w /workspace \
  -e VLX_SEEK_BACKEND=nvidia \
  registry.linker.cc/vlx/vlx-seek:py3.12-ubuntu24.04-nvidia-x86_64 \
  sleep infinity
docker exec -it seek-dev bash
```

The command mounts the repository at `/workspace` and sets it as the working directory, making your host-side source code, local weights, input images, and saved results available inside the container. The container stays running so you can enter it again with `docker exec -it seek-dev bash`.

### Run Inference Inside the Container

```bash
bash scripts/run_nvidia_x86.sh
```

The first run requires access to Hugging Face to download the main model and, when proposals are needed, `resources/wedetect_base_uni.pth`. To use a local main-model checkpoint, place it in the host repository's `resources/VLX-Seek-1.5-10B/` directory and use `--model-path resources/VLX-Seek-1.5-10B` instead. The Docker image does not bundle model weights.

Results are printed as JSON, and the example visualization is saved to `demo/demo_image_result.png` in the mounted host repository. You can run the other CLI examples and Python API below from `/workspace` in the same container.

## Docker Inference on T5000 and GB10 (aarch64)

Host platform requirements:

| Platform | Requirement |
| --- | --- |
| Jetson series (T5000) | JetPack 7.0 or later; JetPack 7.2 recommended |
| NVIDIA Spark GB10 | NVIDIA Driver Version 580 or later |

Use the prebuilt image matching your aarch64 platform:

| Platform | Image |
| --- | --- |
| T5000 | `registry.linker.cc/vlx/vlx-seek:py3.12-ubuntu24.04-t5000-aarch64` |
| GB10 | `registry.linker.cc/vlx/vlx-seek:py3.12-ubuntu24.04-gb10-aarch64` |

The host needs the NVIDIA driver, Docker, and NVIDIA Container Toolkit with the `nvidia` runtime configured. Clone the repository as shown above and run all host-side commands from its root directory.

### T5000 Docker Run

Run from the repository root to start and enter the container:

```bash
docker run -d --name seek-dev \
  --runtime=nvidia \
  --restart=always \
  --shm-size=40g \
  -p 1233:8000 \
  -v "$PWD:/workspace" \
  -w /workspace \
  -e VLX_SEEK_BACKEND=nvidia \
  registry.linker.cc/vlx/vlx-seek:py3.12-ubuntu24.04-t5000-aarch64 \
  sleep infinity
docker exec -it seek-dev bash
```

### GB10 Docker Run

Run from the repository root to start and enter the container:

```bash
docker run -d --name seek-dev \
  --runtime=nvidia \
  --restart=always \
  --shm-size=40g \
  -p 1233:8000 \
  -v "$PWD:/workspace" \
  -w /workspace \
  -e VLX_SEEK_BACKEND=nvidia \
  registry.linker.cc/vlx/vlx-seek:py3.12-ubuntu24.04-gb10-aarch64 \
  sleep infinity
docker exec -it seek-dev bash
```

`docker run` pulls the selected image if it is not available locally. Both commands mount the repository at `/workspace`, publish container port `8000` on host port `1233`, and use the container name `seek-dev`. Stop and remove an existing container with that name before switching platforms.

From `/workspace` inside either container, run `bash scripts/run_t5000.sh` or
`bash scripts/run_spark_gb10.sh`, respectively. Model downloads and local weight
paths work as described in the NVIDIA x86 section.

## Docker Inference on Ascend 310P (aarch64)

Use the published 310P image on a Linux aarch64 host with an Ascend 310P NPU.
The image is `registry.linker.cc/vlx/vlx-seek:py3.11-openeuler24.03-310p-aarch64`;
it contains the AscendC GDN operator package at `/opt/ascend/custom_transformer`.
The host needs Docker and a working Ascend driver. From the repository root, pull
the image and start a container with the project mounted (no `.cache` mount is needed):

```bash
docker pull registry.linker.cc/vlx/vlx-seek:py3.11-openeuler24.03-310p-aarch64
docker run -d --name seek-dev \
  --entrypoint sleep \
  --device /dev/davinci0 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  --shm-size=10g \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v "$PWD:/workspace" -w /workspace \
  -e VLX_SEEK_BACKEND=ascend \
  registry.linker.cc/vlx/vlx-seek:py3.11-openeuler24.03-310p-aarch64 \
  infinity
docker exec -it seek-dev bash
```

Replace `/dev/davinci0` with the device selected using `npu-smi info` if needed.
The repository mount provides source code and output paths; the image supplies the
310P operator package and Python dependencies.

Run inference inside the container:

```bash
cd /workspace
bash scripts/run_310p.sh
```

The actual chip selects the compatibility path: FP16 weights and SDPA attention
on 310P. The image includes the AscendC GDN operator package; no `.cache` volume
mount or runtime `set_env.bash` is needed. With that image, `full` uses AscendC for supported GDN
calls. An image without the package falls back to PyTorch linear attention.
See [310P FLA setup](npu_fla_zh.md#ascend-npu-的-fla-推理加速) to build the image.

The `torchvision::nms` CPU fallback warning does not prevent inference. Compare
cold and warmed inference times separately because the first call may compile
operators. This validation covers inference; training on 310P has not been tested.

With 310P3, CANN 9.0.1, PyTorch 2.9.0, torch_npu 2.9.0.post4,
and Transformers 5.13.0, the default detection example completed 10 runs. The
final result identified five oranges and two apples using 19 generated tokens.
The first run took about 59.4 seconds and subsequent runs about 9.3 seconds each,
excluding model loading and WeDetect proposals. Other inputs require their
own accuracy and performance evaluation.

#### Two-Device Tensor Parallelism (TP=2)

The two logical 310P3 devices on the tested machine passed HCCL AllReduce /
AllGather and two-device detection. Inside the container, launch detection with:

```bash
cd /workspace
TP_SIZE=2 bash scripts/run_310p.sh
```

To launch detection with `torchrun` directly:

```bash
torchrun --standalone --nproc-per-node=2 inference.py \
  --tp-size 2 --task detection --text "orange; apple"
```

`--tp 2` aliases `--tp-size 2`. Processes bind to `npu:0` and `npu:1`; omit a
fixed `--device npu:0`. Single-device inference remains the default. Transformers'
TP plan shards text attention and MLP projections; the language output head is
also sharded. Vision encoders, token embeddings, and some linear-attention work
are replicated, so neither memory nor latency is expected to halve. The 310P
FP16 / SDPA path still applies; GDN follows the AscendC package availability
described above. WeDetect runs only on
rank 0 and shares proposals over a CPU Gloo group. Only rank 0 writes results
and visualizations. Other 310P board combinations require HCCL verification.

The same detection example completed 10 TP=2 runs: about 24.0 seconds for the
first call and a mean of 5.946 seconds for the remaining nine. The final answer
and boxes matched the earlier single-device example, with 19 generated tokens.
These times exclude weight loading and WeDetect proposals. Earlier warmed
single-device runs took about 9.3 seconds. Compilation cache states differed,
so first-call times are not directly comparable; measure other tasks separately.

The Python API accepts `VLXSeekWorker(model_path, tp_size=2)`. Launch the script
with `torchrun --standalone --nproc-per-node=2`. Both ranks must call inference
in the same order with identical images, questions, proposals, and generation
options. Sampling seeds are synchronized between ranks. Call
`vlx_seek.parallel.shutdown_tensor_parallel()` on exit to release process groups.

## Docker Inference on Ascend 910B (aarch64)

Use the prebuilt image on a Linux aarch64 host with an Ascend 910B NPU. The host needs Docker and a working Ascend driver; verify the device with `npu-smi info`. The image includes CANN 9.0.1, PyTorch 2.9.0, `torch_npu` 2.9.0.post4, Transformers 5.13.0, Ascend Triton, and NPU FLA dependencies. This inference path has been tested on a 910B4-1.

### Prepare the Image and Container

Clone the repository as shown above, then run the following commands from its root:

```bash
docker pull registry.linker.cc/vlx/vlx-seek:py3.11-openeuler24.03-910b-aarch64
docker run -d --name seek-dev \
  --entrypoint sleep \
  --device /dev/davinci0 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  --shm-size=10g \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v "$PWD:/workspace" \
  -w /workspace \
  -e VLX_SEEK_BACKEND=ascend \
  registry.linker.cc/vlx/vlx-seek:py3.11-openeuler24.03-910b-aarch64 \
  infinity
docker exec -it seek-dev bash
```

Replace `/dev/davinci0` with the host device selected using `npu-smi info`, for example `/dev/davinci2`. Adjust the driver mount paths if your host uses a different layout. All Docker examples use the container name `seek-dev`; use another name if it is already occupied.

### Run Inference Inside the Container

Run from `/workspace`:

```bash
bash scripts/run_910b.sh
```

The Ascend backend imports `torch_npu` and places both the proposal detector and VLX-Seek on NPU explicitly. It does not globally replace CUDA APIs. The image does not include model weights. The first run downloads the main model and, if needed, the proposal detector. To use local weights, set `--model-path resources/VLX-Seek-1.5-10B`. Results and the visualization are saved in the mounted repository as described above.

NPU FLA is enabled by default when its dependencies are available:

| `VLX_SEEK_NPU_FLA` | Behavior |
| --- | --- |
| `full` (default) | FLA prefill and token decoding |
| `decode` | PyTorch prefill and FLA token decoding |
| `off` | Original PyTorch linear attention |

Look for `NPU FLA enabled on 24 linear-attention layers` to confirm activation. Transformers may print `The fast path is not available` before this message during model construction. The NPU path uses native ROIAlign and retains PyTorch causal convolution; the upstream CUDA `causal-conv1d` package is not required.

The first run may take several minutes to compile Triton kernels. Measure performance after warmup; numerical differences from the PyTorch implementation are possible. See the [NPU FLA implementation and measurements (Chinese)](npu_fla_zh.md) for details.

## Region Proposals

Most region-level tasks first require candidate bounding boxes. Due to company policy, we are unable to release the internally trained OPN referenced in our blog. The repository integrates the open-source, lightweight **WeDetect-Base-Uni** detector as an alternative proposal generator.

There are two ways to provide proposals:

1. Omit `--bbox-list`. The CLI loads WeDetect-Uni and generates 100 proposals.
2. Use any detector of your choice and pass its proposals through `--bbox-list`.

Custom boxes must use original-image pixel coordinates in `[x1, y1, x2, y2]` format:

```bash
python inference.py \
  --image-path demo/demo_image.jpg \
  --task detection \
  --text "orange; apple" \
  --bbox-list '[[x1, y1, x2, y2], [x1, y1, x2, y2]]'
```

The model supports up to 100 object-region tokens, so keep the proposal list at or below 100 boxes.

## Command-Line Usage

### Open-Vocabulary Detection

```bash
python inference.py \
  --image-path demo/demo_image.jpg \
  --task detection \
  --text "orange; apple"
```

Multiple detection categories are separated by semicolons. Add `--lang zh` when using the built-in Chinese task templates.

### Referring Expression Grounding

```bash
python inference.py \
  --image-path demo/demo_image2.jpg \
  --task grounding_single \
  --text "the person in a white shirt"
```

### Object Counting

```bash
python inference.py \
  --image-path demo/demo_image.jpg \
  --task counting \
  --text "orange"
```

### Reasoning Detection

```bash
python inference.py \
  --image-path demo/demo_image2.jpg \
  --task reasoning_detection \
  --text "the woman wearing red clothes"
```

### Region Captioning

Region captioning requires both proposals and one or more zero-based proposal indexes:

```bash
python inference.py \
  --image-path demo/demo_image.jpg \
  --task brief_region_caption \
  --bbox-list '[[x1, y1, x2, y2], [x1, y1, x2, y2]]' \
  --target-region-indexes '[1]'
```

The indexes refer to the order of the boxes supplied through `--bbox-list`.

### Region OCR

```bash
python inference.py \
  --image-path path/to/image.jpg \
  --task region_ocr \
  --bbox-list '[[40, 60, 360, 180]]' \
  --target-region-indexes '[0]'
```

### General VQA

VQA operates on the full image and does not load a proposal model:

```bash
python inference.py \
  --image-path demo/demo_image.jpg \
  --task vqa \
  --text "What fruits are in the image?"
```

## Supported Tasks

| Task | `--text` | Proposals | Additional requirement |
| --- | --- | --- | --- |
| `detection` | One or more `;`-separated categories/descriptions | Required | None |
| `grounding_single` | A referring expression | Required | None |
| `counting` | A target description | Required | None |
| `reasoning_detection` | A target description or category | Required | None |
| `brief_region_caption` | Generated from selected regions | Required | `--target-region-indexes` |
| `region_ocr` | Generated from selected regions | Required | `--target-region-indexes` |
| `vqa` | A free-form question | Not used | None |

For proposal-dependent tasks, omitting `--bbox-list` automatically invokes the proposal detector.

## Common Options

| Option | Default | Description |
| --- | --- | --- |
| `--model-path` | `resources/VLX-Seek-1.5-10B` | Local checkpoint directory or Hugging Face model identifier, e.g. `omlab/VLX-Seek-1.5-10B` |
| `--image-path` | `demo/demo_image.jpg` | Input image |
| `--lang` | `en` | Built-in task-template language: `en` or `zh` |
| `--bbox-list` | None | JSON list of pixel-coordinate proposal boxes |
| `--detector-checkpoint` | `resources/wedetect_base_uni.pth` | WeDetect-Uni checkpoint used when proposals are omitted; auto downloaded if the default path is missing |
| `--target-region-indexes` | None | JSON list of selected proposal indexes for captioning or OCR |
| `--max-new-tokens` | `2048` | Maximum completion length |
| `--temperature` | `0.0` | Sampling temperature; values above zero enable sampling. Temperature `0.0` is recommended for all detection-related tasks |
| `--no-visualize` | Off | Disable result-box visualization |
| `--visualization-output` | `<image_stem>_result.png` | Custom visualization path |

Run `python inference.py --help` for the complete CLI reference.

## Output

The command prints a result dictionary as formatted JSON:

```json
{
  "answer": "<ground>orange</ground><objects><obj0><obj2></objects>",
  "result_bbox_list": [
    {
      "object_index": "<obj0>",
      "xmin": 12.0,
      "ymin": 20.0,
      "xmax": 180.0,
      "ymax": 210.0,
      "label": "orange"
    }
  ],
  "prompt_tokens": 1234,
  "completion_tokens": 16,
  "visualization_path": "demo/demo_image_result.png"
}
```

- `answer` is the raw model response.
- `result_bbox_list` resolves generated `<objN>` references to the input proposal coordinates.
- `prompt_tokens` and `completion_tokens` report token counts.
- `visualization_path` is included when selected boxes are visualized.

VQA, captioning, or OCR answers may be free-form text and can have an empty `result_bbox_list`.

## Python API

`VLXSeekWorker` keeps the model loaded across requests. Proposal generation is handled by the CLI, so Python API callers should prepare their own boxes for region-level tasks.

```python
from PIL import Image

from vlx_seek_worker import VLXSeekWorker

worker = VLXSeekWorker(
    model_path="omlab/VLX-Seek-1.5-10B",
)
image = Image.open("demo/demo_image.jpg").convert("RGB")
boxes = [
    [12, 20, 180, 210],
    [190, 25, 430, 250],
]

detection = worker.detect(
    image,
    boxes,
    ["orange", "apple"],
    lang="en",
)

vqa = worker.predict(
    image,
    "What fruits are in the image?",
)
```

Other task wrappers are `ground`, `count`, `reasoning_detect`, `describe_region`, and `read_region_text`. Boxes are clipped to the input image bounds and must retain positive area after clipping.
