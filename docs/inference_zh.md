# VLX-Seek 1.5 推理指南

[English](inference.md) | 中文

本文介绍 VLX-Seek 1.5-10B 的命令行与 Python 推理方式。模型简介和安装方法请参阅[项目 README](../README_zh.md)。

## 环境要求

当前提供的推理流程需要：

- 推理依赖，可通过 `requirements.txt` 在本地安装，或使用下文提供的 Docker 镜像
- VLX-Seek 1.5-10B 模型权重：[omlab/VLX-Seek-1.5-10B](https://huggingface.co/omlab/VLX-Seek-1.5-10B)
- WeDetect-Base-Uni 模型权重：[WeDetect](https://huggingface.co/fushh7/WeDetect)

可直接通过 Hugging Face 模型名加载 VLX-Seek 主模型：

```bash
python inference.py --model-path omlab/VLX-Seek-1.5-10B ...
```

未提供 `--bbox-list` 时，程序会在需要时自动把 `wedetect_base_uni.pth` 下载到 `resources/wedetect_base_uni.pth`。

也可以将权重下载到本地默认目录：

```text
resources/
├── VLX-Seek-1.5-10B/
└── wedetect_base_uni.pth
```

`--model-path` 既支持 Hugging Face 模型标识符，也支持本地 checkpoint 目录。模型结构由本仓库中的 `vlx_seek` 包提供。

也可以通过 `--model-path` 和 `--detector-checkpoint` 指定其他路径。

### 硬件选择

命令行和 `VLXSeekWorker` 共用环境变量 `VLX_SEEK_BACKEND` 选择硬件：

| 值 | 设备 |
| --- | --- |
| `auto`（默认） | 优先检测 NVIDIA CUDA，再检测昇腾 NPU、Apple MPS，否则使用 CPU |
| `nvidia` | NVIDIA CUDA GPU |
| `ascend` / `310p` | 昇腾 NPU，已在 910B 和 310P3 上验证 |
| `cpu` | CPU，使用 SDPA 注意力 |
| `apple` / `mps` | Apple Silicon GPU，使用 MPS 和 SDPA 注意力 |

```bash
VLX_SEEK_BACKEND=nvidia python inference.py --task detection --text "orange; apple"
VLX_SEEK_BACKEND=ascend python inference.py --task detection --text "orange; apple"
VLX_SEEK_BACKEND=apple python inference.py --task detection --text "orange; apple"
```

显式传入 `--device cuda:1`、`--device npu:0`、`--device mps` 或 Python API 的 `device=` 参数时，会覆盖环境变量。指定的硬件不可用时会报错，不会自动切换到其他硬件。选择 NVIDIA/CPU/MPS 时不需要 NPU 依赖，也不会改写 CUDA API。

### Apple Silicon（macOS）

MPS 只能在 macOS 宿主机上使用。Docker 容器无法访问宿主机的 MPS 驱动，
以下安装和推理命令均需直接在 macOS 终端中执行。

建议从[清华 Miniforge 镜像源](https://mirrors-i.tuna.tsinghua.edu.cn/github-release/conda-forge/miniforge/LatestRelease/Miniforge3-26.7.2-0-MacOSX-arm64.sh)
下载 Apple Silicon 安装脚本。下载后，在脚本所在目录执行：

```bash
bash Miniforge3-26.7.2-0-MacOSX-arm64.sh
```

按安装提示初始化 conda，然后重新打开终端。在仓库根目录创建环境，
并通过清华 PyPI 源安装 macOS 依赖，其中不包含仅适用于 CUDA 的 Flash Attention：

```bash
conda create -n vlx-seek python=3.12
conda activate vlx-seek
pip install -r requirements_mac.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

在已激活的环境中运行推理：

```bash
bash scripts/run_mac.sh
```

MPS 默认使用 FP16 模型权重、SDPA 注意力和 PyTorch 线性注意力实现，候选框检测器也在 MPS 上运行。由于 torchvision 没有原生 MPS ROIAlign 内核，该算子以 FP32 在 CPU 上计算，然后将特征传回 MPS。此路径无需安装 Flash Attention、Triton 或 CUDA FLA 内核。

### 可选：加速推理

模型使用了 Linear Attention 层。如果缺少对应的加速内核，会看到类似如下警告，并回退到较慢的 PyTorch 实现：

```text
The fast path is not available because one of the required library is not installed. Falling back to torch implementation.
```

在 NVIDIA GPU 上，安装 [flash-linear-attention](https://github.com/fla-org/flash-linear-attention#installation) 和 [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) 可以启用快速路径。NPU 加速方式请参阅下文的[昇腾 310P](#昇腾-310paarch64-docker-推理)和[910B](#昇腾-910baarch64docker-推理)说明。

## NVIDIA x86（如 A100）Docker 推理

支持在搭载 A100 等 NVIDIA GPU 的 Linux x86_64 主机上使用 Docker 推理。宿主机需要安装 NVIDIA 驱动、Docker，并配置 NVIDIA Container Toolkit 以供容器访问 GPU。镜像已包含推理依赖，以及 `flash-linear-attention` 和 `causal-conv1d` 加速库。

宿主机 NVIDIA 驱动要求：

| x86 GPU 类型 | 最低 Driver Version |
| --- | --- |
| A100 等数据中心卡 | 535 及以上 |
| RTX 4090 等消费级卡 | 580 及以上 |

### 准备镜像

如果尚未克隆仓库，先执行以下命令。后续宿主机命令均在仓库根目录运行：

```bash
git clone https://github.com/om-ai-lab/VLX-Seek.git
cd VLX-Seek
```

任选一种方式准备镜像：

- 使用 [Dockerfile.nvidia_x86](../docker/Dockerfile.nvidia_x86) 构建：

  ```bash
  docker build -f docker/Dockerfile.nvidia_x86 -t vlx-seek:py3.12-ubuntu24.04-nvidia-x86_64 .
  ```

- 拉取我们提供的预构建镜像：

  ```bash
  docker pull registry.linker.cc/vlx/vlx-seek:py3.12-ubuntu24.04-nvidia-x86_64
  ```

### 启动并进入容器

在仓库根目录执行以下命令，通过 `--gpus device=0` 选择宿主机 GPU `0`，可修改编号使用其他 GPU。如使用本地构建的镜像，将 `registry.linker.cc/vlx/vlx-seek:py3.12-ubuntu24.04-nvidia-x86_64` 替换为 `vlx-seek:py3.12-ubuntu24.04-nvidia-x86_64`。

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

上述命令将仓库挂载到 `/workspace` 并将其设为工作目录，宿主机上的源码、本地权重、输入图片和保存的结果均可在容器内访问。容器会持续运行，后续可通过 `docker exec -it seek-dev bash` 再次进入。

### 在容器内执行推理

```bash
bash scripts/run_nvidia_x86.sh
```

首次运行需要访问 Hugging Face，以下载主模型，并在需要候选框时下载 `resources/wedetect_base_uni.pth`。如使用本地主模型权重，将其放在宿主机仓库的 `resources/VLX-Seek-1.5-10B/` 目录，再将参数改为 `--model-path resources/VLX-Seek-1.5-10B`。Docker 镜像不包含模型权重。

结果以 JSON 打印，示例可视化保存到宿主机挂载目录中的 `demo/demo_image_result.png`。下文其他命令行示例和 Python API 也可在该容器的 `/workspace` 目录下运行。

## T5000 和 GB10（aarch64）Docker 推理

宿主机平台要求：

| 平台 | 要求 |
| --- | --- |
| Jetson 系列（T5000） | JetPack 7.0 及以上，建议使用 JetPack 7.2 |
| NVIDIA Spark GB10 | NVIDIA Driver Version 580 及以上 |

根据 aarch64 平台选择对应的预构建镜像：

| 平台 | 镜像 |
| --- | --- |
| T5000 | `registry.linker.cc/vlx/vlx-seek:py3.12-ubuntu24.04-t5000-aarch64` |
| GB10 | `registry.linker.cc/vlx/vlx-seek:py3.12-ubuntu24.04-gb10-aarch64` |

宿主机需要安装 NVIDIA 驱动、Docker 及 NVIDIA Container Toolkit，并配置 `nvidia` runtime。按上文克隆仓库，以下宿主机命令均在仓库根目录运行。

### T5000 Docker Run

在仓库根目录启动并进入容器：

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

在仓库根目录启动并进入容器：

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

本地没有镜像时，`docker run` 会自动拉取对应镜像。两条命令均将仓库挂载到 `/workspace`，将容器端口 `8000` 映射到宿主机端口 `1233`，并使用容器名 `seek-dev`。切换平台前请先停止并移除同名容器。

进入对应容器后，在 `/workspace` 下分别运行 `bash scripts/run_t5000.sh` 或
`bash scripts/run_spark_gb10.sh`。模型下载和本地权重路径的使用方式与
NVIDIA x86 小节相同。

## 昇腾 310P（aarch64）Docker 推理

Linux aarch64 主机可使用预构建的 310P 镜像：
`registry.linker.cc/vlx/vlx-seek:py3.11-openeuler24.03-310p-aarch64`。
镜像已包含 `/opt/ascend/custom_transformer` 下的 AscendC GDN 算子包。
宿主机需安装 Docker 和可用的昇腾驱动。在仓库根目录拉取镜像并启动容器
（无需挂载 `.cache`）：

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

如需使用其他 NPU，将 `/dev/davinci0` 替换为 `npu-smi info` 显示的设备编号。
仓库挂载用于提供源码和保存结果；Python 依赖及 310P 算子包由镜像提供。

进入容器后，在 `/workspace` 下执行推理：

```bash
cd /workspace
bash scripts/run_310p.sh
```

程序按实际芯片型号自动选择兼容路径：310P 使用 FP16 权重和 SDPA 注意力。
上述镜像已包含 AscendC GDN 算子包，无需挂载 `.cache` 或在运行时 source
`set_env.bash`。在该镜像中，`full` 模式会对支持的 GDN 调用使用 AscendC；
不含算子包的镜像会回退到 PyTorch 线性注意力。构建方式见
[310P FLA 配置](npu_fla_zh.md#ascend-npu-的-fla-推理加速)。

`torchvision::nms` 的 CPU 回退警告不影响运行。首次推理可能包含算子编译，
应将首次耗时与预热后的耗时分开比较。此处验证的是推理流程，未验证 310P 训练。

在 310P3、CANN 9.0.1、PyTorch 2.9.0、torch_npu 2.9.0.post4、
Transformers 5.13.0 环境实测：默认示例图检测连续运行 10 次，最终识别出
5 个橘子和 2 个苹果，生成 19 个 token。首轮约 59.4 秒，后续每轮约 9.3 秒，
计时不含模型加载及 WeDetect 候选框生成。这些样例不代表其他输入的精度或性能保证。

#### 双卡张量并行（TP=2）

当前机器的两个 310P3 逻辑设备已通过 HCCL AllReduce / AllGather 和双卡检测验证。
容器内使用仓库的 310P 脚本启动双进程检测：

```bash
cd /workspace
TP_SIZE=2 bash scripts/run_310p.sh
```

也可以直接用 `torchrun` 启动检测：

```bash
torchrun --standalone --nproc-per-node=2 inference.py \
  --tp-size 2 --task detection --text "orange; apple"
```

`--tp 2` 是 `--tp-size 2` 的别名。两个进程分别绑定 `npu:0` 和 `npu:1`，
不要指定固定的 `--device npu:0`。不指定 TP 时保持单卡运行。
文本模型使用 Transformers 自带的 TP 方案切分注意力和 MLP 投影，输出层也切分；
视觉编码器、词嵌入和线性注意力的部分计算仍在每张卡上保留。
因此显存占用和耗时不会简单减半。310P 仍使用 FP16 和 SDPA；GDN 按上述配置
根据镜像内是否包含算子包使用 AscendC 或回退到 PyTorch。
WeDetect 只在 rank 0 运行，候选框通过 CPU Gloo 通信同步，结果和可视化也只由
rank 0 输出。其他 310P 板卡组合需先确认 HCCL 通信支持。

同一检测示例的 TP=2 实测共运行 10 轮，首轮约 24.0 秒，后续 9 轮平均
5.946 秒，最终答案和目标框与此前单卡样例相同，均生成 19 个 token。
上述计时不含权重加载和 WeDetect 候选框生成；此前单卡预热后约 9.3 秒。
两次测试的编译缓存状态不同，首次耗时不宜直接比较，其他任务需另行测量。

Python API 同样支持 `VLXSeekWorker(model_path, tp_size=2)`，脚本必须用
`torchrun --standalone --nproc-per-node=2` 启动，且两个进程需按相同顺序调用
推理 API，并传入相同图片、问题、候选框及生成参数。随机采样的种子会在两卡间同步。
退出时可调用 `vlx_seek.parallel.shutdown_tensor_parallel()` 释放进程组。

## 昇腾 910B（aarch64）Docker 推理

支持在搭载昇腾 910B NPU 的 Linux aarch64 主机上使用预构建镜像推理。宿主机需安装 Docker 和可用的昇腾驱动，可通过 `npu-smi info` 检查设备。镜像包含 CANN 9.0.1、PyTorch 2.9.0、`torch_npu` 2.9.0.post4、Transformers 5.13.0、Ascend Triton 和 NPU FLA 依赖，当前已在 910B4-1 上验证。

### 准备镜像并启动容器

按上文克隆仓库，然后在仓库根目录执行：

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

根据 `npu-smi info` 选择宿主机设备，将 `/dev/davinci0` 改为实际使用的编号，例如 `/dev/davinci2`。如果宿主机驱动安装目录不同，请相应调整挂载路径。各平台示例均使用容器名 `seek-dev`，已有同名容器时请使用其他名称。

### 在容器内执行推理

在 `/workspace` 下运行：

```bash
bash scripts/run_910b.sh
```

昇腾后端会导入 `torch_npu`，并将候选框检测器和 VLX-Seek 显式放置到 NPU 上，不会全局改写 CUDA API。镜像不包含模型权重，首次运行会下载主模型及需要的候选框检测器。如使用本地权重，将参数改为 `--model-path resources/VLX-Seek-1.5-10B`。结果和可视化文件保存到宿主机挂载的仓库目录，使用方式与上文一致。

依赖可用时，默认启用 NPU FLA，可通过环境变量切换：

| `VLX_SEEK_NPU_FLA` | 行为 |
| --- | --- |
| `full`（默认） | FLA 预填充和逐 token 解码 |
| `decode` | PyTorch 预填充，FLA 解码 |
| `off` | 原 PyTorch 线性注意力 |

以 `NPU FLA enabled on 24 linear-attention layers` 日志确认启用状态。Transformers 可能在模型构造时先打印 `The fast path is not available`。NPU 路径使用原生 ROIAlign，因果卷积继续使用 PyTorch，无需安装上游 CUDA 版 `causal-conv1d`。

首次运行可能需要数分钟编译 Triton 内核，请在预热后测量性能。FLA 与 PyTorch 实现也可能存在小幅数值差异。实现细节和测量结果见 [NPU FLA 说明](npu_fla_zh.md)。

## 候选区域

大多数区域级任务需要先准备候选框。受公司政策限制，我们无法开源blog中使用的内部训练 OPN。本仓库集成开源、轻量的 **WeDetect-Base-Uni** 检测器，作为候选区域生成的替代方案。

有两种方式可以提供 proposals：

1. 不传 `--bbox-list`，由命令行程序加载 WeDetect-Uni 并生成 100 个候选框。
2. 使用任意检测器生成候选框，再通过 `--bbox-list` 传入。

自定义候选框应使用原图像素坐标，格式为 `[x1, y1, x2, y2]`：

```bash
python inference.py \
  --image-path demo/demo_image.jpg \
  --task detection \
  --text "橘子; 苹果" \
  --lang zh \
  --bbox-list '[[x1, y1, x2, y2], [x1, y1, x2, y2]]'
```

模型最多支持 100 个目标区域 token，因此 proposals 数量应不超过 100。

## 命令行用法

### 开放目标检测

```bash
python inference.py \
  --image-path demo/demo_image.jpg \
  --task detection \
  --text "橘子; 苹果" \
  --lang zh
```

多个检测类别使用分号分隔。使用内置中文任务模板时，请添加 `--lang zh`。

### 指代表达定位

```bash
python inference.py \
  --image-path demo/demo_image2.jpg \
  --task grounding_single \
  --text "穿白色上衣的人" \
  --lang zh
```

### 目标计数

```bash
python inference.py \
  --image-path demo/demo_image.jpg \
  --task counting \
  --text "橘子" \
  --lang zh
```

### 推理检测

```bash
python inference.py \
  --image-path demo/demo_image2.jpg \
  --task reasoning_detection \
  --text "穿着红色衣服的女人" \
  --lang zh
```

### 区域描述

区域描述需要同时提供 proposals 和一个或多个从零开始的区域索引：

```bash
python inference.py \
  --image-path demo/demo_image.jpg \
  --task brief_region_caption \
  --bbox-list '[[x1, y1, x2, y2], [x1, y1, x2, y2]]' \
  --target-region-indexes '[1]' \
  --lang zh
```

索引对应 `--bbox-list` 中的输入顺序。

### 区域 OCR

```bash
python inference.py \
  --image-path path/to/image.jpg \
  --task region_ocr \
  --bbox-list '[[40, 60, 360, 180]]' \
  --target-region-indexes '[0]' \
  --lang zh
```

### 通用 VQA

VQA 直接处理整幅图像，不会加载候选区域模型：

```bash
python inference.py \
  --image-path demo/demo_image.jpg \
  --task vqa \
  --text "图片中有哪些水果？"
```

## 支持的任务

| 任务 | `--text` 内容 | 是否需要 proposals | 额外要求 |
| --- | --- | --- | --- |
| `detection` | 一个或多个以 `;` 分隔的类别或描述 | 是 | 无 |
| `grounding_single` | 指代表达 | 是 | 无 |
| `counting` | 目标描述 | 是 | 无 |
| `reasoning_detection` | 目标描述或类别 | 是 | 无 |
| `brief_region_caption` | 根据选中区域自动生成 | 是 | `--target-region-indexes` |
| `region_ocr` | 根据选中区域自动生成 | 是 | `--target-region-indexes` |
| `vqa` | 自由形式问题 | 否 | 无 |

对于依赖 proposals 的任务，如果省略 `--bbox-list`，程序会自动调用候选框检测器生成。

## 常用参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model-path` | `resources/VLX-Seek-1.5-10B` | 本地权重目录或 Hugging Face 模型标识符，例如 `omlab/VLX-Seek-1.5-10B` |
| `--image-path` | `demo/demo_image.jpg` | 输入图像 |
| `--lang` | `en` | 内置任务模板语言：`en` 或 `zh` |
| `--bbox-list` | 无 | 像素坐标 proposals 的 JSON 列表 |
| `--detector-checkpoint` | `resources/wedetect_base_uni.pth` | 未提供 proposals 时使用的 WeDetect-Uni 权重；默认路径缺失时会自动下载 |
| `--target-region-indexes` | 无 | 区域描述或 OCR 选中的 proposal 索引 JSON 列表 |
| `--max-new-tokens` | `2048` | 最大生成长度 |
| `--temperature` | `0.0` | 采样温度；大于零时启用采样。建议检测相关任务均使用温度 `0.0` |
| `--no-visualize` | 关闭 | 禁用检测结果可视化 |
| `--visualization-output` | `<image_stem>_result.png` | 自定义可视化输出路径 |

运行 `python inference.py --help` 可以查看完整命令行参数。

## 输出格式

命令最终会以格式化 JSON 打印结果字典：

```json
{
  "answer": "<ground>橘子</ground><objects><obj0><obj2></objects>",
  "result_bbox_list": [
    {
      "object_index": "<obj0>",
      "xmin": 12.0,
      "ymin": 20.0,
      "xmax": 180.0,
      "ymax": 210.0,
      "label": "橘子"
    }
  ],
  "prompt_tokens": 1234,
  "completion_tokens": 16,
  "visualization_path": "demo/demo_image_result.png"
}
```

- `answer` 是模型的原始回答。
- `result_bbox_list` 将生成的 `<objN>` 引用解析为输入 proposal 坐标。
- `prompt_tokens` 和 `completion_tokens` 是输入与输出 token 数量。
- 对选中目标进行可视化时，结果中会包含 `visualization_path`。

VQA、区域描述或 OCR 可以返回自由文本，其 `result_bbox_list` 可能为空。

## Python API

`VLXSeekWorker` 可以在多次请求之间保持模型常驻。候选区域生成由命令行入口负责，因此 Python API 调用方需要为区域级任务自行准备候选框。

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
    ["橘子", "苹果"],
    lang="zh",
)

vqa = worker.predict(
    image,
    "图片中有哪些水果？",
)
```

其他任务封装包括 `ground`、`count`、`reasoning_detect`、`describe_region` 和 `read_region_text`。候选框会被裁剪到输入图像边界，裁剪后必须保持有效的正面积。
