<p align="center">
  <img src="assets/logo-small.png" alt="VLX-Seek logo" width="180">
</p>

<h1 align="center">VLX-Seek</h1>

<h3 align="center">VLM 细粒度感知增强：从“坐标生成”到“区域指代”</h3>

<p align="center">
  <a href="README.md">English</a> | 中文
</p>

<p align="center">
  <a href="https://x.com/OmAI_lab">
    <img alt="X" src="https://img.shields.io/badge/%F0%9F%93%A3%20X-%E5%85%B3%E6%B3%A8%20%40OmAI_lab-000000">
  </a>
  <a href="https://www.youtube.com/@OmAILab_global">
    <img alt="YouTube" src="https://img.shields.io/badge/%F0%9F%93%A3%20YouTube-%E8%AE%A2%E9%98%85%40OmAI%20lab-FF0000">
  </a>
  <a href="https://discord.gg/c3BNhbcyd">
    <img alt="Discord" src="https://img.shields.io/badge/%F0%9F%93%A3%20Discord-%E5%8A%A0%E5%85%A5%40OmAI%20lab-5865F2">
  </a>
  <br>
  <a href="https://om-ai-lab.github.io/2026_07_06_vlx_seek_1_5_zh.html">
    <img alt="VLX-Seek 1.5 博客" src="https://img.shields.io/badge/%F0%9F%93%9D%20VLX--Seek%201.5-%E9%98%85%E8%AF%BB%E5%8D%9A%E5%AE%A2-2563eb">
  </a>
  <a href="https://huggingface.co/blog/omlab/vlx-seek">
    <img alt="Hugging Face 博客" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-%E9%98%85%E8%AF%BB%E6%96%87%E7%AB%A0-f9d54a">
  </a>
  <a href="https://huggingface.co/omlab/VLX-Seek-1.5-10B">
    <img alt="Hugging Face 模型" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-%E6%A8%A1%E5%9E%8B-f9d54a">
  </a>
  <a href="https://om-agent.cn/#/front">
    <img alt="体验页面" src="https://img.shields.io/badge/%F0%9F%9A%80%20%E4%BD%93%E9%AA%8C%E9%A1%B5%E9%9D%A2-%E7%AB%8B%E5%8D%B3%E4%BD%93%E9%AA%8C-16a34a">
  </a>
</p>

<p align="center"><sub>介绍视频：VLX-Seek 面向细粒度感知的多模态端侧模型</sub></p>

https://github.com/user-attachments/assets/74a4d0b3-bd82-4168-af4c-123029312252

VLX-Seek 是一个面向端侧具身视觉的细粒度感知视觉语言模型。它关注的不是让模型只回答“画面里有什么”，而是让模型进一步知道目标在哪里、是哪一个实例、是否符合用户描述，以及目标不存在时是否应该拒识。

不同于让语言模型直接生成边界框坐标，VLX-Seek 将定位任务改写为区域检索与区域引用问题。候选区域会被编码成区域 token，语言模型通过选择、比较和引用这些区域来完成 grounded 输出。

**🚀 立即[体验 VLX-Seek](https://om-agent.cn/)**，探索它如何理解、定位视觉世界并进行推理。

## 社区

加入 VLX 社区，与开发者交流、探索应用、分享反馈，并共同塑造多模态 AI 的未来。

<table align="center">
  <thead>
    <tr>
      <th><div align="center">官方微信</div></th>
      <th><div align="center">Discord 社区</div></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">
        <img src="assets/WeChat.png" alt="VLX 微信社群二维码" width="220">
      </td>
      <td align="center">
        <a href="https://discord.gg/c3BNhbcyd">
          <img src="assets/Discord.png" alt="VLX Discord 社区二维码" width="220">
        </a>
      </td>
    </tr>
  </tbody>
</table>

如需技术支持、商务合作或社区咨询，请通过 **[marketing@hzlh.com](mailto:marketing@hzlh.com)** 与我们联系。

## 更新

- **[2026-07-23]** 🔥🔥🔥 **VLX-Seek 1.5-10B** 推理代码与模型权重现已开源，权重下载地址：[omlab/VLX-Seek-1.5-10B](https://huggingface.co/omlab/VLX-Seek-1.5-10B)。
- **[2026-07-06]** [VLX-Seek 1.5](https://om-ai-lab.github.io/2026_07_06_vlx_seek_1_5_zh.html) 正式发布，面向具身场景带来更强的细粒度感知、更快的推理速度，以及更可靠的缺失目标拒识能力。


## 项目概览
<p align="center">
  <img src="assets/vlx_seek_model_flow_realistic_visual_sources.png" alt="VLX-Seek 总览图：视觉来源、候选区域、区域 token 与 grounded output" width="88%">
</p>
现代 VLM 在全局场景理解上已经很强，可以描述图像、回答视觉问题、理解复杂指令，并进行多模态推理。但细粒度感知需要另一类能力：

- **精确定位：** 判断目标在哪里，以及边界应如何与相邻物体区分。
- **实例区分：** 找到自然语言描述真正指向的目标实例。
- **多目标推理：** 判断有多少个目标，以及应该返回哪些区域。
- **开放词汇拒识：** 当图像中没有匹配目标时，应该回答不存在，而不是幻觉式生成框。

许多 VLM 会通过生成 `[x1, y1, x2, y2]` 这样的坐标来处理定位。但这种形式对语言模型并不稳定。坐标是较长的数字序列，多目标会进一步拉长输出，一处格式、顺序或范围错误都可能导致结果无法解析或位置明显偏离。同时，传统坐标生成方式需要输出更多 token，解码链路更长，推理效率也更低。

VLX-Seek 将任务从：

```text
图像 + 文本查询 -> 生成坐标数字 -> 解析边界框
```

转变为：

```text
图像 + 区域 token + 文本查询 -> 检索匹配区域 -> grounded answer
```

这种形式更接近 LLM 擅长的能力：比较、选择、指代、解释和推理。

## 开源模型

VLX-Seek 1.5 规划了 0.6B、3B 和 10B 三种模型规模。本仓库开源 **10B** 模型的推理代码与模型权重。

| 模型 | 权重 | 状态 |
| --- | --- | --- |
| VLX-Seek 1.5-10B | [omlab/VLX-Seek-1.5-10B](https://huggingface.co/omlab/VLX-Seek-1.5-10B) | 已开源 |



### VLX-Seek 1.5 的主要更新

- **更强的具身感知：** 扩充无人机、监控、机器人等视角和具身场景的训练数据。
- **升级视觉架构：** 更强的辅助视觉塔与 VLM 主干提升细粒度区域理解和复杂目标检测能力。
- **更快的推理：** 更快的候选区域生成链路和更多 Linear Attention 层降低推理与显存开销。
- **更少的幻觉检测：** 难负样本拒识训练和显式 `None` 格式帮助模型拒绝图像中不存在的请求目标。

架构细节、基准结果和定性案例请参阅 [VLX-Seek 1.5 博客](https://om-ai-lab.github.io/2026_07_06_vlx_seek_1_5_zh.html)。

## 环境要求

- Python 3.10+
- PyTorch（推荐 GPU）。请安装与你系统匹配的 CUDA 版本。
- 主要在 Linux 上完成测试与验证。

## 安装


```bash
git clone https://github.com/om-ai-lab/VLX-Seek.git
cd VLX-Seek
pip install -r requirements.txt
```


## 模型权重

VLX-Seek 1.5-10B 权重已发布至 Hugging Face：[omlab/VLX-Seek-1.5-10B](https://huggingface.co/omlab/VLX-Seek-1.5-10B)。

### 从 Hugging Face 直接加载

VLX-Seek 主模型无需手动下载，可直接通过 Hugging Face 模型名加载：

```bash
python inference.py \
  --model-path omlab/VLX-Seek-1.5-10B \
  --image-path demo/demo_image.jpg \
  --task detection \
  --text "橘子; 苹果" \
  --lang zh
```

首次运行会自动从 Hugging Face 下载并缓存 VLX-Seek 权重。请先完成本仓库依赖安装，模型结构由本地 `vlx_seek` 包提供。

对于检测、定位等需要候选区域的任务，如果未提供 `--bbox-list`，程序会在需要时自动从 Hugging Face下载推荐的候选区域检测器权重到默认路径 `resources/`下。也可通过 `--detector-checkpoint` 指定本地路径，或直接传入 `--bbox-list` 跳过检测器。

在 Python 中也可以这样加载：

```python
from vlx_seek_worker import VLXSeekWorker

worker = VLXSeekWorker("omlab/VLX-Seek-1.5-10B", device="cuda")
```

### 下载到本地

也可以将 VLX-Seek 和候选区域检测器权重下载到以下目录：

```text
resources/
├── VLX-Seek-1.5-10B/
└── wedetect_base_uni.pth
```

下载地址：

- VLX-Seek 1.5-10B：[omlab/VLX-Seek-1.5-10B](https://huggingface.co/omlab/VLX-Seek-1.5-10B)
- 候选区域生成模型：[WeDetect](https://huggingface.co/fushh7/WeDetect)

也可以通过 `--model-path` 和 `--detector-checkpoint` 指定其他本地路径。

### 候选区域生成模型

> **说明：** 受公司政策限制，我们无法开源blog中使用的内部训练 OPN。本仓库因此集成开源、轻量的 **WeDetect-Base-Uni** 检测器，作为生成 proposal bboxes 的替代方案。

用户也可以使用任意目标检测器，只需将其 proposals 转换为 `[x1, y1, x2, y2]` 格式的像素坐标框，并通过 `--bbox-list` 传入。仅当未提供 `--bbox-list` 时，程序才会加载内置的候选区域检测器。

## 快速开始

自动生成候选区域，运行开放词汇检测：

```bash
python inference.py \
  --image-path demo/demo_image.jpg \
  --task detection \
  --text "橘子; 苹果" \
  --lang zh
```

无需候选区域，运行通用视觉问答：

```bash
python inference.py \
  --image-path demo/demo_image.jpg \
  --task vqa \
  --text "图片中有哪些水果？"
```

检测结果默认以 JSON 打印，并可视化保存到 `<image_stem>_result.png`。所有任务、自定义 proposals、生成参数、输出字段和 Python API 请参阅**[完整推理指南](docs/inference_zh.md)**。

## 问题设定

具身和端侧视觉系统需要稳定的空间锚点。机器人、无人机、摄像头、移动设备和巡检系统往往不仅需要知道“画面里有什么”，还需要知道：

- 目标在哪里
- 指令指的是哪一个实例
- 目标是否仍然存在
- 检测的对象是否不存在
- 能否高效快速地找到目标

VLX-Seek 关注的核心问题是：

> 如何让 VLM 获得细粒度定位能力、提升推理效率，并避免让语言模型生成脆弱的坐标字符串？

VLX-Seek 的答案是：
> 把视觉区域变成语言模型可以寻址和引用的实体。

## 区域引用

VLX-Seek 将候选视觉区域建模为可寻址的区域 token。为了便于理解，可以把它们看作 `<region0>`、`<region1>` 和 `<region2>` 这样的区域索引；在模型实际使用的 special token 形式中，这些区域会以 `<obj0>`、`<obj1>` 和 `<obj2>` 表示。两种写法指向的是同一件事：每个 token 都对应图像中的一个候选视觉区域。

当用户问“找到穿红衣服的人”时，模型不需要从零开始写出四个坐标数字。它可以阅读候选区域 token，判断哪个区域最符合描述，然后输出对应的 `<obj*>` 区域引用。

例如：

```text
<ground>穿红衣服的人</ground><objects><obj2><obj5></objects>.
```

其中 `<obj2><obj5>` 是模型对第 2 个和第 5 个候选区域的实际 special token 输出形式。模型输出后，系统可以通过这些区域索引快速回查输入时对应的候选区域，并映射到实际 bbox 坐标。这种输出更短、更容易解析，也比长坐标序列更符合语言模型的工作方式。同一套机制可以支持开放目标检测、指代表达理解、区域描述、区域问答、OCR、计数和视觉推理。

## 推理流程

VLX-Seek 使用解耦的区域优先推理流程。

### 1. 候选区域生成

系统首先通过候选区域生成网络召回可能包含前景目标的候选区域。这一步负责提出可能的对象区域，而不是做最终语义判断。

候选区域生成模块与 VLM 主体解耦。实际部署中，它可以灵活地替换为其他检测器，也可以直接使用用户给定的框或视觉提示区域。

### 2. 混合细粒度区域编码器

候选框本身只是几何提示，并不能告诉语言模型区域里是什么。VLX-Seek 因此使用混合细粒度区域编码器 HFRE，将每个候选区域转换成区域级视觉表示。

HFRE 结合两条互补视觉路径：

- **语义路径：** 保留基础 VLM 的视觉语言对齐能力和高层图像理解能力。
- **细节路径：** 提供更高分辨率的局部细节、空间结构、边界、纹理和小目标信息。

SimpleFP 为 ViT 类视觉特征补足多尺度表达，使模型能够同时处理大目标和小目标。随后，区域特征会通过区域-语言连接器投影到 LLM 的嵌入空间中，让每个候选框真正变成语言模型可以读取和引用的区域 token。

### 3. 基于 Token 的推理

区域编码完成后，模型输入中同时包含全局图像 token、文本 token 和带编号的区域 token。LLM 可以基于语言查询在候选区域中进行检索，并通过区域 ID 输出 grounded 结果。

整体路径可以概括为：

1. 召回候选区域。
2. 将每个区域编码成 token。
3. 将语言查询与区域 token 匹配。
4. 输出区域引用和自然语言推理。

由于最终 grounded 结果以区域引用表示，语言模型在定位上消耗的输出 token 更少。这对端侧具身系统尤为重要：更快的解码可以降低交互感知、导航、巡检和人机交互的响应时延。

## 支持的能力

VLX-Seek 支持多种细粒度感知任务：

- **开放词汇检测：** 根据灵活文本标签查找目标。
- **指代表达理解：** 找到复杂描述对应的目标实例。
- **区域 OCR：** 读取指定视觉区域中的文字。
- **区域描述：** 对选中区域进行细粒度描述。
- **目标计数：** 先检测目标实例，再聚合计数。
- **视觉区域推理：** 使用显式区域作为多步回答的视觉证据。
- **通用 VQA：** 无需候选区域，回答关于整幅图像的自由形式问题。

## 训练策略

VLX-Seek 使用两阶段训练策略，在增强细粒度感知能力的同时，尽量保留基础 VLM 的通用能力。

### 1. 区域-语言对齐

第一阶段让模型学习区域 token 与视觉区域之间的对应关系。训练时主要冻结 VLM 主干，将学习压力集中在 HFRE、区域-语言连接器和新增特殊 token 上。

这个阶段的目标是建立基础能力：让语言模型能够把一个区域 token 当作视觉实体来读取。

### 2. 感知指令微调

第二阶段引入更丰富的感知指令，包括检测、指代表达理解、区域描述、区域推理、计数和 OCR。

这一阶段重点处理两个风险：

- **灾难性遗忘：** 混入通用 VLM 指令数据，保留图像理解、VQA、描述和推理能力。
- **幻觉式定位：** 加入负样本和拒识样本，让模型在目标不存在时回答没有匹配目标，而不是强行输出区域。

VLX-Seek 不只学习“如何找到目标”，也学习“什么时候不该找”。

## 结果对比

以下图片展示原始 VLX-Seek 3B 模型的结果。VLX-Seek 1.5-3B 和 1.5-10B 在通用识别、无人机场景、具身空间推理及目标幻觉等评测上的完整结果，请参阅 [VLX-Seek 1.5 博客](https://om-ai-lab.github.io/2026_07_06_vlx_seek_1_5_zh.html)。

<table align="center">
  <tr>
    <td width="50%" align="center">
      <img src="assets/result-ms-coco.png" alt="VLX-Seek results on MSCOCO val2017 detection benchmark" width="100%">
    </td>
    <td width="50%" align="center">
      <img src="assets/result-odinw13.png" alt="VLX-Seek results on ODinW13 open-vocabulary detection benchmark" width="100%">
    </td>
  </tr>
  <tr>
    <td width="50%" align="center">
      <img src="assets/result-refcoco.png" alt="VLX-Seek results on RefCOCO, RefCOCO+, and RefCOCOg referring expression comprehension benchmarks" width="100%">
    </td>
    <td width="50%" align="center">
      <img src="assets/result-pixmo-count.png" alt="VLX-Seek results on PixMo Count object counting benchmark" width="100%">
    </td>
  </tr>
</table>

## 为什么是 VLX-Seek

- 相比通用 VLM，VLX-Seek 显式建模候选视觉区域，可以把回答锚定到具体对象实例。

- 相比传统检测器，VLX-Seek 能利用自然语言、开放词汇语义和视觉推理，而不只是预测封闭类别。

- 相比坐标生成式 VLM，VLX-Seek 避免输出较长的数字坐标序列，改用更短、更稳定的区域引用，可降低多目标场景下的解码开销并提升响应速度。

- 相比简单外接检测头的方案，VLX-Seek 把区域变成模型内部可以读取和引用的视觉语言实体，使区域能够参与推理、比较、对话和解释。

## 技术脉络

我们团队长期深耕视觉感知领域，此前推出的 [OmDet-Turbo](https://github.com/om-ai-lab/OmDet)、[VLM-R1](https://github.com/om-ai-lab/VLM-R1) 与 [VLM-FO1](https://github.com/om-ai-lab/VLM-FO1) 等开源项目在社区获得了广泛关注和认可。VLX-Seek 汇聚并延续了这些工作在开放词汇检测、区域级理解与细粒度感知上的技术积累，是 VLX 系列的最新作品之一，后续也将持续更新迭代。
