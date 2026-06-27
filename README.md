<p align="center">
  <img src="assets/logo-small.png" alt="VLX-Seek logo" width="180">
</p>

<h1 align="center">VLX-Seek</h1>

<h3 align="center">Fine-Grained Perception VLM: From Coordinate Generation to Region Reference</h3>

<p align="center">
  English | <a href="README_zh.md">中文</a>
</p>

<p align="center">
  <a href="https://x.com/OmAI_lab">
    <img alt="X" src="https://img.shields.io/badge/%F0%9F%93%A3%20X-Follow%20%40OmAI_lab-000000">
  </a>
  <a href="">
    <img alt="Blog" src="https://img.shields.io/badge/%F0%9F%93%9D%20Blog-Read%20Article-2563eb">
  </a>
  <!-- <a href="https://platform.om-agent.cn/subapp-index/#/front"> -->
    <img alt="Demo coming soon" src="https://img.shields.io/badge/%F0%9F%9A%80%20Demo-Coming%20Soon-94a3b8">
  </a>
  <a href="">
    <img alt="Hugging Face blog" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Read%20Blog-f9d54a">
  </a>
</p>

<p align="center"><sub>Overview video: VLX-Seek for edge multimodal fine-grained perception</sub></p>

<p align="center">
  <video
    controls
    muted
    playsinline
    preload="metadata"
    width="88%"
    poster="assets/vlx_seek_en_cover.jpg"
  >
    <source src="https://airesources.oss-cn-hangzhou.aliyuncs.com/Demo/VLX-Seek/vlx_seek_en.mp4" type="video/mp4">
    Your browser does not support video playback. Please
    <a href="https://airesources.oss-cn-hangzhou.aliyuncs.com/Demo/VLX-Seek/vlx_seek_en.mp4">watch the video</a>.
  </video>
</p>


<p align="center">
  <img src="assets/vlx_seek_model_flow_realistic_visual_sources.png" alt="VLX-Seek overview: visual sources, region proposals, region tokens, and grounded output" width="88%">
</p>

VLX-Seek is a fine-grained perception vision-language model for edge-side embodied vision. It is designed for scenarios where a model must not only understand what is in an image, but also identify where the relevant objects are, which instance is being referred to, and when a target does not exist.

Instead of asking the language model to directly generate bounding-box coordinates, VLX-Seek reformulates localization as region retrieval and region reference. Candidate regions are encoded as addressable region tokens, and the language model answers by selecting, comparing, and referring to those regions.

## Overview

Modern VLMs are strong at global scene understanding. They can describe images, answer visual questions, follow complex instructions, and perform multimodal reasoning. However, fine-grained perception requires a different level of visual grounding:

- **Precise localization:** where the target is and how its boundary should be separated from nearby objects.
- **Instance distinction:** which object instance matches a language description.
- **Multi-object reasoning:** how many targets exist and which regions should be returned.
- **Open-vocabulary rejection:** when no region matches the query, the model should say so instead of hallucinating a box.

Many VLMs handle localization by generating coordinates such as `[x1, y1, x2, y2]`. This format is brittle for language models. Coordinates are long numeric sequences, multiple objects multiply the output length, and small formatting or ordering errors can make the result invalid or inaccurate.

VLX-Seek changes the task from:

```text
image + text query -> generate coordinate numbers -> parse boxes
```

to:

```text
image + region tokens + text query -> retrieve matching regions -> grounded answer
```

This makes localization closer to what LLMs already do well: compare, select, refer, explain, and reason over explicit entities.

## Open-Source Models

Model weights will be released soon.

## Problem Setting

Embodied and edge-side visual systems need stable spatial anchors. Monitoring cameras, drones, robots, robot dogs, mobile devices, and inspection systems often need to know not only "what is visible", but also:

- where the object is
- which instance the instruction refers to
- whether the target is still present
- how the target relates to nearby objects
- whether the requested object is absent

For these devices, fine-grained grounding is not just an offline benchmark. A monitoring camera may need to locate a person entering a restricted area, a drone may need to find a small target from an aerial view, a robot may need to grasp or avoid a specific object, and a robot dog may need to follow, inspect, or navigate around a referenced target under limited compute and power budgets.

VLX-Seek focuses on a core question:

> How can VLMs acquire fine-grained localization capabilities, improve reasoning efficiency, and avoid requiring language models to generate fragile coordinate strings?

VLX-Seek’s answer is:

> Turn visual regions into entities that the language model can address and refer to.

## Region Reference

VLX-Seek models candidate visual regions as addressable region tokens. Conceptually, these can be understood as region indices such as `<region0>`, `<region1>`, and `<region2>`; in the model's actual special-token format, they are serialized as `<obj0>`, `<obj1>`, and `<obj2>`. Both forms refer to the same idea: each token corresponds to a candidate visual region in the image.

When a user asks "Find the people wearing red", the model does not need to write four numeric coordinates from scratch. It can inspect the candidate region tokens, identify the regions that best match the description, and output the corresponding `<obj*>` region references.

For example:

```text
<ground>people wearing red</ground><object><obj2><obj5></object>.
```

Here, `<obj2><obj5>` is the model's concrete special-token output for the selected candidate regions. After generation, the system can use these region indices to quickly look up the corresponding input proposals and map them back to actual bbox coordinates. This output is compact, easier to parse, and better aligned with language-model behavior than long coordinate sequences. It is also faster to decode: for multiple targets, VLX-Seek only needs to emit short region IDs instead of full `[x1, y1, x2, y2]` coordinate tuples for every object. The same mechanism can support open-vocabulary detection, referring expression comprehension, region captioning, region VQA, OCR, counting, and visual reasoning.

## Inference Pipeline

VLX-Seek uses a decoupled region-first pipeline.

### 1. Region Proposal

The system first recalls candidate foreground regions with a candidate region generation network. This stage is responsible for proposing possible object regions, not for making the final semantic decision.

The proposal module is decoupled from the VLM backbone. In practical deployments, it can be flexibly replaced by another detector, user-provided boxes, or visual prompts.

### 2. Hybrid Fine-Grained Region Encoder

Candidate boxes are geometric hints; by themselves they do not tell the language model what is inside each region. VLX-Seek therefore uses a Hybrid Fine-Grained Region Encoder (HFRE) to convert each candidate region into a region-level visual representation.

HFRE combines two complementary visual pathways:

- **Semantic pathway:** preserves the base VLM's visual-language alignment and high-level image understanding.
- **Detail pathway:** provides higher-resolution local details, spatial structure, boundaries, textures, and small-object cues.

SimpleFP adds multi-scale visual features for ViT-style representations, helping the model handle both large objects and small local targets. Region features are then projected into the LLM embedding space through a region-language connector, so each candidate box becomes a readable and referable region token.

### 3. Token-Based Reasoning

After region encoding, the model receives global image tokens, text tokens, and numbered region tokens. The LLM can then perform language-conditioned retrieval over the candidate regions and produce grounded output by referencing region IDs.

This makes the inference path clear:

1. Recall candidate regions.
2. Encode each region as a token.
3. Match language queries to region tokens.
4. Output region references and natural-language reasoning.

Because the final grounded answer is expressed as region references, the language model spends fewer output tokens on localization. This matters on edge-side embodied systems, where faster decoding can reduce response latency for interactive perception, navigation, inspection, and human-robot interaction.

## Supported Capabilities

VLX-Seek supports a broad set of region-centric perception tasks:

- **Open-vocabulary detection:** find targets described by flexible text labels.
- **Referring expression comprehension:** identify the instance that matches a complex description.
- **Region OCR:** read text from selected visual regions.
- **Region VQA:** answer questions about a specified region.
- **Region captioning:** describe selected areas in detail.
- **Object counting:** count instances by detecting and aggregating regions.
- **Visual region reasoning:** use explicit regions as evidence for multi-step answers.

## Training Strategy

VLX-Seek uses a two-stage training strategy to add fine-grained perception while preserving the VLM's general capability.

### 1. Region-Language Alignment

The first stage teaches the model how region tokens correspond to visual regions. The main VLM backbone is largely frozen, while HFRE, the region-language connector, and newly added special tokens are trained to align region features with the LLM embedding space.

This stage establishes the basic ability to read a region token as a visual entity.

### 2. Perception Instruction Tuning

The second stage introduces richer perception instructions, including detection, referring expression comprehension, region captioning, region reasoning, counting, and OCR.

Two risks are handled during this stage:

- **Catastrophic forgetting:** general VLM instruction data is mixed in to preserve broad image understanding, VQA, captioning, and reasoning ability.
- **Hallucinated localization:** negative and rejection samples teach the model to answer that no matching target exists instead of forcing a region output.

The goal is not only to teach the model how to find objects, but also when not to find them.

## Results Comparison

The 3B version of VLX-Seek shows strong performance across fine-grained perception tasks, outperforming many larger open-source and closed-source models.

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

## Why VLX-Seek

- Compared with a general VLM, VLX-Seek explicitly models candidate visual regions and can ground answers to object instances.

- Compared with a traditional detector, VLX-Seek can use natural language, open-vocabulary semantics, and visual reasoning rather than only predicting closed-set categories.

- Compared with coordinate-generation VLMs, VLX-Seek avoids long numeric box outputs and uses shorter, more stable region references, which can reduce decoding cost and improve response speed in multi-object scenarios.

- Compared with simply attaching a detector head, VLX-Seek makes regions internal visual-language entities that can participate in reasoning, comparison, dialogue, and explanation.

## Technology Lineage

Our team has spent years building in visual perception, with open-source projects such as [OmDet-Turbo](https://github.com/om-ai-lab/OmDet), [VLM-R1](https://github.com/om-ai-lab/VLM-R1), and [VLM-FO1](https://github.com/om-ai-lab/VLM-FO1) receiving strong attention and recognition from the community. VLX-Seek brings together and extends these accumulated strengths in open-vocabulary detection, region-level understanding, and fine-grained perception as one of the latest works in the VLX series, which will continue to evolve through future updates.

## Follow us

Follow Om AI Lab on [X](https://x.com/OmAI_lab), or scan the WeChat group QR code below for VLX updates and discussion.

<p align="left">
  <img src="assets/qrcode_0627.jpg" alt="WeChat community QR code" width="200">
</p>
