# Ascend NPU 的 FLA 推理加速

模型加载器会在模型权重位于 NPU 且安装了可导入的 FLA 与 Ascend Triton 时，
为 Qwen3.5 的线性注意力层启用 FLA。CPU/CUDA 模型继续使用原加载路径。
未安装 FLA 时仍使用 PyTorch；训练及需要梯度的调用也使用原实现。

当前验证环境为 Ascend 910B4-1、PyTorch 2.9.0、torch_npu 2.9.0.post4、
Transformers 5.13.0、FLA/fla-core 0.5.0 和 triton-ascend 3.2.2。
需要 Ascend Triton 后端，只有常规 CUDA Triton 的环境无法运行 NPU 内核。

310P 使用单独的 AscendC ACLNN 适配器，不支持本页的 Ascend Triton 内核。
没有安装并配置 310P 自定义算子包时，继续使用 PyTorch 线性注意力；配置后，
`decode` 模式将逐 token GDN 解码切换到 AscendC，`full` 还会将预填充中的
状态更新和输出阶段切换到 AscendC，WY 分块求解仍由 PyTorch 执行。因果卷积
仍使用 PyTorch。FP16、头维度 128、稠密序列等不符合适配器限制的调用也会回退。

310P 镜像构建时会将已编译的 ACLNN 算子包放入
`/opt/ascend/custom_transformer`，并设置 `ASCEND_CUSTOM_OPP_PATH` 和
`LD_LIBRARY_PATH`。从仓库根目录构建镜像时，Docker 构建上下文只包含这份算子包，
无需挂载或复制整个 `.cache` 目录：

```bash
docker build -f docker/Dockerfile.310p -t vlx-seek:310p .
VLX_SEEK_NPU_FLA=full python inference.py --task detection --text "orange; apple"
```

若自行替换算子包，需重新构建镜像。桥接层首次启动时会编译一次 C++ 扩展，
缓存使用 PyTorch 的用户缓存目录（可用 `TORCH_EXTENSIONS_DIR` 覆盖），不写入项目
`.cache`。预填充输出算子在该版本的多任务长序列
调度下会破坏结果，因此适配器按 batch、每 64 token 分块单独调用输出算子；
该方案已在当前 310P3 环境完成数值对照，其他卡型和 CANN 版本需重新验证。
当前 310P3 实测使用默认检测示例、同一模型实例和相同候选框，预热后每种模式
运行 3 次：`off` 平均 8.954 秒，`decode` 平均 8.930 秒，`full` 平均 8.913 秒。
三种模式输出一致，均生成 19 个 token；调用计数确认 `decode` 执行 1728 次
recurrent 算子，`full` 执行 96 次 chunk 算子及 3456 次 recurrent 算子。
该样例的端到端差异小于 0.5%，不足以确认整体提速；其他输入需另行测量。
详情见[310P 推理说明](inference_zh.md#昇腾-310p)。

```bash
docker exec -w /workspace seek-dev bash scripts/run_910b.sh
```

模型加载后会出现明确的启用日志：

```text
NPU FLA enabled on 24 linear-attention layers: prefill=FLA, decode=FLA. Causal convolution uses PyTorch.
```

Transformers 在模型构造时可能先打印 `The fast path is not available`。
这是其 CUDA 检测及 causal-conv1d 检查产生的日志；本适配器随后在模型实例上
接入 FLA，以后面的 `NPU FLA enabled` 日志为准。

使用 `VLX_SEEK_NPU_FLA` 选择模式：

| 值 | 行为 |
|---|---|
| `full`（默认） | FLA 预填充和逐 token 解码 |
| `decode` | PyTorch 预填充，FLA 解码 |
| `off` | 原 PyTorch 路径，适合性能对照 |

```bash
docker exec -e VLX_SEEK_NPU_FLA=off -w /workspace seek-dev bash scripts/run_910b.sh
docker exec -e VLX_SEEK_NPU_FLA=decode -w /workspace seek-dev bash scripts/run_910b.sh
```

预填充保留 FLA 的归一化、分块求解和状态更新内核，输出阶段使用本仓库的
Ascend Triton 内核：每次处理 16 行，注意力和缓存的块大小仍是 64。
这避免了原 FLA 输出内核在 910B 上的 UB 容量溢出。支持本仓库推理使用的
非打包序列、批处理和缓存续推，不支持 prefill 的 `cu_seqlens` 打包序列。
因果卷积沿用 PyTorch；它与 FLA 线性注意力是独立的加速部分。

内核的低精度计算会产生小幅数值差异，温度为零也不保证所有输入输出逐字一致。
首次运行涉及 Triton 编译和预热，应比较后续运行耗时。

### 首次编译与缓存

完整冷缓存时，Ascend 编译器可能花数分钟编译内核。预填充和解码分别有
首次调用的开始、完成日志；开始日志会显示缓存路径。启用 FLA 的日志只
说明模型已接入内核，不表示编译已经完成。Ctrl+C 后出现的
`SIGINT`、`main process disappeared` 等错误是中断编译产生的退出错误。

910B 的 Ascend Triton 内核缓存使用 Triton 的用户缓存目录。运行时不需要挂载
项目 `.cache`；若容器被删除重建，容器内的编译缓存会随之丢失，之后首次运行
会重新编译。也可以用 `TRITON_CACHE_DIR` 指定容器内其他可写目录。更换编译器、
设备或输入形状可能触发新的编译，缓存不能保证完全消除冷启动。

```bash
TRITON_CACHE_DIR=/tmp/triton-cache VLX_SEEK_NPU_FLA=full python inference.py
```

2026-10-14 重建容器后的一次复测，复用之前中断运行已生成的部分缓存，
首次推理耗时 23.2 秒，后续约 2.3 秒；这不是完全冷缓存的编译耗时。
将缓存迁移到挂载目录并修正缓存索引后，独立进程复测的首次推理为
16.4 秒，后续约 2.2 秒，答案一致。上述推理计时均不包含模型加载时间。

在上述环境中，对 `demo/demo_image.jpg` 的 `orange; apple` 检测进行同一模型
实例的对照：固定候选框、温度 0、生成 19 个 token，各模式预热后运行 5 次。

| 模式 | 平均耗时 |
|---|---:|
| `off` | 2.291 秒 |
| `decode` | 2.109 秒 |
| `full` | 2.027 秒 |

`full` 耗时降低约 11.5%，该样例三种模式的答案一致。调用计数确认每次完整
检测实际进入 24 次 FLA prefill 和 432 次 FLA decode。其他输入、输出长度
及设备上的性能需重新测量。

另一次加载 WeDetect 候选框检测器的同进程对照，依次运行
`off → decode → full → off`，每组去掉首次预热后取 4 次平均：
`full` 为 2.244 秒，最后一组 `off` 为 2.428 秒，耗时降低约 7.6%，
答案一致。但分别启动独立进程运行日常 `inference.py` 命令时，
`off` 和 `full` 的预热后平均耗时均约 2.32 秒，未观察到明确的整体收益。
这些结果确认 FLA 内核已经生效，但整体提速受运行波动和其他模块耗时影响，
不能保证日常命令固定获得上述比例的加速。

```bash
docker exec -w /workspace seek-dev python -m unittest discover -s tests -v
```
