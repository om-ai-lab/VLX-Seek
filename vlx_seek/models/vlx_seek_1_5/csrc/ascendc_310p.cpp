// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Adapted from vllm-project/vllm-ascend (Apache-2.0), revision
// 63a5c0834137c3eded23ac1345fe5547095ae6ef, csrc/torch_binding.cpp and
// attention/recurrent_gated_delta_rule_v310/recurrent_gated_delta_rule_310_torch_adpt.h.
// Licensed under the Apache License, Version 2.0.
// https://www.apache.org/licenses/LICENSE-2.0
// Minimal ACLNN bridge; no vLLM runtime or Triton dependency.
#include <ATen/ATen.h>
#include <torch/library.h>
#include "op_plugin/utils/op_api_common_base.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"

namespace {
constexpr int64_t CHUNK_SIZE = 64;

at::Tensor recurrent(const at::Tensor& q, const at::Tensor& k,
                     const at::Tensor& v, const at::Tensor& beta,
                     at::Tensor& state, const at::Tensor& lengths,
                     const at::Tensor& indices, const at::Tensor& g, double scale) {
    c10_npu::NPUGuard guard(q.device());
    auto out = at::empty(v.sizes(), v.options());
    c10::optional<at::Tensor> absent;
    float scale_value = static_cast<float>(scale);
    EXEC_NPU_CMD_EXT(aclnnRecurrentGatedDeltaRuleV310, q, k, v, beta,
                     state, lengths, indices, g, absent, absent, scale_value, out);
    return out;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> chunk_h(
        const at::Tensor& k, const at::Tensor& w, const at::Tensor& u,
        const at::Tensor& g, const at::Tensor& state) {
    c10_npu::NPUGuard guard(k.device());
    auto h = at::zeros({k.size(0), u.size(1), k.size(2) / CHUNK_SIZE,
                        k.size(3), u.size(3)}, k.options());
    auto v_new = at::zeros(u.sizes(), u.options());
    auto final_state = at::empty(state.sizes(), state.options());
    c10::optional<at::Tensor> absent;
    c10::optional<at::IntArrayRef> absent_indices;
    bool output_final_state = true, save_new_value = true;
    bool use_exp2 = false, transpose_state_layout = false;
    EXEC_NPU_CMD_EXT(aclnnChunkGatedDeltaRuleFwdHVllm,
                     k, w, u, g, absent, state, output_final_state, CHUNK_SIZE,
                     save_new_value, absent_indices, absent_indices, use_exp2,
                     transpose_state_layout, h, v_new, final_state);
    return {h, v_new, final_state};
}

at::Tensor chunk_o(const at::Tensor& q, const at::Tensor& k,
                   const at::Tensor& v, const at::Tensor& h,
                   const at::Tensor& g, double scale) {
    c10_npu::NPUGuard guard(q.device());
    auto out = at::zeros(v.sizes(), v.options());
    c10::optional<at::IntArrayRef> absent;
    EXEC_NPU_CMD_EXT(aclnnChunkFwdOVllm, q, k, v, h, g, absent, absent,
                     scale, CHUNK_SIZE, out);
    return out;
}
}  // namespace

TORCH_LIBRARY(vlx_seek_310p, ops) {
    ops.def("recurrent(Tensor q, Tensor k, Tensor v, Tensor beta, Tensor(a!) state, Tensor lengths, Tensor indices, Tensor g, float scale) -> Tensor");
    ops.impl("recurrent", c10::DispatchKey::PrivateUse1, &recurrent);
    ops.def("chunk_h(Tensor k, Tensor w, Tensor u, Tensor g, Tensor state) -> (Tensor, Tensor, Tensor)");
    ops.impl("chunk_h", c10::DispatchKey::PrivateUse1, &chunk_h);
    ops.def("chunk_o(Tensor q, Tensor k, Tensor v, Tensor h, Tensor g, float scale) -> Tensor");
    ops.impl("chunk_o", c10::DispatchKey::PrivateUse1, &chunk_o);
}
