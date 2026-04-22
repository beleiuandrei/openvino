// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API SelectWithFP16MinToFP32Min;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief SelectWithFP16MinToFP32Min transformation replaces a constant else-branch of a Select
 * node that contains only -65504.0f (FP16 minimum) values with a constant holding
 * std::numeric_limits<float>::lowest() (FP32 minimum) values.
 *
 * This is useful when a causal-attention mask encoded as FP16-min in a float32 graph causes
 * incorrect softmax results because -65504 does not underflow to -inf in float32 arithmetic.
 */
class ov::pass::SelectWithFP16MinToFP32Min : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("SelectWithFP16MinToFP32Min");
    SelectWithFP16MinToFP32Min();
};
