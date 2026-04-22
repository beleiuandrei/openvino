// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/select_with_fp16_min_to_fp32_min.hpp"

#include <gtest/gtest.h>

#include <limits>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/select.hpp"
#include "openvino/opsets/opset10_decl.hpp"

using namespace ov;
using namespace std;
using namespace testing;
using namespace ov::opset10;
using namespace ov::element;

namespace {

// Builds: Select(cond_param, then_param, Constant(et, shape, values))
shared_ptr<Model> gen_model(element::Type et,
                             const Shape& const_shape,
                             const vector<float>& const_values) {
    auto cond = make_shared<Parameter>(boolean, const_shape);
    auto then_br = make_shared<Parameter>(et, const_shape);
    auto else_br = make_shared<Constant>(et, const_shape, const_values);
    auto select = make_shared<Select>(cond, then_br, else_br);
    return make_shared<Model>(OutputVector{select}, ParameterVector{cond, then_br});
}

// Reference: Select(cond_param, then_param, Constant(f32, shape, fp32_min_values))
shared_ptr<Model> gen_reference(const Shape& const_shape) {
    constexpr float fp32_min = std::numeric_limits<float>::lowest();
    auto cond = make_shared<Parameter>(boolean, const_shape);
    auto then_br = make_shared<Parameter>(f32, const_shape);
    auto else_br = make_shared<Constant>(f32, const_shape, vector<float>(shape_size(const_shape), fp32_min));
    auto select = make_shared<Select>(cond, then_br, else_br);
    return make_shared<Model>(OutputVector{select}, ParameterVector{cond, then_br});
}

}  // namespace

class SelectWithFP16MinToFP32MinTest : public TransformationTestsF {};

// f32 constant with all fp16_min -> replaced with fp32 lowest
TEST_F(SelectWithFP16MinToFP32MinTest, ElseBranchF32AllFP16Min) {
    const Shape shape{2, 3};
    const float fp16_min = static_cast<float>(std::numeric_limits<ov::float16>::lowest());
    const vector<float> fp16_min_values(shape_size(shape), fp16_min);

    model = gen_model(f32, shape, fp16_min_values);
    manager.register_pass<pass::SelectWithFP16MinToFP32Min>();

    model_ref = gen_reference(shape);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

// f16 constant with all fp16_min -> replaced with fp32 lowest (type promoted to f32)
TEST_F(SelectWithFP16MinToFP32MinTest, ElseBranchF16AllFP16Min) {
    const Shape shape{4};
    const float fp16_min = static_cast<float>(std::numeric_limits<ov::float16>::lowest());
    const vector<float> fp16_min_values(shape_size(shape), fp16_min);

    model = gen_model(f16, shape, fp16_min_values);
    manager.register_pass<pass::SelectWithFP16MinToFP32Min>();

    model_ref = gen_reference(shape);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

// Mixed values -> transformation must NOT fire
TEST_F(SelectWithFP16MinToFP32MinTest, ElseBranchMixedValues_NoChange) {
    const Shape shape{3};
    const float fp16_min = static_cast<float>(std::numeric_limits<ov::float16>::lowest());
    const vector<float> mixed_values{fp16_min, 0.0f, fp16_min};

    model = gen_model(f32, shape, mixed_values);
    manager.register_pass<pass::SelectWithFP16MinToFP32Min>();

    // Reference equals the original model (no change expected)
    model_ref = gen_model(f32, shape, mixed_values);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

// Scalar constant with fp16_min -> replaced
TEST_F(SelectWithFP16MinToFP32MinTest, ElseBranchScalarFP16Min) {
    const Shape shape{};
    const float fp16_min = static_cast<float>(std::numeric_limits<ov::float16>::lowest());
    const vector<float> fp16_min_values{fp16_min};

    model = gen_model(f32, shape, fp16_min_values);
    manager.register_pass<pass::SelectWithFP16MinToFP32Min>();

    model_ref = gen_reference(shape);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}
