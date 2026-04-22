// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/select_with_fp16_min_to_fp32_min.hpp"

#include <limits>
#include <memory>

#include "openvino/core/type.hpp"

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/select.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

using namespace ov;

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;

namespace ov::pass {

SelectWithFP16MinToFP32Min::SelectWithFP16MinToFP32Min() {
    MATCHER_SCOPE(SelectWithFP16MinToFP32Min);

    auto condition = pattern::any_input();
    auto then_branch = pattern::any_input();
    auto else_branch = pattern::wrap_type<v0::Constant>();
    auto select_pattern = std::make_shared<v1::Select>(condition, then_branch, else_branch);

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto& pattern_map = m.get_pattern_value_map();

        auto select = ov::as_type_ptr<v1::Select>(pattern_map.at(select_pattern).get_node_shared_ptr());
        if (!select)
            return false;

        auto else_const =
            ov::as_type_ptr<v0::Constant>(pattern_map.at(else_branch).get_node_shared_ptr());
        if (!else_const)
            return false;

        const auto& et = else_const->get_element_type();
        if (et != element::f32 && et != element::f16)
            return false;

        // All values in the constant must equal the FP16 minimum
        const float fp16_min = static_cast<float>(std::numeric_limits<ov::float16>::lowest());
        auto values = else_const->cast_vector<float>();
        if (values.empty())
            return false;
        for (float v : values) {
            if (v != fp16_min)
                return false;
        }

        // Replace every value with FP32 lowest
        constexpr float fp32_min = std::numeric_limits<float>::lowest();
        std::vector<float> new_values(values.size(), fp32_min);

        auto new_const = std::make_shared<v0::Constant>(element::f32, else_const->get_shape(), new_values);
        new_const->set_friendly_name(else_const->get_friendly_name());
        copy_runtime_info(else_const, new_const);

        else_const->output(0).replace(new_const->output(0));
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(select_pattern, matcher_name);
    register_matcher(m, callback);
}

}  // namespace ov::pass
