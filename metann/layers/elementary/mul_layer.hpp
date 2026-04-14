//
// Created by asus on 2025/1/15.
//

#ifndef MUL_LAYER_HPP
#define MUL_LAYER_HPP
#include "../../policy/policy.hpp"
#include "../../utils/vartype_dict.hpp"
#include "../layer_helper.hpp"
#include "../layer_io.hpp"
#include "../policies/input_policy.hpp"
#include "../policies/operand_policy.hpp"
#include "../policies/update_policy.hpp"

namespace metann {
using MulLayerInput = VarTypeDict<struct MulLayerIn1, struct MulLayerIn2>;

template <typename PoliciesContainer>
    requires IsPolicyContainer_v<PoliciesContainer>
class MulLayer {
    using CurLayerPolicies = PlainPolicy_t<PoliciesContainer, PolicyContainer<>>;

public:
    static constexpr bool isFeedbackOutput =
        details::PolicySelect_t<FeedbackPolicy, CurLayerPolicies>::isFeedbackOutput;
    static constexpr bool isUpdate = false;
    using InputType = MulLayerInput;
    using OutputType = LayerIO;

    template <typename InType>
    auto feedForward(const InType& input) {
        const auto& val1 = input.template get<MulLayerIn1>();
        const auto& val2 = input.template get<MulLayerIn2>();
        static_assert(!std::is_same_v<std::decay_t<decltype(val1)>, details::NullParameter>, "parameter1 is invalid");
        static_assert(!std::is_same_v<std::decay_t<decltype(val2)>, details::NullParameter>, "parameter2 is invalid");
        if constexpr (isFeedbackOutput) {
            m_data1.push(make_dynamic(val1));
            m_data2.push(make_dynamic(val2));
        }
        return LayerIO::create().template set<LayerIO>(val1 * val2);
    }

    template <typename GradType>
    auto feedBackward(const GradType& grad) {
        if constexpr (isFeedbackOutput) {
            if (m_data1.empty() || m_data2.empty()) {
                throw std::runtime_error("feedBackward: Empty Inner Data");
            }
            auto top1 = m_data1.top();
            auto top2 = m_data2.top();
            m_data1.pop();
            m_data2.pop();

            auto gradEval = grad.template get<LayerIO>();
            return MulLayerInput::create()
                .template set<MulLayerIn1>(gradEval * top2)
                .template set<MulLayerIn2>(gradEval * top1);
        } else {
            return MulLayerInput::create();
        }
    }

    void neutralInvariant() {
        if constexpr (isFeedbackOutput) {
            if (m_data1.empty() && m_data2.empty()) {
                return;
            }
            throw std::runtime_error("neutralInvariant: Neural Invariant Fail!");
        }
    }

private:
    using DataType =
        details::LayerInternalBuf_t<isFeedbackOutput,
                                    details::PolicySelect_t<InputPolicy, CurLayerPolicies>::BatchModel,
                                    typename details::PolicySelect_t<OperandPolicy, CurLayerPolicies>::ElementType,
                                    typename details::PolicySelect_t<OperandPolicy, CurLayerPolicies>::Device,
                                    CategoryTags::Matrix,
                                    CategoryTags::BatchMatrix>;
    DataType m_data1;
    DataType m_data2;
};
}  // namespace metann

#endif  // MUL_LAYER_HPP
