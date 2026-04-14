//
// Created by asus on 2025/1/15.
//

#ifndef SOFTMAX_LAYER_HPP
#define SOFTMAX_LAYER_HPP

#include "../../operators/binary_operators.hpp"
#include "../../operators/unary_operators.hpp"
#include "../../policy/policy.hpp"
#include "../../utils/vartype_dict.hpp"
#include "../layer_helper.hpp"
#include "../layer_io.hpp"
#include "../policies/input_policy.hpp"
#include "../policies/operand_policy.hpp"
#include "../policies/update_policy.hpp"

namespace metann {
template <typename PoliciesContainer>
    requires IsPolicyContainer_v<PoliciesContainer>
class SoftmaxLayer {
    using CurLayerPolicies = PlainPolicy_t<PoliciesContainer, PolicyContainer<>>;

public:
    static constexpr bool isFeedbackOutput =
        details::PolicySelect_t<FeedbackPolicy, CurLayerPolicies>::isFeedbackOutput;
    static constexpr bool isUpdate = false;
    using InputType = LayerIO;
    using OutputType = LayerIO;

    template <typename InType>
    auto feedForward(const InType& input) {
        const auto& val = input.template get<LayerIO>();
        static_assert(!std::is_same_v<std::decay_t<decltype(val)>, details::NullParameter>, "parameter1 is invalid");
        auto tmp = vec_softmax(val);

        if constexpr (isFeedbackOutput) {
            m_data.push(make_dynamic(tmp));
            return LayerIO::create().template set<LayerIO>(tmp);
        } else {
            return LayerIO::create().template set<LayerIO>(tmp);
        }
    }

    template <typename GradType>
    auto feedBackward(const GradType& grad) {
        if constexpr (isFeedbackOutput) {
            if (m_data.empty()) {
                throw std::runtime_error("feedBackward: Empty Inner Data");
            }

            auto prevGrad = evaluate(grad.template get<LayerIO>());
            auto curRes = evaluate(m_data.top());
            auto res =
                LayerIO::create().template set<LayerIO>(softmax_derivative(std::move(prevGrad), std::move(curRes)));
            m_data.pop();
            return res;
        } else {
            return LayerIO::create();
        }
    }

    void neutralInvariant() {
        if constexpr (isFeedbackOutput) {
            if (m_data.empty()) {
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
    DataType m_data;
};
}  // namespace metann

#endif  // SOFTMAX_LAYER_HPP
