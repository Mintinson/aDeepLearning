//
// Created by asus on 2025/1/15.
//

#ifndef ABS_LAYER_HPP
#define ABS_LAYER_HPP

#include "../../policy/policy.hpp"
#include "../layer_helper.hpp"
#include "../layer_io.hpp"
#include "../policies/operand_policy.hpp"
#include "../policies/update_policy.hpp"

namespace metann {
template <typename PoliciesContainer>
    requires IsPolicyContainer_v<PoliciesContainer>
class AbsLayer {
    using CurLayerPolicies = PlainPolicy_t<PoliciesContainer, PolicyContainer<>>;

public:
    static constexpr bool isFeedbackOutput = details::PolicySelect_t<
        FeedbackPolicy, CurLayerPolicies>::isFeedbackOutput;
    static constexpr bool isUpdate = false;
    using InputType = LayerIO;
    using OutputType = LayerIO;

private:
    using ElementType = typename details::PolicySelect_t<OperandPolicy, CurLayerPolicies>::ElementType;
    using DeviceType = typename details::PolicySelect_t<OperandPolicy, CurLayerPolicies>::Device;

    using FeedbackOut = details::FeedbackOut<isFeedbackOutput>;

public:
    template <typename InType>
    auto feedForward(const InType& input)
    {
        // const auto& val1 = input.template get<AddLayerIn1>();
        // const auto& val2 = input.template get<AddLayerIn2>();
        // using RawType1 = std::decay_t<decltype(val1)>;
        // using RawType2 = std::decay_t<decltype(val2)>;
        // static_assert(!std::is_same_v<RawType1, details::NullParameter>,
        //               "parameter1 is invalid!");
        // static_assert(!std::is_same_v<RawType2, details::NullParameter>,
        //               "parameter2 is invalid");
        // return OutputType::create().template set<LayerIO>(val1 + val2);
        const auto& val = input.template get<LayerIO>();

        using rawType = std::decay_t<decltype(val)>;
        static_assert(!std::is_same_v<rawType, details::NullParameter>, "parameter is invalid");

        auto tmp = abs(val);
        FeedbackOut::recordData(val, m_data);
        return LayerIO::create().template set<LayerIO>(tmp);
    }

    template <typename InGrad>
    auto feedBackward(const InGrad& grad)
    {
        return FeedbackOut::feedback(m_data, grad);
    }
    void neutralInvariant() const
    {
        if constexpr (isFeedbackOutput) {
            if (!m_data.empty()) {
                throw std::runtime_error("NeutralInvariant Fail!");
            }
        }
    }

private:
    using InternalType = typename FeedbackOut::template InternalType<ElementType, DeviceType>;
    InternalType m_data;
};
} // namespace metann

#endif // ABS_LAYER_HPP
