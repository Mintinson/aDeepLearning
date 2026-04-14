//
// Created by asus on 2025/1/14.
//

#ifndef ADD_LAYER_HPP
#define ADD_LAYER_HPP
#include <type_traits>

#include "../../policy/policy.hpp"
#include "../../utils/vartype_dict.hpp"
#include "../layer_io.hpp"
#include "../policies/update_policy.hpp"

namespace metann {
using AddLayerInput = VarTypeDict<struct AddLayerIn1, struct AddLayerIn2>;

template <typename PoliciesContainer>
    requires IsPolicyContainer_v<PoliciesContainer>
class AddLayer {
    using CurLayerPolicies = PlainPolicy_t<PoliciesContainer, PolicyContainer<>>;

public:
    static constexpr bool isFeedbackOutput =
        details::PolicySelect_t<FeedbackPolicy, CurLayerPolicies>::isFeedbackOutput;
    static constexpr bool isUpdate = false;
    using InputType = AddLayerInput;
    using OutputType = LayerIO;

    template <typename InType>
    auto feedForward(const InType& input) {
        const auto& val1 = input.template get<AddLayerIn1>();
        const auto& val2 = input.template get<AddLayerIn2>();
        using RawType1 = std::decay_t<decltype(val1)>;
        using RawType2 = std::decay_t<decltype(val2)>;
        static_assert(!std::is_same_v<RawType1, details::NullParameter>, "parameter1 is invalid!");
        static_assert(!std::is_same_v<RawType2, details::NullParameter>, "parameter2 is invalid");
        return OutputType::create().template set<LayerIO>(val1 + val2);
    }

    template <typename InGrad>
    auto feedBackward(const InGrad& grad) {
        if constexpr (isFeedbackOutput) {
            auto res = grad.template get<LayerIO>();
            return AddLayerInput::create().template set<AddLayerIn1>(res).template set<AddLayerIn2>(res);
        } else {
            return AddLayerInput::create();
        }
    }
};
}  // namespace metann

#endif  // ADD_LAYER_HPP
