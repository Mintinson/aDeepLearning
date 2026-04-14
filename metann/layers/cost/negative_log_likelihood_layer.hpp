#pragma once
#include "../../operators/ternary_operators.hpp"
#include "../../policy/policy.hpp"
#include "../layer_helper.hpp"
#include "../layer_io.hpp"
#include "../policies/input_policy.hpp"
#include "../policies/operand_policy.hpp"
#include "../policies/update_policy.hpp"

namespace metann {
struct OperandPolicy;

namespace details {
template <bool isFeedback>
struct NLLFeedback {
    template <typename TLabel, typename TIn, typename TData>
    static void recordData(const TLabel& label, const TIn& in, TData& label_stack, TData& pred_stack) {
        label_stack.push(make_dynamic(label));
        pred_stack.push(make_dynamic(in));
    }

    template <typename TGrad, typename TData>
    static auto feedback(const TGrad& p_grad, TData& label, TData& pred) {
        if ((label.empty()) || (pred.empty())) {
            throw std::runtime_error("Cannot do FeedBackward for Negative Log-likelihood Layer");
        }
        auto l = label.top();
        auto p = pred.top();
        label.pop();
        pred.pop();

        auto g = evaluate(p_grad.template get<LayerIO>());
        auto l_eval = evaluate(std::move(l));
        auto p_eval = evaluate(std::move(p));
        auto res = neg_log_likelihood_derivative(std::move(g), std::move(l_eval), std::move(p_eval));
        return CostLayerIO::create().template set<CostLayerIO>(std::move(res));
    }
};

template <>
struct NLLFeedback<false> {
    template <typename TLabel, typename TIn, typename TData>
    static void recordData(TLabel&&, TIn&& p_in, TData&&, TData&&) {}

    template <typename TGrad, typename TData>
    static auto feedback(TGrad&&, TData&&, TData&&) {
        return CostLayerIO::create();
    }
};
}  // namespace details

template <typename PolicyCont>
    requires IsPolicyContainer_v<PolicyCont>
class NegativeLogLikelihoodLayer {
    using CurrentLayerPolicy = PlainPolicy_t<PolicyCont, PolicyContainer<>>;

public:
    static constexpr bool isFeedbackOutput =
        details::PolicySelect_t<FeedbackPolicy, CurrentLayerPolicy>::isFeedbackOutput;
    static constexpr bool isUpdate = false;
    using InputType = CostLayerIO;
    using OutputType = LayerIO;

private:
    using ElementType = details::PolicySelect_t<OperandPolicy, CurrentLayerPolicy>::ElementType;
    using DeviceType = details::PolicySelect_t<OperandPolicy, CurrentLayerPolicy>::Device;
    using Feedback = details::NLLFeedback<isFeedbackOutput>;
    // using FeedBack =
public:
    template <typename Input>
    auto feedForward(const Input& p_in) {
        const auto& input = p_in.template get<InputType>();
        const auto& label = p_in.template get<CostLayerLabel>();

        using rawType1 = std::decay_t<decltype(input)>;
        using rawType2 = std::decay_t<decltype(label)>;
        static_assert(!std::is_same_v<rawType1, details::NullParameter>, "Input is invalid");
        static_assert(!std::is_same_v<rawType2, details::NullParameter>, "Label is invalid");

        Feedback::recordData(label, input, m_label, m_pred);
        auto pred_eval = evaluate(input);
        auto label_eval = evaluate(label);
        return OutputType::create().template set<OutputType>(
            neg_log_likelihood(std::move(label_eval), std::move(pred_eval)));
    }

    template <typename TGrad>
    auto feedBackward(TGrad&& p_grad) {
        return Feedback::feedback(std::forward<TGrad>(p_grad), m_label, m_pred);
    }

    void neutralInvariant() const {
        if constexpr (isFeedbackOutput) {
            if (!m_label.empty() || !m_pred.empty()) {
                throw std::runtime_error("neutralInvariant: Neural Invariant Fail!");
            }
        }
    }

private:
    using DataType = details::LayerInternalBuf_t<isFeedbackOutput,
                                                 details::PolicySelect_t<InputPolicy, CurrentLayerPolicy>::BatchModel,
                                                 ElementType,
                                                 DeviceType,
                                                 CategoryTags::Matrix,
                                                 CategoryTags::BatchMatrix>;
    DataType m_label;
    DataType m_pred;
};
}  // namespace metann
