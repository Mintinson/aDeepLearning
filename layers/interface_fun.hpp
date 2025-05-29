//
// Created by asus on 2025/1/21.
//

#ifndef INTERFACE_FUN_HPP
#define INTERFACE_FUN_HPP
#include <type_traits>
#include <iostream>

namespace metann
{
    template <typename Layer, typename TIn>
    auto layer_feedforward(Layer& layer, TIn&& p_in)
    {
        return layer.feedForward(std::forward<TIn>(p_in));
    }

    template <typename Layer, typename TGrad>
    auto layer_feedbackward(Layer& layer, TGrad&& p_grad)
    {
        return layer.feedBackward(std::forward<TGrad>(p_grad));
    }

    /// init interface ========================================
    namespace details
    {

        template <typename L, typename Initializer, typename Buffer, typename InitPolicies>
        std::true_type init_test(decltype(&L::template init<Initializer, Buffer, InitPolicies>));

        template <typename L, typename InitPolicies, typename TInitContainer, typename TLoad>
        std::false_type init_test(...);

        template <typename L, typename GradCollector>
        std::true_type grad_collect_test(decltype(&L::template gradCollect<GradCollector>));

        template <typename L, typename GradCollector>
        std::false_type grad_collect_test(...);

        template <typename L, typename Save>
        std::true_type save_weights_test(decltype(&L::template saveWeights<Save>));

        template <typename L, typename Save>
        std::false_type save_weights_test(...);

        template <typename L>
        std::true_type neutral_invariant_test(decltype(&L::neutralInvariant));

        template <typename L>
        std::false_type neutral_invariant_test(...);

    }
    /**
     * @brief init and load weights
     * @tparam Layer
     * @tparam Initializer
     * @tparam Buffer
     * @tparam InitPolicies
     * @param layer
     * @param initializer
     * @param loadBuffer
     * @param log
     */
    template <typename Layer, typename Initializer, typename Buffer,
        typename InitPolicies = typename Initializer::PolicyCont>
    void layer_init(Layer& layer, Initializer& initializer, Buffer& loadBuffer, std::ostream* log = nullptr)
    {
        if constexpr (decltype(details::init_test<Layer, Initializer, Buffer, InitPolicies>(nullptr)
            )::value)
            layer.template init<Initializer, Buffer, InitPolicies>(initializer, loadBuffer, log);
    }

    /**
     * @brief collect the gradients
     * @tparam Layer
     * @tparam GradCollector
     * @param layer
     * @param gc
     */
    template <typename Layer, typename GradCollector>
    void layer_grad_collect(Layer& layer, GradCollector& gc)
    {
        if constexpr (decltype(details::grad_collect_test<Layer, GradCollector>(nullptr))::value)
            layer.gradCollect(gc);
    }

    /**
     * @brief save the weights to the saver
     * @tparam Layer
     * @tparam Save
     * @param layer
     * @param saver
     */
    template <typename Layer, typename Save>
    void layer_save_weights(const Layer& layer, Save& saver)
    {
        if constexpr (decltype(details::save_weights_test<Layer, Save>(nullptr))::value)
            layer.saveWeights(saver);
    }

    /**
     * @brief make sure the layer is neutral invariant
     * @tparam Layer
     * @param layer
     */
    template <typename Layer>
    void layer_neutral_invariant(Layer& layer)
    {
        if constexpr (decltype(details::neutral_invariant_test<Layer>(nullptr))::value)
            layer.neutralInvariant();
    }
}

#endif //INTERFACE_FUN_HPP
