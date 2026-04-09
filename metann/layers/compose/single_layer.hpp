//
// Created by asus on 2025/1/23.
//

#ifndef SINGLE_LAYER_HPP
#define SINGLE_LAYER_HPP


#include "compose_core.hpp"
#include "../layer_io.hpp"
#include "../elementary/bias_layer.hpp"
#include "../elementary/weight_layer.hpp"
#include "structure.hpp"
#include "../elementary/sigmoid_layer.hpp"
#include "../elementary/tanh_layer.hpp"
#include "../policies/single_layer_policy.hpp"

namespace metann
{
    template <typename Policies>
    class SingleLayer;

    template <>
    struct SubLayerOf<SingleLayer>
    {
        struct Weight;
        struct Bias;
        struct Action;
    };

    namespace details::single_layer
    {
        template <typename PlainPolicies>
        constexpr static bool HasBias_v = PolicySelect_t<SingleLayerPolicy, PlainPolicies>::HasBias;

        template <typename PlainPolicies>
        struct ActionPicker;

        template <>
        struct ActionPicker<SingleLayerPolicy::ActionTypeCate::Sigmoid>
        {
            template <typename T>
            using type = SigmoidLayer<T>;
        };

        template <>
        struct ActionPicker<SingleLayerPolicy::ActionTypeCate::Tanh>
        {
            template <typename T>
            using type = TanhLayer<T>;
        };

        template <bool hasBias, template <typename> class ActLayer>
        struct TopologyHelper
        {
            using type = ComposeTopology<SubLayer<SubLayerOf<SingleLayer>::Weight, WeightLayer>,
                                         SubLayer<SubLayerOf<SingleLayer>::Bias, BiasLayer>,
                                         SubLayer<SubLayerOf<SingleLayer>::Action, ActLayer>,
                                         InConnect<LayerIO, SubLayerOf<SingleLayer>::Weight, LayerIO>,
                                         InternalConnect<
                                             SubLayerOf<SingleLayer>::Weight, LayerIO, SubLayerOf<SingleLayer>::Bias,
                                             LayerIO>,
                                         InternalConnect<
                                             SubLayerOf<SingleLayer>::Bias, LayerIO, SubLayerOf<SingleLayer>::Action,
                                             LayerIO>,
                                         OutConnect<SubLayerOf<SingleLayer>::Action, LayerIO, LayerIO>>;
        };

        template <template <typename> class TActLayer>
        struct TopologyHelper<false, TActLayer>
        {
            using type = ComposeTopology<SubLayer<SubLayerOf<SingleLayer>::Weight, WeightLayer>,
                                         SubLayer<SubLayerOf<SingleLayer>::Action, TActLayer>,
                                         InConnect<LayerIO, SubLayerOf<SingleLayer>::Weight, LayerIO>,
                                         InternalConnect<
                                             SubLayerOf<SingleLayer>::Weight, LayerIO,
                                             SubLayerOf<SingleLayer>::Action, LayerIO>,
                                         OutConnect<SubLayerOf<SingleLayer>::Action, LayerIO, LayerIO>>;
        };

        template <typename Policies>
        struct KernelHelper
        {
            using PlainPolicies = PlainPolicy_t<Policies, PolicyContainer<>>;
            constexpr static bool hasBias = HasBias_v<PlainPolicies>;

            template <typename T>
            using ActType = typename ActionPicker<
                typename details::PolicySelect_t<SingleLayerPolicy,
                                                 PlainPolicies>::Action>::template type<T>;

            using type = typename TopologyHelper<hasBias, ActType>::type;
        };

        template <bool HasBias, typename TBase>
        auto TupleCreator(const std::string& p_name, size_t p_inputLen, size_t p_outputLen)
        {
            if constexpr (HasBias)
            {
                return TBase::createSubLayers()
                       .template set<SubLayerOf<SingleLayer>::Weight>(p_name + "-weight", p_inputLen, p_outputLen)
                       .template set<SubLayerOf<SingleLayer>::Bias>(p_name + "-bias", p_outputLen)
                       .template set<SubLayerOf<SingleLayer>::Action>();
            }
            else
            {
                return TBase::createSubLayers()
                       .template set<SubLayerOf<SingleLayer>::Weight>(p_name + "-weight", p_inputLen, p_outputLen)
                       .template set<SubLayerOf<SingleLayer>::Action>();
            }
        }
    }

    template <typename Policies>
    class SingleLayer
        : public ComposeKernel<LayerIO, LayerIO, Policies,
                               typename details::single_layer::KernelHelper<Policies>::type>
    {
        using Base = ComposeKernel<LayerIO, LayerIO, Policies,
                                   typename details::single_layer::KernelHelper<Policies>::type>;

    public:
        SingleLayer(const std::string& name, const size_t inputLen, const size_t outputLen):
            Base{
                details::single_layer::TupleCreator<
                    details::single_layer::KernelHelper<Policies>::hasBias,
                    Base>(name, inputLen, outputLen)
            }
        {
        }
    };
}

#endif //SINGLE_LAYER_HPP
