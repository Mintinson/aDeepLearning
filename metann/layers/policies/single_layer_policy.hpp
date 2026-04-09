//
// Created by asus on 2025/1/23.
//

#ifndef SINGLE_LAYER_POLICY_HPP
#define SINGLE_LAYER_POLICY_HPP
namespace metann
{
    struct SingleLayerPolicy
    {
        using MajorClass = SingleLayerPolicy;

        struct ActionTypeCate
        {
            struct Sigmoid;
            struct Tanh;
        };

        struct HasBiasValueCate;

        using Action = ActionTypeCate::Sigmoid;
        static constexpr bool HasBias = true;
    };

    struct SigmoidAction : virtual SingleLayerPolicy
    {
        using MinorClass = ActionTypeCate;
        using Action = ActionTypeCate::Sigmoid;
    };

    struct TanhAction : virtual SingleLayerPolicy
    {
        using MinorClass = ActionTypeCate;
        using Action = ActionTypeCate::Tanh;
    };

    struct BiasSingleLayer : virtual SingleLayerPolicy
    {
        using MinorClass = HasBiasValueCate;
        static constexpr bool HasBias = true;
    };
    struct NoBiasSingleLayer : virtual SingleLayerPolicy
    {
        using MinorClass = HasBiasValueCate;
        static constexpr bool HasBias = false;
    };

}
#endif //SINGLE_LAYER_POLICY_HPP
