//
// Created by asus on 2025/1/23.
//

#ifndef LINEAR_LAYER_HPP
#define LINEAR_LAYER_HPP

#include "../elementary/bias_layer.hpp"
#include "../elementary/weight_layer.hpp"
#include "../layer_io.hpp"
#include "compose_core.hpp"
#include "structure.hpp"

namespace metann {
template <typename Policies>
class LinearLayer;

template <>
struct SubLayerOf<LinearLayer> {
    struct Weight;
    struct Bias;
};

using LinearTopology =
    ComposeTopology<SubLayer<SubLayerOf<LinearLayer>::Weight, WeightLayer>,
                    SubLayer<SubLayerOf<LinearLayer>::Bias, BiasLayer>,
                    InConnect<LayerIO, SubLayerOf<LinearLayer>::Weight, LayerIO>,
                    InternalConnect<SubLayerOf<LinearLayer>::Weight, LayerIO, SubLayerOf<LinearLayer>::Bias, LayerIO>,
                    OutConnect<SubLayerOf<LinearLayer>::Bias, LayerIO, LayerIO>>;

template <typename Policies>
class LinearLayer : public ComposeKernel<LayerIO, LayerIO, Policies, LinearTopology> {
    using Base = ComposeKernel<LayerIO, LayerIO, Policies, LinearTopology>;
    using WeightSubLayer = SubLayerOf<LinearLayer>::Weight;
    using BiasSubLayer = SubLayerOf<LinearLayer>::Bias;

public:
    LinearLayer(const std::string& name, std::size_t inputLen, std::size_t outputLen)
        : Base(Base::createSubLayers()
                   .template set<WeightSubLayer>(name + "-weight", inputLen, outputLen)
                   .template set<BiasSubLayer>(name + "-bias", outputLen)) {}
};
}  // namespace metann

#endif  // LINEAR_LAYER_HPP
