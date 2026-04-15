#ifndef SGD_OPTIMIZER_HPP
#define SGD_OPTIMIZER_HPP

#include <stdexcept>
#include <unordered_map>

#include "../../operators/binary_operators.hpp"
#include "../../operators/unary_operators.hpp"
#include "../grad_collector.hpp"
#include "../interface_fun.hpp"

namespace metann::optim {
template <typename Element, DeviceConcept Device, typename ParamMap>
void sgd_update_parameters(ParamMap& params, GradCollector<Element, Device>& collector, const Element learningRate) {
    if (learningRate <= static_cast<Element>(0)) {
        throw std::runtime_error("sgd_update_parameters: learning rate must be positive.");
    }

    std::unordered_map<const Element*, Matrix<Element, Device>> gradientMap;
    gradientMap.reserve(collector.size());

    for (auto it = collector.begin(); it != collector.end(); ++it) {
        const auto& item = *it;
        const auto grad = evaluate(collapse(item.m_grad));

        const auto weightMem = lower_access(item.m_weight);
        const Element* weightPtr = weightMem.rawMemory();

        gradientMap.emplace(weightPtr, grad);
    }

    const Scalar<Element, Device> lr{learningRate};
    for (auto& kv : params) {
        auto& param = kv.second;

        const auto paramMem = lower_access(param);
        const Element* paramPtr = paramMem.rawMemory();

        if (auto it = gradientMap.find(paramPtr); it != gradientMap.end()) {
            param = evaluate(param - it->second * lr);
        }
    }
}

template <typename Layer, typename Element, DeviceConcept Device, typename Initializer, typename ParamMap>
void layer_sgd_step(Layer& layer, Initializer& initializer, ParamMap& params, const Element learningRate) {
    GradCollector<Element, Device> collector;
    layer_grad_collect(layer, collector);
    if (collector.size() == 0) {
        return;
    }

    sgd_update_parameters(params, collector, learningRate);
    layer_init(layer, initializer, params);
}
}  // namespace metann::optim

#endif  // SGD_OPTIMIZER_HPP
