//
// Created by asus on 2025/1/19.
//

#ifndef CONTAINERS_HPP
#define CONTAINERS_HPP
#include "../../utils/type_traits.hpp"
#include "structure.hpp"

namespace metann {
template <typename...>
struct TagContainer;

template <typename...>
struct SubLayerContainer;
template <typename...>
struct InterConnectContainer;
template <typename...>
struct InConnectContainer;
template <typename...>
struct OutConnectContainer;

template <typename LayerTag, template <typename> class LayerType, typename PC>
struct SublayerPolicies {
    using Tag = LayerTag;
    template <typename T>
    using Layer = LayerType<T>;
    using Policy = PC;
};

template <typename...>
struct SublayerPolicyContainer;

/**
 * @brief Separate description statement into different containers
 * @tparam ...Parameters : description statement
 * @return SubLayerRes : SubLayerContainer<SubLayerPolicyContainer<...>>
 * @return InterConnectRes : InterConnectContainer<InterConnect<...>...>
 * @return InConnectRes : InConnectContainer<InConnect<...>...>
 * @return OutConnectRes : OutConnectContainer<OutConnect<...>...>
 */
template <typename... Parameters>
struct SeparateParameters {
    using SubLayerRes = Filter_t<true, IsSubLayer, SubLayerContainer, Parameters...>;
    using InterConnectRes = Filter_t<true, IsInternalConnect, InterConnectContainer, Parameters...>;
    using InConnectRes = Filter_t<true, IsInConnect, InConnectContainer, Parameters...>;
    using OutConnectRes = Filter_t<true, IsOutConnect, OutConnectContainer, Parameters...>;
};
}  // namespace metann

#endif  // CONTAINERS_HPP
