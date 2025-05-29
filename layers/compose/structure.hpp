//
// Created by asus on 2025/1/19.
//

#ifndef STRUCTURE_HPP
#define STRUCTURE_HPP
#include <type_traits>

namespace metann
{
    /// for example: SubLayer<S1, SigmoidLayer>
    template <typename LayerTag, template <typename> class LayerTemplate>
    struct SubLayer
    {
        using Tag = LayerTag;
        template <typename T>
        using Layer = LayerTemplate<T>;
    };

    template <typename>
    struct IsSubLayer : std::false_type
    {
    };

    template <typename LayerTag, template <typename> class LayerTemplate>
    struct IsSubLayer<SubLayer<LayerTag, LayerTemplate>> : std::true_type
    {
    };

    template <typename T>
    constexpr bool IsSubLayer_v = IsSubLayer<T>::value;

    /// for example: InConnect<Input1, S1, LayerIO>
    template <typename InNameType, typename InLayerTagType, typename InLayerNameType>
    struct InConnect
    {
        using InName = InNameType;
        using InLayerTag = InLayerTagType;
        using InLayerName = InLayerNameType;
    };

    template <typename>
    struct IsInConnect : std::false_type
    {
    };

    template <typename InName, typename InLayerTag, typename InLayerName>
    struct IsInConnect<InConnect<InName, InLayerTag, InLayerName>> : std::true_type
    {
    };

    template <typename T>
    constexpr bool IsInConnect_v = IsInConnect<T>::value;


    /// for example: OutConnect<S1, LayerIO, Output1>
    template <typename OutLayerTagType, typename OutLayerNameType, typename OutNameType>
    struct OutConnect
    {
        using OutLayerTag = OutLayerTagType;
        using OutLayerName = OutLayerNameType;
        using OutName = OutNameType;
    };

    template <typename>
    struct IsOutConnect : std::false_type
    {
    };

    template <typename OutLayerTag, typename OutLayerName, typename OutName>
    struct IsOutConnect<OutConnect<OutLayerTag, OutLayerName, OutName>> : std::true_type
    {
    };

    template <typename T>
    constexpr bool IsOutConnect_v = IsOutConnect<T>::value;

    /// for example: InternalConnect<S1, LayerIO, S2, AddLayerIn1>
    template <typename OutLayerTagType, typename OutNameType, typename InLayerTagType, typename InNameType>
    struct InternalConnect
    {
        using OutTag = OutLayerTagType;
        using OutName = OutNameType;
        using InTag = InLayerTagType;
        using InName = InNameType;
    };

    template <typename>
    struct IsInternalConnect : std::false_type
    {
    };

    template <typename OutLayerTag, typename OutName, typename InLayerTag, typename InName>
    struct IsInternalConnect<InternalConnect<OutLayerTag, OutName, InLayerTag, InName>> : std::true_type
    {
    };

    template <typename T>
    constexpr bool IsInternalConnect_v = IsInternalConnect<T>::value;


} // namespace matann


#endif //STRUCTURE_HPP
