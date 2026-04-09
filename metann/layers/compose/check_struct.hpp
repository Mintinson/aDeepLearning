//
// Created by asus on 2025/1/19.
//

#ifndef CHECK_STRUCT_HPP
#define CHECK_STRUCT_HPP

#include "containers.hpp"
#include "../policies/update_policy.hpp"


namespace metann::details
{
    /**
     * @brief check whether the tag is in the composed container
     * @tparam CheckTag: the tag to be checked
     * @tparam ... Array: the composed containers
     */
    template <typename CheckTag, typename... Array>
    struct TagExistInLayerComps
    {
        template <typename Tag, typename Other>
        struct TagExitsInLayerHelper;

        template <typename Tag, typename LayerTag, template<typename> class Layer>
        struct TagExitsInLayerHelper<Tag, SubLayer<LayerTag, Layer>>
        {
            constexpr static bool value = std::is_same_v<Tag, LayerTag>;
        };

        template <typename Tag, typename OutLayerTag, typename OutName,
                  typename InLayerTag, typename InName>
        struct TagExitsInLayerHelper<Tag, InternalConnect<OutLayerTag, OutName, InLayerTag, InName>>
        {
            constexpr static bool value = std::is_same_v<Tag, InLayerTag> || std::is_same_v<Tag, OutLayerTag>;
        };

        template <typename Tag, typename InName, typename InLayerTag, typename InLayerName>
        struct TagExitsInLayerHelper<Tag, InConnect<InName, InLayerTag, InLayerName>>
        {
            constexpr static bool value = std::is_same_v<Tag, InLayerTag>;
        };

        template <typename Tag, typename OutLayerTag, typename OutLayerName, typename OutName>
        struct TagExitsInLayerHelper<Tag, OutConnect<OutLayerTag, OutLayerName, OutName>>
        {
            constexpr static bool value = std::is_same_v<Tag, OutLayerTag>;
        };

        constexpr static bool value = Any_v<CheckTag, TagExitsInLayerHelper, Array...>;
    };
    /**
     * @brief check whether the tag is in the composed container
     * @tparam CheckTag: the tag to be checked
     * @tparam ... Array: the composed containers
     */
    template <typename CheckTag, typename... Array>
    constexpr static bool TagExistInLayerComps_v = TagExistInLayerComps<CheckTag, Array...>::value;

    /**
     * @brief check whether the tag is in the InternalConnect's OutTag
     * @tparam CheckTag: the tag to be checked
     * @tparam ... InternalConnects: the InternalConnect s
     */
    template <typename CheckTag, typename... InternalConnects>
    struct TagInInternalOut
    {
        template <typename Tag, typename Other>
        struct TagInInternalOutHelper
        {
            using OtherTag = typename Other::OutTag;
            constexpr static bool value = std::is_same_v<Tag, OtherTag>;
        };

        constexpr static bool value = Any_v<CheckTag, TagInInternalOutHelper, InternalConnects...>;
    };

    /**
     * @brief check whether the tag is in the InternalConnect's InTag
     * @tparam CheckTag: the tag to be checked
     * @tparam ... InternalConnects: the InternalConnect s
     */
    template <typename CheckTag, typename... InternalConnects>
    struct TagInInternalIn
    {
        template <typename Tag, typename Other>
        struct TagInInternalInHelper
        {
            using OtherTag = typename Other::InTag;
            constexpr static bool value = std::is_same_v<Tag, OtherTag>;
        };

        constexpr static bool value = Any_v<CheckTag, TagInInternalInHelper, InternalConnects...>;
    };

    template <typename>
    struct SubLayerChecker;

    template <typename... Sublayers>
    struct SubLayerChecker<SubLayerContainer<Sublayers...>>
    {
        // whether the tag is Unique
        constexpr static bool IsUnique = IsUnique_v<typename Sublayers::Tag...>;
    };

    template <typename>
    struct InternalConnectChecker;

    template <typename... Sublayers>
    struct InternalConnectChecker<InterConnectContainer<Sublayers...>>
    {
        constexpr static bool NoSelfCycle = !std::disjunction_v<std::is_same<
            typename Sublayers::OutTag, typename Sublayers::InTag>...>;

        template <typename Type1, typename Type2>
        struct UniqueSourceHelper
        {
            constexpr static bool value1 = std::is_same_v<typename Type1::InTag,
                                                          typename Type2::InTag>;
            constexpr static bool value2 = std::is_same_v<typename Type1::InName,
                                                          typename Type2::InName>;
            constexpr static bool value = value1 && value2;
        };

        constexpr static bool UniqueSource = IsGeneralUnique_v<UniqueSourceHelper, Sublayers...>;
    };

    template <typename>
    struct InputConnectChecker;

    template <typename... Sublayers>
    struct InputConnectChecker<InConnectContainer<Sublayers...>>
    {
        template <typename Type1, typename Type2>
        struct UniqueSourceHelper
        {
            constexpr static bool value1 = std::is_same_v<typename Type1::InLayerTag,
                                                          typename Type2::InLayerTag>;
            constexpr static bool value2 = std::is_same_v<typename Type1::InLayerName,
                                                          typename Type2::InLayerName>;
            constexpr static bool value = value1 && value2;
        };

        constexpr static bool UniqueSource = IsGeneralUnique_v<UniqueSourceHelper, Sublayers...>;
    };

    template <typename>
    struct OutputConnectChecker;

    template <typename... Sublayers>
    struct OutputConnectChecker<OutConnectContainer<Sublayers...>>
    {
        template <typename Type1, typename Type2>
        struct UniqueSourceHelper
        {
            constexpr static bool value = std::is_same_v<typename Type1::OutName,
                                                         typename Type2::OutName>;
        };

        constexpr static bool UniqueSource = IsGeneralUnique_v<UniqueSourceHelper, Sublayers...>;
    };

    template <typename, typename>
    struct InternalTagInSublayer;

    template <typename... IntSubLayers, typename... Sublayers>
    struct InternalTagInSublayer<InterConnectContainer<IntSubLayers...>,
                                 SubLayerContainer<Sublayers...>>
    {
        template <typename T, typename... Args>
        struct InternalTagHelper
        {
            using CurIn = typename T::InTag;
            using CurOut = typename T::OutTag;
            constexpr static bool value = TagExistInLayerComps<CurOut, Args...>::value
                && TagExistInLayerComps<CurIn, Args...>::value;
        };

        constexpr static bool value = std::conjunction_v<InternalTagHelper<IntSubLayers, Sublayers...>...>;
    };

    template <typename, typename>
    struct InputTagInSubLayer;

    template <typename... InputSubLayers, typename... Sublayers>
    struct InputTagInSubLayer<InConnectContainer<InputSubLayers...>,
                              SubLayerContainer<Sublayers...>>
    {
        template <typename T, typename... Args>
        struct InputTagHelper
        {
            constexpr static bool value = TagExistInLayerComps<typename T::InLayerTag, Args...>::value;
        };

        constexpr static bool value = std::conjunction_v<InputTagHelper<InputSubLayers, Sublayers...>...>;
    };

    template <typename, typename>
    struct OutputTagInSubLayer;

    template <typename... OutputSubLayers, typename... Sublayers>
    struct OutputTagInSubLayer<OutConnectContainer<OutputSubLayers...>,
                               SubLayerContainer<Sublayers...>>
    {
        template <typename T, typename... Args>
        struct OutputTagHelper
        {
            constexpr static bool value = TagExistInLayerComps<typename T::OutLayerTag, Args...>::value;
        };

        constexpr static bool value = std::conjunction_v<OutputTagHelper<OutputSubLayers, Sublayers...>...>;
    };

    template <typename InterArray, typename InArray, typename OutArray,
              typename SublayerArray>
    struct SublayerTagInOtherArrays;

    template <typename... InterElems, typename... InElems, typename... OutElems,
              typename... SublayerElems>
    struct SublayerTagInOtherArrays<InterConnectContainer<InterElems...>,
                                    InConnectContainer<InElems...>,
                                    OutConnectContainer<OutElems...>,
                                    SubLayerContainer<SublayerElems...>>
    {
        // template <typename T>
        // struct imp
        // {
        //     static constexpr bool value = true;
        // };

        template <typename Cur>
        struct SublayerTagHelper
        {
            using CurLayerTag = typename Cur::Tag;
            static constexpr bool tmp1 = TagExistInLayerComps<CurLayerTag, InterElems...>::value;
            static constexpr bool tmp2 = TagExistInLayerComps<CurLayerTag, InElems...>::value;
            static constexpr bool tmp3 = TagExistInLayerComps<CurLayerTag, OutElems...>::value;
            static constexpr bool value = tmp1 || tmp2 || tmp3;
        };

        constexpr static bool value = std::conjunction_v<SublayerTagHelper<SublayerElems>...>;
    };

    template <typename InterArray, typename OutArray>
    struct UsefulInternalPostLayer;

    template <typename... InterElems, typename... OutElems>
    struct UsefulInternalPostLayer<InterConnectContainer<InterElems...>,
                                   OutConnectContainer<OutElems...>>
    {
        template <typename Cur>
        struct UsefulInternalHelper
        {
            using CurCheckTag = typename Cur::InTag;
            static constexpr bool tmp1 = TagInInternalOut<CurCheckTag, InterElems...>::value;
            static constexpr bool tmp2 = TagExistInLayerComps<CurCheckTag, OutElems...>::value;
            static constexpr bool value = tmp1 || tmp2;
        };

        constexpr static bool value = std::conjunction_v<UsefulInternalHelper<InterElems>...>;
    };

    template <typename InArray, typename InterArray, typename OutArray>
    struct UsefulInputLayer;

    template <typename... InElems, typename... InterElems, typename... OutElems>
    struct UsefulInputLayer<InConnectContainer<InElems...>,
                            InterConnectContainer<InterElems...>,
                            OutConnectContainer<OutElems...>>
    {
        template <typename Cur>
        struct UsefulInternalHelper
        {
            using CurCheckTag = typename Cur::InLayerTag;
            static constexpr bool tmp1 = TagInInternalOut<CurCheckTag, InterElems...>::value;
            static constexpr bool tmp2 = TagExistInLayerComps<CurCheckTag, OutElems...>::value;
            static constexpr bool value = tmp1 || tmp2;
        };

        constexpr static bool value = std::conjunction_v<UsefulInternalHelper<InElems>...>;
    };

    template <bool plainFBO, typename Ints>
    struct FeedbackOutChecker
    {
        constexpr static bool value = true;
    };

    template <typename... InstElements>
    struct FeedbackOutChecker<true, SublayerPolicyContainer<InstElements...>>
    {
        template <typename SubPolicy>
        struct FeedbackOutPred
        {
            constexpr static bool value = PolicySelect_t<FeedbackPolicy,
                                                         typename SubPolicy::Policy>
                ::isFeedbackOutput;
        };

        constexpr static bool value = std::conjunction_v<FeedbackOutPred<InstElements>...>;
    };


}

#endif //CHECK_STRUCT_HPP
