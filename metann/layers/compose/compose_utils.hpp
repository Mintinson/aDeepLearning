//
// Created by asus on 2025/1/21.
//

#ifndef COMPOSE_UTILS_HPP
#define COMPOSE_UTILS_HPP

#include "containers.hpp"
#include "../../utils/type_traits.hpp"
#include "../../policy/policy.hpp"
#include <memory>
#include "../../utils/vartype_dict.hpp"

namespace metann::details
{
    /**
     * @brief Iterate through all the elements in CheckInternals and 
        find sublayers from them that don't have any information about the predecessor layer, 
        put those sublayers in the PostTags queue, and also, if InternalLayerPrune::PostTags 
        introduces a new layer, remove the connection corresponding to that layer in 
        CheckInternals relationship in CheckInternals, and eventually, 
        InternalLayerPrune will construct a new InterConnectContainer container that returns as type
     * @tparam RemainInters 
     * @tparam CheckInters 
     * @tparam PostTagsType 
     * @tparam ...T 
     * @return PostTags: the tags of the sublayers that are not connected to the predecessor layer
     * @return type: new InterConnectContainer container tahat contains the sublayers that are connected to the predecessor layer
     */
    template <typename RemainInters, typename CheckInters, typename PostTagsType, typename... T>
    struct InternalLayerPrune
    {
        using PostTags = PostTagsType;
        using type = RemainInters;
    };

    template <typename... T1, typename... T2, typename... TTags, typename TCur, typename... T3>
    struct InternalLayerPrune<InterConnectContainer<T1...>,
                              InterConnectContainer<T2...>,
                              TagContainer<TTags...>,
                              TCur, T3...>
    {
        using CheckTag = typename TCur::OutTag;
        static constexpr bool inInterIn = TagInInternalIn<CheckTag, T2...>::value;

        template <bool isCheckOk, typename TDummy = void>
        struct put
        {
            using NewTagContainer = TagContainer<TTags...>;
            using type = InterConnectContainer<T1..., TCur>;
        };

        template <typename TDummy>
        struct put<false, TDummy>
        {
            constexpr static bool CheckTagInTTags = IsInPack_v<CheckTag, TTags...>;
            using NewTagContainer = std::conditional_t<CheckTagInTTags,
                                                       TagContainer<TTags...>,
                                                       TagContainer<TTags..., CheckTag>>;
            using type = InterConnectContainer<T1...>;
        };

        using tmp = typename put<inInterIn>::type;
        using tmpTagContainer = typename put<inInterIn>::NewTagContainer;

        using nextStep = InternalLayerPrune<tmp, InterConnectContainer<T2...>, tmpTagContainer, T3...>;

        using type = typename nextStep::type;
        using PostTags = typename nextStep::PostTags;
    };

    // If the tag of an element in UnOrderedSubLayers is in PostTagsContains
    // move the element to the end of OrderedSubLayers as Ordered.
    // other elements remain in Remain.

    /**
     * @brief If the tag of an element in UnOrderedSubLayers is in PostTagsContains
         move the element to the end of OrderedSubLayers as Ordered.
         other elements remain in Remain.
     * @tparam OrderedSubLayers 
     * @tparam UnOrderedSubLayers 
     * @tparam PostTagsContains 
     * @return Ordered: new ordered container
     * @return Remain: new unordered container
     */
    template <typename OrderedSubLayers,
              typename UnOrderedSubLayers,
              typename PostTagsContains>
    struct SeparateByPostTag;

    template <typename... OrderedElements, typename... UnOrderedElements, typename... PostTags>
    struct SeparateByPostTag<SubLayerContainer<OrderedElements...>,
                             SubLayerContainer<UnOrderedElements...>,
                             TagContainer<PostTags...>>
    {
        using OldOrdered = SubLayerContainer<OrderedElements...>;

        template <typename T>
        struct SeparateHelper
        {
            constexpr static bool value = IsInPack_v<typename T::Tag, PostTags...>;
        };

        using SeparatedFunc = SeparateBy<SeparateHelper,
                                         SubLayerContainer, SubLayerContainer,
                                         UnOrderedElements...>;
        using Ordered = ConcatContainer_t<OldOrdered, typename SeparatedFunc::TrueType>;
        using Remain = typename SeparatedFunc::FalseType;
    };

    /**
     * @brief Internally call `SubPolicyPicker` on each sublayer to derive a policy for each sublayer
        based on the policy of the composite layer.
     * @tparam PolicyCont 
     * @tparam OrderedSublayerCont
     * @return type: SubLayerPolicyContainer container, whose internal elements are `SubLayerPolicies`.
     */
    template <typename PolicyCont, typename OrderedSublayerCont>
    struct GetSublayerPolicy;

    template <typename PolicyCont, typename... SubLayers>
    struct GetSublayerPolicy<PolicyCont, SubLayerContainer<SubLayers...>>
    {
        template <typename Cur>
        struct GetSublayerPolicyHelper
        {
            using CurTag = typename Cur::Tag;

            template <typename T>
            using CurLayer = typename Cur::template Layer<T>;

            using CurPolicy = SubPolicyPicker_t<PolicyCont, CurTag>;
            using type = SublayerPolicies<CurTag, CurLayer, CurPolicy>;
        };

        //template <typename TRes, typename... T>
        //struct imp
        //{
        //    using type = TRes;
        //};

        //template <typename... T1, typename TCur, typename... T2>
        //struct imp<SublayerPolicyContainer<T1...>, TCur, T2...>
        //{
        //    using CurTag = typename TCur::Tag;
        //    template <typename T>
        //    using CurLayer = typename TCur::template Layer<T>;
        //    using CurPolicy = SubPolicyPicker_t<PolicyCont, CurTag>;
        //    using tmp = SublayerPolicyContainer<T1..., SublayerPolicies<CurTag, CurLayer, CurPolicy>>;
        //    using type = typename imp<tmp, T2...>::type;
        //};

        using type = SublayerPolicyContainer<typename GetSublayerPolicyHelper<SubLayers>::type...>;
        // using type = typename imp<SublayerPolicyContainer<>, SubLayers...>::type;
    };

    template <bool isUpdated, typename TTag, typename TRes, typename TInstCont>
    struct UpdateByTag
    {
        using type = TInstCont;
    };

    template <typename TTag, typename TRes, typename TInstCont>
    struct UpdateByTag<true, TTag, TRes, TInstCont>
    {
        using type = TRes;
    };

    template <typename TTag, typename... TInstRes, typename TCur, typename... TInstRemain>
    struct UpdateByTag<true, TTag, SublayerPolicyContainer<TInstRes...>,
                       SublayerPolicyContainer<TCur, TInstRemain...>>
    {
        using OriTag = typename TCur::Tag;
        using OriPolicy = typename TCur::Policy;
        template <typename T>
        using OriLayer = typename TCur::template Layer<T>;

        constexpr static bool checkOK = std::is_same_v<TTag, OriTag>;
        using NewPolicy = std::conditional_t<checkOK,
                                             ChangePolicy_t<FeedbackOutputPolicy, OriPolicy>,
                                             std::type_identity_t<OriPolicy>>;

        using TModified = SublayerPolicies<OriTag, OriLayer, NewPolicy>;
        using NewRes = SublayerPolicyContainer<TInstRes..., TModified>;
        using type = typename UpdateByTag<true, TTag, NewRes, SublayerPolicyContainer<TInstRemain...>>::type;
    };

    template <typename TTag, typename TInterConnects, typename TInstCont>
    struct UpdateBySourceLayer
    {
        static_assert(ContainerSize_v<TInterConnects> == 0, "Test Error");
        using type = TInstCont;
    };

    template <typename TTag, typename TI, typename... TIRemain, typename TInstCont>
    struct UpdateBySourceLayer<TTag, InterConnectContainer<TI, TIRemain...>, TInstCont>
    {
        using OutTag = typename TI::OutTag;
        using InTag = typename TI::InTag;
        constexpr static bool isOutTag = std::is_same_v<TTag, OutTag>;
        using tmp = typename UpdateByTag<isOutTag, InTag,
                                         SublayerPolicyContainer<>, TInstCont>::type;
        using type = typename UpdateBySourceLayer<TTag, InterConnectContainer<TIRemain...>, tmp>::type;
    };
    /**
     * @brief Iterate through each sublayer in the SublayerPolicyContainer, and 
            if the current sublayer needs to compute the output gradient or parameter gradient, 
            then correct the policy of the succeeding layer based on the connectivity of the sublayers
     * @tparam Insts 
     * @tparam InterConnects 
     */
    template <typename Insts, typename InterConnects>
    struct FeedbackOutSet;

    template <typename... InstElements, typename InterConnects>
    struct FeedbackOutSet<SublayerPolicyContainer<InstElements...>, InterConnects>
    {
        template <typename TRes, typename TNotProcessed>
        struct imp
        {
            using type = TRes;
        };

        template <typename... TProcessedInst, typename TCur, typename... TInsts>
        struct imp<SublayerPolicyContainer<TProcessedInst...>,
                   SublayerPolicyContainer<TCur, TInsts...>>
        {
            using Tag = typename TCur::Tag;
            using CurPolicies = typename TCur::Policy;

            constexpr static bool isUpdate = PolicySelect_t<FeedbackPolicy, CurPolicies>::isFeedbackOutput
                || PolicySelect_t<FeedbackPolicy, CurPolicies>::isUpdate;

            using tmp1 = SublayerPolicyContainer<TProcessedInst..., TCur>;
            using tmp2
            = typename std::conditional_t<isUpdate,
                                          UpdateBySourceLayer<Tag, InterConnects,
                                                              SublayerPolicyContainer<TInsts...>>,
                                          std::type_identity<SublayerPolicyContainer<TInsts...>>>::type;
            using type = typename imp<tmp1, tmp2>::type;
        };

        using type = typename imp<SublayerPolicyContainer<>, SublayerPolicyContainer<InstElements...>>::type;
    };

    template <typename SubLayerTuple>
    struct SubLayerArrayMaker;

    template <typename... SubLayers>
    struct SubLayerArrayMaker<std::tuple<SubLayers...>>
    {
        using SublayerArray = std::tuple<std::shared_ptr<typename SubLayers::Layer>...>;
        operator SublayerArray() const { return m_tuple; }

        template <typename Tag, typename... Params>
        auto set(Params&&... params)
        {
            constexpr static std::size_t tagPos = mapTag2ID<Tag, 0>();
            using TargetType = typename std::tuple_element_t<tagPos, std::tuple<SubLayers...>>::Layer;

            std::get<tagPos>(m_tuple) = std::make_shared<TargetType>(std::forward<Params>(params)...);
            return *this;
        }

    private:
        SublayerArray m_tuple;

        template <typename Tag, std::size_t N>
        constexpr static std::size_t mapTag2ID()
        {
            if constexpr (N >= ContainerSize_v<std::tuple<SubLayers...>>)
            {
                static_assert(false, "size is too big");
            }
            else if constexpr (std::is_same_v<Tag,
                                              typename std::tuple_element_t<N, std::tuple<SubLayers...>>::Tag>)
            {
                return N;
            }
            else
            {
                return mapTag2ID<Tag, N + 1>();
            }
        }
    };

    template <typename Container>
    struct InternalResult;

    template <typename... Args>
    struct InternalResult<std::tuple<Args...>>
    {
        using type = VarTypeDict<typename Args::Tag...>;
    };

    /**
     * Input a sequence of InstantiatedSubLayers and
     * output a VarTypeDict type with the Tag of each SubLayer as the key value.
     */
    template <typename Container>
    using InternalResult_t = typename InternalResult<Container>::type;

    /**
     * @brief Obtain the corresponding input keys from the InputConnects
     * @tparam AimTag: Tag of the current layer to be processed
     * @tparam InputConnects: Container loaded with the InConnect type,
     *                       indicating the connection relationship between the
     *                       input container of the conforming layer
     *                       and the input container of the sublayer.
     * @tparam Input: Input container of the composite layer
     * @tparam Res: Input container of the sub-layer
     */
    template <typename AimTag, typename InputConnects, typename Input, typename Res>
    auto input_from_in_connect(const Input& input, Res&& res)
    {
        if constexpr (ContainerSize_v<InputConnects> == 0)
        {
            return std::forward<Res>(res);
        }
        else
        {
            // using Cur = ID2Type_t<0, >
            using Cur = ContainerHead_t<InputConnects>;
            using Tail = PopFrontFromContainer_t<InputConnects>;
            if constexpr (std::is_same_v<AimTag, typename Cur::InLayerTag>)
            {
                using InName = typename Cur::InName;
                using InLayerName = typename Cur::InLayerName;
                // put input into sublayer's input container
                auto cur = std::forward<Res>(res).template set<InLayerName>(
                    input.template get<InName>());
                return input_from_in_connect<AimTag, Tail>(input, std::move(cur));
            }
            else
            {
                return input_from_in_connect<AimTag, Tail>(input, std::forward<Res>(res));
            }
        }
    }

    /**
     * @brief Obtain the corresponding input keys from the forward propagation results of the precursor layer
     * @tparam AimTag Tag of the current layer to be processed
     * @tparam InternalConnects Container loaded with the InternalConnect type,
     *                       indicating the connection relationship between the
     *                       input container of each sub layer
     * @tparam Internal Internal container to store the intermediate results
     * @tparam Res Input container of the sub-layer
     */
    template <typename AimTag, typename InternalConnects,
              typename Internal, typename Res>
    auto input_from_internal_connect(const Internal& input, Res&& res)
    {
        if constexpr (ContainerSize_v<InternalConnects> == 0)
        {
            return std::forward<Res>(res);
        }
        else
        {
            // using Cur = ID2Type_t<0, >
            using Cur = ContainerHead_t<InternalConnects>;
            using Tail = PopFrontFromContainer_t<InternalConnects>;
            if constexpr (std::is_same_v<AimTag, typename Cur::InTag>)
            {
                using OutTag = typename Cur::OutTag;
                using OutName = typename Cur::OutName;
                using InName = typename Cur::InName;
                auto preLayer = input.template get<OutTag>();
                auto cur = std::forward<Res>(res).template set<InName>(
                    preLayer.template get<OutName>());
                return input_from_internal_connect<AimTag, Tail>(input, std::move(cur));
            }
            else
            {
                return input_from_internal_connect<AimTag, Tail>(input, std::forward<Res>(res));
            }
        }
    }

    /**
     * @brief Recurrently filling the output container
     * @tparam AimTag : Tag of the current layer to be processed
     * @tparam OutputConnects : Container loaded with the OutConnect type,
     *                       indicating the connection relationship between the
     *                       output container of each sub layer and output container of Composite layer
     * @tparam Res : Input container of the current sub-layer
     * @tparam OutCont : Output Container of Composite Layer
     */
    template <typename AimTag, typename OutputConnects, typename Res, typename OutCont>
    auto fill_output(const Res& curLayerRes, OutCont&& output)
    {
        if constexpr (ContainerSize_v<OutputConnects> == 0)
        {
            return std::forward<OutCont>(output);
        }
        else
        {
            using Cur = ContainerHead_t<OutputConnects>;
            using Tail = PopFrontFromContainer_t<OutputConnects>;
            if constexpr (std::is_same_v<AimTag, typename Cur::OutLayerTag>)
            {
                using OutLayerName = typename Cur::OutLayerName;
                using OutName = typename Cur::OutName;

                auto tmp = curLayerRes.template get<OutLayerName>();
                auto cur = std::forward<OutCont>(output).template set<OutName>(std::move(tmp));
                return fill_output<AimTag, Tail>(curLayerRes, std::move(cur));
            }
            else
            {
                return fill_output<AimTag, Tail>(curLayerRes,
                                                 std::forward<OutCont>(output));
            }
        }
    }

    template <typename AimTag, typename OutputConnects>
    struct CreateGradFromOutside
    {
        template <typename Grad, typename Res>
        static auto eval(const Grad&, const Res& res)
        {
            return res;
        }
    };

    template <typename AimTag, typename Cur, typename... Others>
    struct CreateGradFromOutside<AimTag, OutConnectContainer<Cur, Others...>>
    {
        template <typename Grad, typename Res>
        static auto eval(const Grad& grad, Res&& res)
        {
            using NextStep = CreateGradFromOutside<AimTag, OutConnectContainer<Others...>>;
            if constexpr (std::is_same_v<AimTag, typename Cur::OutLayerTag>)
            {
                using OutLayerName = typename Cur::OutLayerName;
                using OutName = typename Cur::OutName;
                auto tmp = grad.template get<OutName>();

                using OrigParamType = typename Res::template ValueType<OutLayerName>;
                if constexpr (std::is_same_v<std::remove_cvref_t<OrigParamType>,
                                             NullParameter>)
                {
                    auto cur = std::forward<Res>(res).template set<OutLayerName>(tmp);
                    return NextStep::eval(grad, std::move(cur));
                }
                else
                {
                    auto origParam = res.template get<OutLayerName>();
                    auto cur = std::forward<Res>(res).template set<OutLayerName>(origParam + tmp);
                    return NextStep::eval(grad, std::move(cur));
                }
            }
            else
            {
                return NextStep::eval(grad, std::forward<Res>(res));
            }
        }
    };

    template <typename AimTag, typename InternalConnects>
    struct CreateGradFromInternal
    {
        template <typename Internal, typename Res>
        static auto eval(const Internal&, Res&& res)
        {
            return std::forward<Res>(res);
        }
    };

    template <typename AimTag, typename Cur, typename... Others>
    struct CreateGradFromInternal<AimTag, InterConnectContainer<Cur, Others...>>
    {
        template <typename Internal, typename Res>
        static auto eval(const Internal& input, Res&& res)
        {
            using NextStep = CreateGradFromInternal<AimTag, InterConnectContainer<Others...>>;
            if constexpr (std::is_same_v<AimTag, typename Cur::OutTag>)
            {
                using InTag = typename Cur::InTag;
                using OutName = typename Cur::OutName;
                using InName = typename Cur::InName;
                auto postLayer = input.template get<InTag>();

                auto tmp = postLayer.template get<InName>();

                using OriParamType = typename Res::template ValueType<OutName>;
                if constexpr (std::is_same_v<std::remove_cvref_t<OriParamType>, NullParameter>)
                {
                    auto cur = std::forward<Res>(res).template set<OutName>(tmp);
                    return NextStep::eval(input, std::move(cur));
                }
                else
                {
                    auto oriParam = res.template get<OutName>();
                    auto cur = std::forward<Res>(res).template set<OutName>(oriParam + tmp);
                    return NextStep::eval(input, std::move(cur));
                }
            }
            else
            {
                return NextStep::eval(input, std::forward<Res>(res));
            }
        }
    };

    template <typename AimTag, typename InputConnects>
    struct FillResult
    {
        template <typename NewInternal, typename NewInput>
        static auto eval(const NewInternal&, NewInput&& res)
        {
            return std::forward<NewInput>(res);
        }
    };

    template <typename AimTag, typename Cur, typename... Others>
    struct FillResult<AimTag, InConnectContainer<Cur, Others...>>
    {
        template <typename NewInternal, typename NewInput>
        static auto eval(const NewInternal& curInternal, NewInput&& res)
        {
            using NextStep = FillResult<AimTag, InConnectContainer<Others...>>;
            if constexpr (std::is_same_v<AimTag, typename Cur::InLayerTag>)
            {
                using InName = typename Cur::InName;
                using InLayerName = typename Cur::InLayerName;

                auto tmp = curInternal.template get<InLayerName>();

                using OriParamType = typename NewInput::template ValueType<InName>;
                if constexpr (std::is_same_v<std::remove_cvref_t<OriParamType>, NullParameter>)
                {
                    auto cur = std::forward<NewInput>(res).template set<InName>(tmp);
                    return NextStep::eval(curInternal, std::move(cur));
                }
                else
                {
                    auto oriParam = res.template get<InName>();
                    auto cur = std::forward<NewInput>(res).template set<InName>(oriParam + tmp);
                    return NextStep::eval(curInternal, std::move(cur));
                }
            }
            else
            {
                return NextStep::eval(curInternal, std::forward<NewInput>(res));
            }
        }
    };
}

#endif //COMPOSE_UTILS_HPP
