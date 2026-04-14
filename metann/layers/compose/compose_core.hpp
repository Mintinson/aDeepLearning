//
// Created by asus on 2025/1/19.
//

#ifndef COMPOSE_CORE_HPP
#define COMPOSE_CORE_HPP

#include <tuple>

#include "../../policy/policy.hpp"
#include "../interface_fun.hpp"
#include "../policies/update_policy.hpp"
#include "check_struct.hpp"
#include "compose_utils.hpp"
#include "containers.hpp"

// #include

namespace metann {
namespace details {
/**
 * @brief According to the topology order in CheckInternals, reordered the UnorderedSublayers,
 *      concat the ordered ones into the end of OrderedSublayers
 * @tparam OrderedSubLayers
 * @tparam UnorderedSubLayers
 * @tparam CheckInternals
 */
template <typename OrderedSubLayers, typename UnorderedSubLayers, typename CheckInternals>
struct MainLoop {
    using Ordered = OrderedSubLayers;
    using Remain = UnorderedSubLayers;
};

template <typename... OrderElems, typename... UnorderedElems, typename InterElem, typename... InterOthers>
struct MainLoop<SubLayerContainer<OrderElems...>,
                SubLayerContainer<UnorderedElems...>,
                InterConnectContainer<InterElem, InterOthers...>> {
    using CurInter = InterConnectContainer<InterElem, InterOthers...>;
    using NewInter =
        typename InternalLayerPrune<InterConnectContainer<>, CurInter, TagContainer<>, InterElem, InterOthers...>::type;
    using PostTags =
        typename InternalLayerPrune<InterConnectContainer<>, CurInter, TagContainer<>, InterElem, InterOthers...>::
            PostTags;
    static_assert(ContainerSize_v<NewInter> < ContainerSize_v<CurInter>, "Cycle exist in the compose layer");
    using SeparateByTagFun =
        SeparateByPostTag<SubLayerContainer<OrderElems...>, SubLayerContainer<UnorderedElems...>, PostTags>;
    using NewOrdered = typename SeparateByTagFun::Ordered;
    using NewUnordered = typename SeparateByTagFun::Remain;

    using Ordered = typename MainLoop<NewOrdered, NewUnordered, NewInter>::Ordered;
    using Remain = typename MainLoop<NewOrdered, NewUnordered, NewInter>::Remain;
    // using ILP = InternalLa
};

/**
 * @brief A layer that only contains LayerTag and LayerType
 * @tparam LayerTag
 * @tparam LayerType
 */
template <typename LayerTag, typename LayerType>
struct InstantiatedSubLayer {
    using Tag = LayerTag;
    using Layer = LayerType;
};
/**
 * @brief create a Container of InstantiatedSubLayer
 * @tparam Container : target container type
 * @tparam SubLayerPoliciesCont : SubLayerPolicyContainer container, whose internal elements are `SubLayerPolicies`.
 *
 * @return Container of InstantiatedSubLayer
 */
template <template <typename> class Container, typename SubLayerPoliciesCont>
struct Instantiation;

template <template <typename> class Container, typename... SubLayers>
struct Instantiation<Container, SublayerPolicyContainer<SubLayers...>> {
    template <typename T>
    struct CreateInstantiatedSubLayers {
        using Tag = typename T::Tag;
        using Policy = typename T::Policy;

        template <typename U>
        using Layer = typename T::template Layer<U>;

        using InstLayer = Layer<Policy>;

        using type = InstantiatedSubLayer<Tag, InstLayer>;
    };

    using type = Container<typename CreateInstantiatedSubLayers<SubLayers>::type...>;
};

/**
 * @brief   1. Calculate the Policy for each sublayer
            2. Output gradient detection behavior
            3. policy correction
            4. sublayer instantiation
 * @tparam PolicyCont : PolicyContainer<...>
 * @tparam OrderedSublayers : SubLayerContainer<...> (which is ordered)
 * @tparam InterConnects : InterConnectContainer<...>
 */
template <typename PolicyCont, typename OrderedSublayers, typename InterConnects>
struct SubLayerInstantiation {
    using SubLayerWithPolicy = typename GetSublayerPolicy<PolicyCont, OrderedSublayers>::type;
    using PlainPolicies = PlainPolicy_t<PolicyCont, PolicyContainer<>>;
    constexpr static bool isPlainPolicyFeedbackOut = PolicySelect_t<FeedbackPolicy, PlainPolicies>::isFeedbackOutput;
    static_assert(FeedbackOutChecker<isPlainPolicyFeedbackOut, SubLayerWithPolicy>::value,
                  "Sublayer not set feedback output, logic error!");
    using FeedbackOutUpdate = typename std::conditional_t<isPlainPolicyFeedbackOut,
                                                          std::type_identity<SubLayerWithPolicy>,
                                                          FeedbackOutSet<SubLayerWithPolicy, InterConnects>>::type;
    using type = typename Instantiation<std::tuple, FeedbackOutUpdate>::type;

    // using type = SubLayerWithPolicy;
};
}  // namespace details

/// Topological ordering that returns a container of sublayers
/// where the preceding sublayers of the container are strictly topologically ordered,
/// i.e., the pre-tendency sublayer must precede the succeeding sublayer,
/// and the final sublayer of the container is a sublayer
/// that is independently not dependent on any other sublayer.
template <typename SubLayerArray, typename InterArray>
struct TopologicalOrdering;

template <typename... SubLayerElems, typename... InterElems>
struct TopologicalOrdering<SubLayerContainer<SubLayerElems...>, InterConnectContainer<InterElems...>> {
    template <typename T>
    struct OrderPred {
        constexpr static bool value = details::TagExistInLayerComps<typename T::Tag, InterElems...>::value;
    };

    // preprocess
    // filter subLayers into two group, unordered one is in the InternalConnect (need to be reordered),
    // the other is ordered one.
    using OrderedAfterPreprocess = Filter_t<false, OrderPred, SubLayerContainer, SubLayerElems...>;
    using UnorderedAfterPreprocess = Filter_t<true, OrderPred, SubLayerContainer, SubLayerElems...>;

    using MainLoopFun =
        details::MainLoop<OrderedAfterPreprocess, UnorderedAfterPreprocess, InterConnectContainer<InterElems...>>;
    using OrderedAfterMain = typename MainLoopFun::Ordered;
    using RemainAfterMain = typename MainLoopFun::Remain;

    using type = ConcatContainer_t<OrderedAfterMain, RemainAfterMain>;
};

template <typename... Parameters>
struct ComposeTopology {
private:
    using SeparateParam = SeparateParameters<Parameters...>;

public:
    // ========================== Separate Parameters ==========================
    using SubLayers = typename SeparateParam::SubLayerRes;
    using InterConnects = typename SeparateParam::InterConnectRes;
    using InputConnects = typename SeparateParam::InConnectRes;
    using OutputConnects = typename SeparateParam::OutConnectRes;

    // /// ========== Asserts =================================================
    static_assert(!IsContainerEmpty_v<SubLayers>, "Sublayer is empty.");
    static_assert(details::SubLayerChecker<SubLayers>::IsUnique, "Two or more sub layers have same tag.");
    static_assert((details::InternalConnectChecker<InterConnects>::NoSelfCycle),
                  "Internal connections have self-connect.");
    static_assert((details::InternalConnectChecker<InterConnects>::UniqueSource),
                  "One internal input corresponds to two or more internal outputs.");
    static_assert(details::InputConnectChecker<InputConnects>::UniqueSource,
                  "One input corresponds to two or more sources.");
    static_assert(details::OutputConnectChecker<OutputConnects>::UniqueSource,
                  "One output corresponds to two or more sources.");
    static_assert(details::InternalTagInSublayer<InterConnects, SubLayers>::value,
                  "Internal connections have tags are not sublayer tags.");
    static_assert(details::InputTagInSubLayer<InputConnects, SubLayers>::value,
                  "One or more input tags are not sublayer tags.");
    static_assert(details::OutputTagInSubLayer<OutputConnects, SubLayers>::value,
                  "One or more output tags are not sublayer tags.");
    static_assert(details::SublayerTagInOtherArrays<InterConnects, InputConnects, OutputConnects, SubLayers>::value,
                  "One ore more sublayer tags not belong to any connection containers.");
    static_assert((details::UsefulInternalPostLayer<InterConnects, OutputConnects>::value),
                  "Internal output info is useless");
    static_assert((details::UsefulInputLayer<InputConnects, InterConnects, OutputConnects>::value),
                  "Input info is useless");
    using TopologicalOrdering = typename TopologicalOrdering<SubLayers, InterConnects>::type;

    template <typename PolicyCont>
    using Instances = typename details::SubLayerInstantiation<PolicyCont, TopologicalOrdering, InterConnects>::type;
    // using SubLayers = typename
};

namespace details {
template <std::size_t N, typename Save, typename SubLayersTuple>
void save_weights(Save& saver, const SubLayersTuple& sublayers) {
    if constexpr (N < std::tuple_size_v<SubLayersTuple>) {
        auto& layer = std::get<N>(sublayers);
        layer_save_weights(*layer, saver);
        save_weights<N + 1>(saver, sublayers);
    }
}

template <std::size_t N, typename Save, typename SubLayersTuple>
void neutral_invariant(SubLayersTuple& sublayers) {
    if constexpr (N < std::tuple_size_v<SubLayersTuple>) {
        auto& layer = std::get<N>(sublayers);
        layer_neutral_invariant(*layer);
        neutral_invariant<N + 1>(sublayers);
    }
}

template <std::size_t N, typename GradCollector, typename SubLayersTuple>
void grad_collect(GradCollector& collector, SubLayersTuple& sublayers) {
    if constexpr (N < std::tuple_size_v<SubLayersTuple>) {
        auto& layer = std::get<N>(sublayers);
        layer_grad_collect(*layer, collector);
        grad_collect<N + 1>(collector, sublayers);
    }
}

template <std::size_t N,
          typename InitPolicies,
          typename SubLayerInfo,
          typename Initializer,
          typename Buffer,
          typename SubLayerTuple>
void init(Initializer& initializer, Buffer& loadBuffer, std::ostream* log, SubLayerTuple& sublayers) {
    if constexpr (N < std::tuple_size_v<SubLayerTuple>) {
        auto& layer = std::get<N>(sublayers);
        using LayerInfo = std::tuple_element_t<N, SubLayerInfo>;
        using NewInitPolicy = SubPolicyPicker_t<InitPolicies, typename LayerInfo::Tag>;
        layer_init<typename LayerInfo::Layer, Initializer, Buffer, NewInitPolicy>(*layer, initializer, loadBuffer, log);
        init<N + 1, InitPolicies, SubLayerInfo>(initializer, loadBuffer, log, sublayers);
    }
}

/**
 * @brief feedforward function for composite layer
 * this function recursively went through all sublayers
 *
 * @tparam N : Sublayer number currently being processed
 * @tparam InputConnects : structural description for input connections
 * @tparam OutputConnects : structural description for output connections
 * @tparam InnerConnects : structural description for inner connections
 * @tparam SubLayerMap : Instantiation information for sublayer
 * @tparam SubLayerTuple : sublayer object  Type
 * @tparam Input : Composite Layer Input Container Type
 * @tparam Internal : Container Type to hold intermediate results
 * @tparam Output : Composite Layer Output Container Type
 */
template <std::size_t N,
          typename InputConnects,
          typename OutputConnects,
          typename InnerConnects,
          typename SubLayerMap,
          typename SubLayerTuple,
          typename Input,
          typename Internal,
          typename Output>
auto feed_forward_func(SubLayerTuple& subLayers, const Input& input, Internal&& internal, Output&& output) {
    if constexpr (N == ContainerSize_v<SubLayerTuple>) {
        return std::forward<Output>(output);
    } else {
        auto& curLayer = *(std::get<N>(subLayers));
        using Sublayer = std::remove_cvref_t<decltype(curLayer)>;
        using AimTag = typename std::tuple_element_t<N, SubLayerMap>::Tag;

        using SublayerInput = typename Sublayer::InputType;
        auto input1 = input_from_in_connect<AimTag, InputConnects>(input, SublayerInput::create());
        auto input2 = input_from_internal_connect<AimTag, InnerConnects>(internal, std::move(input1));

        // current layer feed forward
        auto res = curLayer.feedForward(std::move(input2));
        // fill the output container of composed layer
        auto newOutput = fill_output<AimTag, OutputConnects>(res, std::forward<Output>(output));
        // fill the internal container of composed layer
        auto newInternal = std::forward<Internal>(internal).template set<AimTag>(std::move(res));
        // next layer's feed forward
        return feed_forward_func<N + 1, InputConnects, OutputConnects, InnerConnects, SubLayerMap>(
            subLayers, input, std::move(newInternal), std::move(newOutput));
    }
}

template <std::size_t N,
          typename InputConnects,
          typename InterConnects,
          typename OutputConnects,
          typename SublayerMap,
          typename SublayerTuple,
          typename Grad,
          typename Internal,
          typename Input>
auto feed_backward_func(SublayerTuple& subLayers, const Grad& grad, Internal&& internal, Input&& res) {
    if constexpr (N == 0) {
        return std::forward<Input>(res);
    } else {
        auto& curLayer = *(std::get<N - 1>(subLayers));
        using Sublayer = std::remove_cvref_t<decltype(curLayer)>;
        using AimTag = typename std::tuple_element_t<N - 1, SublayerMap>::Tag;

        using SublayerOutput = typename Sublayer::OutputType;
        auto grad1 = CreateGradFromOutside<AimTag, OutputConnects>::eval(grad, SublayerOutput::create());
        auto grad2 = CreateGradFromInternal<AimTag, InterConnects>::eval(internal, std::move(grad1));

        auto curLayerRes = curLayer.feedBackward(std::move(grad2));
        auto newRes = FillResult<AimTag, InputConnects>::eval(curLayerRes, std::forward<Input>(res));
        auto newInternal = std::forward<Internal>(internal).template set<AimTag>(std::move(curLayerRes));

        return feed_backward_func<N - 1, InputConnects, InterConnects, OutputConnects, SublayerMap>(
            subLayers, grad, std::move(newInternal), std::move(newRes));
    }
}

template <typename Container>
struct IsComposeLayerUpdate;

template <template <typename...> class Cont, typename... Args>
struct IsComposeLayerUpdate<Cont<Args...>> {
    // template <typename TI>
    // struct imp
    // {
    //     constexpr static bool value = false;
    // };
    //
    // template <typename TCur, typename... TInsts>
    // struct imp<std::tuple<TCur, TInsts...>>
    // {
    //     constexpr static bool tmp = TCur::Layer::IsUpdate;
    //     constexpr static bool value = OrValue<tmp, imp<std::tuple<TInsts...>>>;
    // };

    constexpr static bool value = (... || Args::Layer::isUpdate);
};
}  // namespace details

/**
 * @brief
 * @tparam InputType : the type of the input container of the composed layer
 * @tparam OutputType : the type of the output container of the composed layer
 * @tparam PolicyCont : the policy container of the composed layer
 * @tparam KernelTopo : ComposeTopology type
 */
template <typename InputType, typename OutputType, typename PolicyCont, typename KernelTopo>
class ComposeKernel {
    using PlainPolicies = PlainPolicy_t<PolicyCont, PolicyContainer<>>;
    // a tuple containing InstantiatedSubLayer (which contains subLayer's TagName and Type)
    using InstContainer = typename KernelTopo::template Instances<PolicyCont>;
    // a tuple containing std::shared_ptr<Layer>, Layer is the subLayer
    using SubLayerArray = typename details::SubLayerArrayMaker<InstContainer>::SublayerArray;
    SubLayerArray m_subLayers;

public:
    ComposeKernel(SubLayerArray subLayers) : m_subLayers(std::move(subLayers)) {}

    /**
     * a constructor to instantiate the composed layer data.
     * for example:
     *
     * Base<CurPolicy>::createSubLayers()
     *      .template set<WeightSublayer>(params1)
     *      .template set<BiasSubLayer>(params2)
     */
    static auto createSubLayers() { return details::SubLayerArrayMaker<InstContainer>(); }

    template <typename Save>
    void saveWeights(Save& saver) {
        details::save_weights<0>(saver, m_subLayers);
    }

    void neutralInvariant() { details::neutral_invariant<0>(m_subLayers); }

    template <typename GradCollector>
    void gradCollect(GradCollector& col) {
        details::grad_collect<0>(col, m_subLayers);
    }

    template <typename Initializer, typename Buffer, typename InitPolicies = typename Initializer::PolicyCont>
    void init(Initializer& initializer, Buffer& loadBuffer, std::ostream* log = nullptr) {
        details::init<0, InitPolicies, InstContainer>(initializer, loadBuffer, log, m_subLayers);
    }

    template <typename In>
    auto feedForward(const In& input) {
        using InternalResType = details::InternalResult_t<InstContainer>;
        return details::feed_forward_func<0, typename KernelTopo::InputConnects, typename KernelTopo::OutputConnects,
                                          typename KernelTopo::InterConnects, InstContainer>(
            m_subLayers, input, InternalResType::create(), OutputType::create());
    }

    template <typename Grad>
    auto feedBackward(const Grad& grad) {
        using InternalResType = details::InternalResult_t<InstContainer>;
        return details::feed_backward_func<ContainerSize_v<InstContainer>, typename KernelTopo::InputConnects,
                                           typename KernelTopo::InterConnects, typename KernelTopo::OutputConnects,
                                           InstContainer>(m_subLayers, grad, InternalResType::create(),
                                                          InputType::create());
    }

    static constexpr bool isFeedbackOutput = details::PolicySelect_t<FeedbackPolicy, PlainPolicies>::isFeedbackOutput;
    static constexpr bool isUpdate = details::IsComposeLayerUpdate<InstContainer>::value;
};

template <template <typename> class Layer>
struct SubLayerOf;
}  // namespace metann

#endif  // COMPOSE_CORE_HPP
