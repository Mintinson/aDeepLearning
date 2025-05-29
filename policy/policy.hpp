/**
 * @file policy.hpp
 * @brief Policy system implementation for metann framework
 *
 * This file implements a flexible policy-based design system that allows
 * composition and specialization of behaviors through policy classes.
 */

#ifndef POLICY_HPP
#define POLICY_HPP
#include "../utils/type_traits.hpp"
#include <concepts>
#include <type_traits>

namespace metann
{
    /**
     * @brief Concept that defines requirements for policy classes
     * 
     * A policy class must define MajorClass and MinorClass types to be valid
     */
    template <typename T>
    concept PolicyClass = requires
    {
        typename T::MajorClass;
        typename T::MinorClass;
    };

    /**
     * @brief Container for storing multiple policy types
     */
    template <typename... Types>
    struct PolicyContainer;

    /**
     * @brief Container for storing sublayer specific policies
     */
    template <typename Sublayer, typename... Types>
    struct SubPolicyContainer;

    /**
     * @brief Type trait to check if a type is a PolicyContainer
     */
    template <typename T>
    struct IsPolicyContainer : std::false_type
    {
    };

    template <typename... Types>
    struct IsPolicyContainer<PolicyContainer<Types...>> : std::true_type
    {
    };

    template <typename T>
    constexpr bool IsPolicyContainer_v = IsPolicyContainer<T>::value;

    /**
     * @brief Type trait to check if a type is a SubPolicyContainer
     */
    template <typename T>
    struct IsSubPolicyContainer : std::false_type
    {
    };

    template <typename NewLayer, typename... Types>
    struct IsSubPolicyContainer<SubPolicyContainer<NewLayer, Types...>> : std::true_type
    {
    };

    template <typename T>
    constexpr bool IsSubPolicyContainer_v = IsSubPolicyContainer<T>::value;

    namespace details
    {
        /**
         * @brief Helper to check for conflicting minor classes in policies
         */
        template <typename MinorClass, typename... Policies>
        struct MinorDedup : std::true_type
        {
        };

        template <typename MinorClass, typename CurPolicies, typename... Policies>
        struct MinorDedup<MinorClass, CurPolicies, Policies...>
            : std::conjunction<std::negation<std::is_same<
            MinorClass, typename CurPolicies::MinorClass>>,
            MinorDedup<MinorClass, Policies...>>
        {
        };

        /**
         * @brief Validates that no minor class conflicts exist in a PolicyContainer
         */
        template <typename PolicyCont>
        struct MinorCheck : std::true_type
        {
        };

        template <typename CurPolicy, typename... Policies>
        struct MinorCheck<PolicyContainer<CurPolicy, Policies...>>
        {
        private:
            static constexpr bool curCheck = MinorDedup<typename CurPolicy::MinorClass, Policies...>::value;

        public:
            static constexpr bool value = std::conjunction_v<std::bool_constant<curCheck>,
                std::bool_constant<MinorCheck<PolicyContainer<Policies
                ...>>::value>>;
        };

        template <typename... Types>
        constexpr bool MinorCheck_v = MinorCheck<Types...>::value;

        /// ====================== Split Major Class ============================
        template <typename Container, typename TMajorClass, typename... TP>
        struct MajorFilter
        {
            using type = Container;
        };

        template <typename... FilteredPolicies, typename MajorClass,
            typename CurPolicy, typename... Policies>
        struct MajorFilter<PolicyContainer<FilteredPolicies...>, MajorClass,
            CurPolicy, Policies...>
        {
            template <typename CurMajor, typename TDummy = void>
            struct _impl
            {
                using type = typename MajorFilter<PolicyContainer<FilteredPolicies...>, MajorClass, Policies...>::type;
            };

            template <typename TDummy>
            struct _impl<MajorClass, TDummy>
            {
                using type = typename MajorFilter<PolicyContainer<FilteredPolicies..., CurPolicy>,
                    MajorClass, Policies...>::type;
            };

            using type = typename _impl<typename CurPolicy::MajorClass>::type;
        };

        template <typename MCO, typename TMajorClass, typename... TP>
        using MajorFilter_t = typename MajorFilter<MCO, TMajorClass, TP...>::type;

        template <typename PolicyContain>
        struct PolicySelRes;

        template <PolicyClass Policy>
        struct PolicySelRes<PolicyContainer<Policy>> : public Policy
        {
        };

        template <PolicyClass CurPolicy, PolicyClass... Policies>
        struct PolicySelRes<PolicyContainer<CurPolicy, Policies...>>
            : public CurPolicy, public PolicySelRes<PolicyContainer<Policies...>>
        {
        };

        template <typename MajorClass, typename PolicyContainer>
        struct Selector;

        template <typename MajorClass, typename... Policies>
        struct Selector<MajorClass, PolicyContainer<Policies...>>
        {
            using FilterContainer = MajorFilter_t<PolicyContainer<>, MajorClass, Policies...>;
            static_assert(
                MinorCheck_v<FilterContainer>, "Minor Class set conflict!");

            using type = std::conditional_t<IsContainerEmpty_v<FilterContainer>, MajorClass, PolicySelRes<
                FilterContainer>>;
        };

        template <typename MajorClass, typename PolicyContainer>
        using PolicySelect_t = typename Selector<MajorClass, PolicyContainer>::type;
    } // namespace details

    /**
     * @brief Base class for policy executors
     *
     * @tparam BasePolicy The base policy type
     * @tparam Policies Additional policy types that derive from BasePolicy
     */
    template <typename BasePolicy, std::derived_from<BasePolicy>... Policies>
    class BasePolicyExecutor
    {
    protected:
        using PolicyContain = PolicyContainer<Policies...>;
        using PolicyRes = details::PolicySelect_t<BasePolicy, PolicyContain>;
    };

    /** remove all SubPolicyContainer */
    template <typename PolicyCont, typename ResContainer>
    struct PlainPolicy
    {
        using type = ResContainer;
    };

    template <typename CurPolicy, typename... Policies, typename... FilteredPolicies>
    struct PlainPolicy<PolicyContainer<CurPolicy, Policies...>,
        PolicyContainer<FilteredPolicies...>>
    {
        using NewFiltered = std::conditional_t<IsSubPolicyContainer_v<CurPolicy>,
            PlainPolicy<PolicyContainer<Policies...>,
            PolicyContainer<FilteredPolicies...>>,
            PlainPolicy<PolicyContainer<Policies...>,
            PolicyContainer<FilteredPolicies..., CurPolicy>>>;
        using type = typename NewFiltered::type;
    };

    /** remove all SubPolicyContainer in the PolicyContainer */
    template <typename PolicyCont, typename ResContainer>
    using PlainPolicy_t = typename PlainPolicy<PolicyCont, ResContainer>::type;

    namespace details
    {
        template <typename PolicyCont, typename LayerName>
        struct ExtractSubPolicy
        {
            using type = PolicyContainer<>;
        };

        template <typename Cur, typename... Policies, typename LayerName>
        struct ExtractSubPolicy<PolicyContainer<Cur, Policies...>, LayerName>
        {
            using type = typename ExtractSubPolicy<PolicyContainer<Policies...>, LayerName>::type;
        };

        template <typename... Cur, typename... Res, typename LayerName>
        struct ExtractSubPolicy<PolicyContainer<SubPolicyContainer<LayerName, Cur...>, Res...>, LayerName>
        {
            using type = PolicyContainer<Cur...>;
        };

        template <typename PolicyCont, typename LayerName>
        using ExtractSubPolicy_t = typename ExtractSubPolicy<PolicyCont, LayerName>::type;
    } // namespace details
    template <typename PolicyCont, typename LayerName>
    struct SubPolicyPicker
    {
    private:
        using tmp1 = details::ExtractSubPolicy_t<PolicyCont, LayerName>;
        using tmp2 = PlainPolicy_t<PolicyCont, PolicyContainer<>>;

    public:
        using type = UniqueFromContainer_t<ConcatContainer_t<tmp1, tmp2>>;
    };

    // TODO: Draft
    template <typename Policy, typename Cont>
    struct PolicyExist;

    template <typename Policy, typename... Policies>
    struct PolicyExist<Policy, PolicyContainer<Policies...>>
    {
        template <typename P1, typename P2>
        struct PolicyEqual
        {
            using MJ1 = typename P1::MajorClass;
            using MJ2 = typename P2::MajorClass;
            using MI1 = typename P1::MinorClass;
            using MI2 = typename P2::MinorClass;
            constexpr static bool tmp1 = std::is_same_v<MJ1, MJ2>;
            constexpr static bool value = tmp1 && std::is_same_v<MI1, MI2>;
        };

        constexpr static bool value = Any_v<Policy, PolicyEqual, Policies...>;
    };

    template <typename TLayerName, typename... T1, typename... T2, typename TPolicy>
    struct PolicyExist<TPolicy, PolicyContainer<SubPolicyContainer<TLayerName, T1...>, T2...>>
    {
        constexpr static bool value = PolicyExist<PolicyContainer<T2...>, TPolicy>::value;
    };

    template <typename TPolicy, typename TArray>
    constexpr static bool PolicyExist_v = PolicyExist<TPolicy, TArray>::value;

    namespace details
    {
        template <typename TProcessingCont, typename TFiltedCont, typename TCompPoliCont>
        struct PolicyDeriveFil_;

        template <typename TCompPoliCont, typename TCur, typename... TProcessings, typename... TProcessed>
        struct PolicyDeriveFil_<PolicyContainer<TCur, TProcessings...>,
            PolicyContainer<TProcessed...>,
            TCompPoliCont>
        {
            constexpr static bool dupe = PolicyExist_v<TCur, TCompPoliCont>;
            using TNewFiltered = std::conditional_t<dupe,
                PolicyContainer<TProcessed...>,
                PolicyContainer<TProcessed..., TCur>>;
            using type = typename PolicyDeriveFil_<PolicyContainer<TProcessings...>,
                TNewFiltered,
                TCompPoliCont>::type;
        };

        template <typename TCompPoliCont, typename... TProcessed>
        struct PolicyDeriveFil_<PolicyContainer<>,
            PolicyContainer<TProcessed...>,
            TCompPoliCont>
        {
            using type = PolicyContainer<TProcessed...>;
        };


        template <typename TSubPolicies, typename TParentPolicies>
        struct PolicyDerive
        {
            using TFiltered = typename PolicyDeriveFil_<TParentPolicies,
                PolicyContainer<>,
                TSubPolicies>::type;
            // template <typename T>
            // struct Pred
            // {
            //     constexpr static bool value = PolicyExist_v<T, TSubPolicies>;
            // };
            // using Filtered = FilterFromContainer_t<false, Pred, TSubPolicies>;
            using type = typename ConcatContainer<TSubPolicies, TFiltered>::type;
        };

        template <typename TPolicyContainer, typename TLayerName>
        struct SPP
        {
            using type = PolicyContainer<>;
        };

        template <typename TCur, typename... TPolicies, typename TLayerName>
        struct SPP<PolicyContainer<TCur, TPolicies...>, TLayerName>
        {
            using type = typename SPP<PolicyContainer<TPolicies...>, TLayerName>::type;
        };

        template <typename... TCur, typename... TPolicies, typename TLayerName>
        struct SPP<PolicyContainer<SubPolicyContainer<TLayerName, TCur...>, TPolicies...>, TLayerName>
        {
            using type = PolicyContainer<TCur...>;
        };
    }

    template <typename TPolicyContainer, typename TLayerName>
    struct SubPolicyPicker_
    {
        using tmp1 = typename details::SPP<TPolicyContainer, TLayerName>::type; // SubPolicies of Sub LayerName
        using tmp2 = PlainPolicy_t<TPolicyContainer, PolicyContainer<>>; // Plain Policy of SupLayer
        using type = typename details::PolicyDerive<tmp1, tmp2>::type;
        // concat the Policy (only the not conflict policies can be concated)
    };

    // TODO: above draft

    /** extract all Policies of LayerName into a new PolicyContainer*/
    template <typename PolicyCont, typename LayerName>
    using SubPolicyPicker_t = typename SubPolicyPicker_<PolicyCont, LayerName>::type;

    template <typename NewPolicy, typename OriContainer>
    struct ChangePolicy;

    template <typename NewPolicy, typename... Policies>
    struct ChangePolicy<NewPolicy, PolicyContainer<Policies...>>
    {
    private:
        using newMajor = typename NewPolicy::MajorClass;
        using newMinor = typename NewPolicy::MinorClass;

        template <typename CurPolicy>
        struct Pred
        {
            static constexpr bool value = IsSubPolicyContainer_v<CurPolicy>
                || !std::conjunction_v<std::is_same<typename CurPolicy::MajorClass, newMajor>,
                std::is_same<typename CurPolicy::MinorClass, newMinor>>;
        };

    public:
        using type = ConcatContainer_t<Filter_t<true, Pred, PolicyContainer, Policies...>, PolicyContainer<NewPolicy>>;
    };

    /// remove the Policy that belongs to the same category of NewPolicy, then append the NewPolicy to the Container
    template <typename NewPolicy, typename OriContainer>
    using ChangePolicy_t = typename ChangePolicy<NewPolicy, OriContainer>::type;

    template <template <typename PolicyCont> class T, typename... Policies>
    using InjectPolicy_t = T<PolicyContainer<Policies...>>;
} // namespace metann

#endif // POLICY_HPP
