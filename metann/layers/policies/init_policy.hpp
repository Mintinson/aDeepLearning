//
// Created by asus on 2025/1/13.
//

#ifndef INIT_POLICY_HPP
#define INIT_POLICY_HPP

#include "../../policy/policy.hpp"
#include <concepts>
#include <random>

namespace metann {
struct InitPolicy {
    using MajorClass = InitPolicy;
    struct OverallTypeCate;
    struct WeightTypeCate;
    struct BiasTypeCate;

    using Overall = void;
    using Weight = void;
    using Bias = void;
    struct RandEngineTypeCate;
    using RandEngine = std::mt19937;
};

template <typename Policy>
concept InitPolicyConcept = std::derived_from<Policy, InitPolicy>;

template <typename T>
struct InitializerIs : virtual InitPolicy {
    using MinorClass = OverallTypeCate;
    using Overall = T;
};

template <typename T>
struct WeightInitializerIs : virtual InitPolicy {
    using MinorClass = WeightTypeCate;
    using Weight = T;
};

template <typename T>
struct BiasInitializerIs : virtual InitPolicy {
    using MinorClass = BiasTypeCate;
    using Bias = T;
};

template <typename T>
struct RandomGeneratorIs : virtual InitPolicy {
    using MinorClass = RandEngineTypeCate;
    using RandEngine = T;
};

struct VarScaleFillerPolicy {
    using MajorClass = VarScaleFillerPolicy;

    struct DistributeTypeCate {
        struct Uniform;
        struct Norm;
    };

    using Distribute = DistributeTypeCate::Uniform;

    struct ScaleModeTypeCate {
        struct FanIn;
        struct FanOut;
        struct FanAvg;
    };

    using ScaleMode = ScaleModeTypeCate::FanAvg;
};

template <typename Policy>
concept VarScaleFillerConcept = std::derived_from<Policy, VarScaleFillerPolicy>;

struct NormVarScale : virtual VarScaleFillerPolicy {
    using MinorClass = DistributeTypeCate;
    using Distribute = DistributeTypeCate::Norm;
};

struct UniformVarScale : virtual VarScaleFillerPolicy {
    using MinorClass = DistributeTypeCate;
    using Distribute = DistributeTypeCate::Uniform;
};

struct VarScaleFanIn : virtual VarScaleFillerPolicy {
    using MinorClass = ScaleModeTypeCate;
    using ScaleMode = ScaleModeTypeCate::FanIn;
};

struct VarScaleFanOut : virtual VarScaleFillerPolicy {
    using MinorClass = ScaleModeTypeCate;
    using ScaleMode = ScaleModeTypeCate::FanOut;
};

struct VarScaleFanAvg : virtual VarScaleFillerPolicy {
    using MinorClass = ScaleModeTypeCate;
    using ScaleMode = ScaleModeTypeCate::FanAvg;
};

template <typename PolicyCont, typename Group>
struct Group2Initializer;

template <typename PolicyCont>
struct Group2Initializer<PolicyCont, InitPolicy::WeightTypeCate> {
    using type = typename details::PolicySelect_t<InitPolicy, PolicyCont>::Weight;
};

template <typename PolicyCont>
struct Group2Initializer<PolicyCont, InitPolicy::BiasTypeCate> {
    using type = typename details::PolicySelect_t<InitPolicy, PolicyCont>::Bias;
};

template <typename PolicyCont, typename SpecInit>
struct PickInitializerBySpec {
    using type = SpecInit;
};

template <typename PolicyCont>
struct PickInitializerBySpec<PolicyCont, void> {
    using type = typename details::PolicySelect_t<InitPolicy, PolicyCont>::Overall;
};

template <typename PolicyCont, typename SpecInitializer>
    requires IsPolicyContainer_v<PolicyCont>
struct PickInitializer {
    using CurInitPolicy = PlainPolicy_t<PolicyCont, PolicyContainer<>>;

    static_assert(!std::is_same_v<SpecInitializer, InitPolicy::OverallTypeCate>);

    using NewSpecInitializer = typename Group2Initializer<CurInitPolicy, SpecInitializer>::type;

    using type = typename PickInitializerBySpec<CurInitPolicy, NewSpecInitializer>::type;
};

template <typename PolicyCont, typename SpecInitializer>
    requires IsPolicyContainer_v<PolicyCont>
using PickInitializer_t = typename PickInitializer<PolicyCont, SpecInitializer>::type;
} // namespace metann

#endif // INIT_POLICY_HPP
