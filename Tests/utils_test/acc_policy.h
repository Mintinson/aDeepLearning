//
// Created by asus on 2025/1/7.
//

#ifndef ACC_POLICY_H
#define ACC_POLICY_H

#include <iostream>

#include <metann/policy/policy.hpp>
#include <metann/utils/type_traits.hpp>
// #include "../policy/policy.hpp"
// #include "../utils/type_traits.hpp"
#include <type_traits>

struct AccPolicy {
    using MajorClass = AccPolicy;

    struct AccTypeCate {
        struct Add;
        struct Mul;
    };

    using Accu = AccTypeCate::Add;

    struct IsAveValueCate;
    static constexpr bool IsAve = false;

    struct ValueTypeCate;
    using Value = float;
};

struct MulAccuPolicy : virtual public AccPolicy {
    using MinorClass = MajorClass::AccTypeCate;
    using Accu = MinorClass::Mul;
};

struct AveAccuPolicy : virtual public AccPolicy {
    using MinorClass = MajorClass::IsAveValueCate;
    static constexpr bool IsAve = true;
};

template <typename T>
struct ValueAccuPolicy : virtual public AccPolicy {
    using MinorClass = MajorClass::ValueTypeCate;
    using Value = T;
};

template <std::derived_from<AccPolicy>... Policies>
struct Accumulator : public metann::BasePolicyExecutor<AccPolicy, Policies...> {
private:
    using BasePolicyExecutor = typename metann::BasePolicyExecutor<AccPolicy, Policies...>;
    using PolicyRes = typename BasePolicyExecutor::PolicyRes;

    static constexpr bool is_ave = PolicyRes::IsAve;
    using ValueType = typename PolicyRes::Value;
    using AccuType = typename PolicyRes::Accu;

public:
    template <std::ranges::range T>
    static auto eval(const T& in) {
        if constexpr (std::is_same_v<AccuType, AccPolicy::AccTypeCate::Add>) {
            size_t count{};
            ValueType res{};

            for (const auto& elem : in) {
                res += elem;
                count += 1;
            }
            if constexpr (is_ave) {
                return res / count;
            } else {
                return res;
            }
        } else if constexpr (std::is_same_v<AccuType, AccPolicy::AccTypeCate::Mul>) {
            size_t count{};
            ValueType res{1.0};
            for (const auto& elem : in) {
                res *= elem;
                count += 1;
            }
            if constexpr (is_ave) {
                return res / count;
            } else {
                return res;
            }
        } else {
            static_assert(false);
        }
    }
};

#endif  // ACC_POLICY_H
