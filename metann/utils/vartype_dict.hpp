#pragma once
#include <array>
#include <cstddef>
#include <memory>
#include <tuple>
#include <type_traits>

#include "type_traits.hpp"

namespace metann {
namespace details {
struct NullParameter {};

template <std::size_t N, template <typename...> class TypeContainer, typename... Types>
struct Create {
    using type = typename Create<N - 1, TypeContainer, NullParameter, Types...>::type;
};

template <template <typename...> class TypeContainer, typename... Types>
struct Create<0, TypeContainer, Types...> {
    using type = TypeContainer<Types...>;
};

template <typename NewType, size_t TypeIndex, size_t ModifiedNum, typename ProcessedTypes, typename... RemainTypes>
struct NewTupleType;

template <typename NewType,
          size_t TypeIndex,
          size_t ModifiedNum,
          template <typename...> class TypeContainer,
          typename... ModifiedTypes,
          typename CurType,
          typename... RemainTypes>
struct NewTupleType<NewType, TypeIndex, ModifiedNum, TypeContainer<ModifiedTypes...>, CurType, RemainTypes...> {
    using type = typename NewTupleType<NewType,
                                       TypeIndex,
                                       ModifiedNum + 1,
                                       TypeContainer<ModifiedTypes..., CurType>,
                                       RemainTypes...>::type;
};

template <typename NewType,
          size_t TypeIndex,
          template <typename...> class TypeContainer,
          typename... ModifiedTypes,
          typename CurType,
          typename... RemainTypes>
struct NewTupleType<NewType, TypeIndex, TypeIndex, TypeContainer<ModifiedTypes...>, CurType, RemainTypes...> {
    using type = TypeContainer<ModifiedTypes..., NewType, RemainTypes...>;
};

template <typename NewType, size_t TypeIndex, typename TypeContainer, typename... RemainTypes>
using NewTupleType_t = typename NewTupleType<NewType, TypeIndex, 0, TypeContainer, RemainTypes...>::type;
}  // namespace details

/**
 * @brief a variant type dict that can be used to be passed as a parameter.
 *
 *   example:
 *
 *  `using FParams = metann::VarTypeDict<A, B, Weight>;`
 *  `foo(FParams::create().set<A>(3.5).set<B>(2.4).set<Weight>(0.25));`;
 */
template <typename... Types>
struct VarTypeDict {
    template <typename... ValuesTypes>
    struct Values {
    public:
        Values() = default;

        explicit Values(std::array<std::shared_ptr<void>, sizeof...(ValuesTypes)>&& input)
            : m_tuple(std::move(input)) {}

        template <typename KType, typename VType>
        auto set(VType&& value) && {
            constexpr static size_t keyPos = Key2ID_v<KType, Types...>;
            using RawVType = std::decay_t<VType>;
            RawVType* tmp{new RawVType{std::forward<VType>(value)}};
            m_tuple[keyPos] = std::shared_ptr<void>(tmp, [](void* ptr) {
                auto* realPtr = static_cast<RawVType*>(ptr);
                delete realPtr;
            });
            using NewType = details::NewTupleType_t<RawVType, keyPos, Values<>, ValuesTypes...>;
            return NewType(std::move(m_tuple));
        }

        template <typename KType>
        auto& get() const {
            constexpr static size_t keyPos = Key2ID_v<KType, Types...>;
            auto* ptr = m_tuple[keyPos].get();

            using ReturnType = ID2Type_t<keyPos, ValuesTypes...>;
            return *(static_cast<ReturnType*>(ptr));
        }

        template <typename TTag>
        using ValueType = ID2Type_t<Key2ID_v<TTag, Types...>, ValuesTypes...>;

    private:
        std::array<std::shared_ptr<void>, sizeof...(ValuesTypes)> m_tuple;
    };

public:
    static auto create() {
        // using namespace
        using type = typename details::Create<sizeof...(Types), Values>::type;
        return type();
    }
};

template <typename... Types>
struct VarTypeDictTuple {
    template <typename... ValuesTypes>
    struct Values {
    public:
        Values() = default;

        explicit Values(std::tuple<ValuesTypes...>&& input) noexcept : m_tuple(std::move(input)) {}

        template <typename KType, typename VType>
        auto set(VType&& value) && {
            using RawVType = std::decay_t<VType>;

            auto newTuple = std::tuple_cat(m_tuple, std::tuple<RawVType>(value));
            return Values<ValuesTypes..., RawVType>(std::move(newTuple));
        }

        template <typename KType>
        const auto& get() const {
            constexpr static size_t keyPos = Key2ID_v<KType, Types...>;
            return std::get<keyPos>(m_tuple);
        }

    private:
        using TupleType = std::tuple<ValuesTypes...>;
        TupleType m_tuple;
    };

public:
    static auto create() { return Values<>{}; }
};

// add a new Key Type in the `VarTypeDict(Tuple)`，fail if same key inserted
template <typename OldDict, typename... Types>
struct AddItem {
    using type = OldDict;
};

template <typename... OldTypes, template <typename...> class OldDict, typename T, typename... Types>
struct AddItem<OldDict<OldTypes...>, T, Types...> {
    static_assert(!IsInPack<T, OldTypes...>::value, "Dict doesn't support multiples keys");
    using type = typename AddItem<OldDict<OldTypes..., T>, Types...>::type;
};

template <typename... OldTypes, template <typename...> class OldDict, typename T>
struct AddItem<OldDict<OldTypes...>, T> {
    static_assert(!IsInPack<T, OldTypes...>::value, "Dict doesn't support multiples keys");
    using type = OldDict<OldTypes..., T>;
};

template <typename OldDict, typename... Types>
using AddItem_t = typename AddItem<OldDict, Types...>::type;

// delete a list of key types in the `VarTypeDict(Tuple)`，
template <typename OldDict, typename... Types>
struct DelItem {
    using type = OldDict;
};

template <typename... OldTypes, template <typename...> class OldDict, typename TargetType, typename... OtherTypes>
struct DelItem<OldDict<OldTypes...>, TargetType, OtherTypes...> {
private:
    // using SubType = typename DelItem<OldDict<OldTypes...>, OtherTypes...>::type;
    using NewType = RemoveTypeFromContainer_t<TargetType, OldDict<OldTypes...>>;

public:
    // using type = typename DelItem<SubType, TargetType>::type;
    using type = typename DelItem<NewType, OtherTypes...>::type;
};

template <typename OldDict, typename... Types>
using DelItem_t = typename DelItem<OldDict, Types...>::type;
}  // namespace metann
