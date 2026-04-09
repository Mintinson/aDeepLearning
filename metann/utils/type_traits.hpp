//
// Created by asus on 2025/1/7.
//

#ifndef TYPE_TRAITS_HPP
#define TYPE_TRAITS_HPP

#include <type_traits>

namespace metann
{
    template <typename T>
    struct ContainerSize;

    template <template <typename...> class Container, typename... Args>
    struct ContainerSize<Container<Args...>>
    {
        constexpr static std::size_t value = sizeof...(Args);
    };

    /**
     * @brief return the size of the container
     * @tparam T : a tuple-like container
     * @return `std::size_t` the size of the container
     */
    template <typename T>
    constexpr std::size_t ContainerSize_v = ContainerSize<T>::value;

    // if Container is empty, return true
    template <typename Container>
    struct IsContainerEmpty : std::true_type
    {
    };

    template <template <typename...> class Container, typename T, typename... Types>
    struct IsContainerEmpty<Container<T, Types...>> : std::false_type
    {
    };

    /**
     * @brief whether the container is empty
     * @tparam Container a tuple-like container
     */
    template <typename Container>
    constexpr bool IsContainerEmpty_v = IsContainerEmpty<Container>::value;

    /// whether the T is in the Types...
    template <typename T, typename... Types>
    struct IsInPack : std::disjunction<std::is_same<T, Types>...>
    {
    };
    /// whether the T is in the Types...
    template <typename T, typename... Types>
    constexpr bool IsInPack_v = IsInPack<T, Types...>::value;

    /// whether the T is in the Type Container...
    template <typename T, typename Container>
    struct isInContainer;

    template <typename T, template <typename...> class Container, typename... Types>
    struct isInContainer<T, Container<Types...>> : IsInPack<T, Types...>
    {
    };
    /// whether the T is in the Container...
    template <typename T, typename Container>
    constexpr bool isInContainer_v = isInContainer<T, Container>::value;

    template <typename T, template <typename, typename> class Operator, typename... Types>
    struct ForAll : std::conjunction<Operator<T, Types>...>
    {
    };

    /**
     * @brief check whether all types in the Types satisfy the Operator
     * @tparam T: The first parameter of Operator
     * @tparam Operator: Predicate Operator that take the T and the Type in Types, return a bool value
     * @tparam ...Types: All Type that need to be checked
     */
    template <typename T, template <typename, typename> class Operator, typename... Types>
    constexpr bool ForAll_v = ForAll<T, Operator, Types...>::value;

    template <typename T, template <typename, typename> class Operator, typename Container>
    struct ForAllInContainer;
    template <typename T, template <typename, typename> class Operator, template <typename...> class Container, typename... Types>
    struct ForAllInContainer<T, Operator, Container<Types...>>
    {
        constexpr static bool value = ForAll<T, Operator, Types...>::value;
    };
    /**
     * @brief check whether all types in the Type Container satisfy the Operator
     * @tparam T: The first parameter of Operator
     * @tparam Operator: Predicate Operator that take the T and the Type in Types, return a bool value
     * @tparam ...Types: Type Container that need to be checked
     */
    template <typename T, template <typename, typename> class Operator, typename Container>
    constexpr bool ForAllInContainer_v = ForAllInContainer<T, Operator, Container>::value;

    template <typename T, template <typename, typename> class Operator, typename... Types>
    struct Any : std::disjunction<Operator<T, Types>...>
    {
    };
    /**
     * @brief check if any types in the Types satisfy the Operator
     * @tparam T: The first parameter of Operator
     * @tparam Operator: Predicate Operator that take the T and the Type in Types, return a bool value
     * @tparam ...Types: All Type that need to be checked
     */
    template <typename T, template <typename, typename> class Operator, typename... Types>
    constexpr bool Any_v = Any<T, Operator, Types...>::value;

    template <typename T, template <typename, typename> class Operator, typename Container>
    struct AnyInContainer;
    template <typename T, template <typename, typename> class Operator, template <typename...> class Container, typename... Types>
    struct AnyInContainer<T, Operator, Container<Types...>>
    {
        constexpr static bool value = Any<T, Operator, Types...>::value;
    };
    /**
     * @brief check if any types in the Type Container satisfy the Operator
     * @tparam T: The first parameter of Operator
     * @tparam Operator: Predicate Operator that take the T and the Type in Types, return a bool value
     * @tparam ...Types: Type Container that need to be checked
     */
    template <typename T, template <typename, typename> class Operator, typename Container>
    constexpr bool AnyInContainer_v = AnyInContainer<T, Operator, Container>::value;


    //concat two tuple-like into a longer Container
    template <typename Cont1, typename Cont2>
    struct ConcatContainer;

    template <typename... Params1, template <typename...> class Container, typename... Params2>
    struct ConcatContainer<Container<Params1...>, Container<Params2...>>
    {
        using type = Container<Params1..., Params2...>;
    };
    /**
     * @brief Concat two tuple-like Container into a longer Container
     * @tparam Cont1 : the first tuple-like Container
     * @tparam Cont2 : the second tuple-like Container
     * @note: both container shuold have the same underly type
     */
    template <typename Cont1, typename Cont2>
    using ConcatContainer_t = typename ConcatContainer<Cont1, Cont2>::type;

    template <typename T, typename Container>
    struct RemoveTypeFromContainer;

    template <typename T, template <typename...> class Container, typename... Types>
    struct RemoveTypeFromContainer<T, Container<Types...>>
    {
        using type = Container<Types...>;
    };

    template <typename T, template <typename...> class Container, typename U, typename... Types>
    struct RemoveTypeFromContainer<T, Container<U, Types...>>
    {
    private:
        using ResType = typename RemoveTypeFromContainer<T, Container<Types...>>::type;

    public:
        using type = std::conditional_t<std::is_same_v<U, T>,
            ResType,
            ConcatContainer_t<Container<U>, ResType>>;
    };

    /**
     * @brief remove all occurrence of T in the Type Container
     * @tparam T: target type
     * @tparam Container: target container
     */
    template <typename T, typename Container>
    using RemoveTypeFromContainer_t = typename RemoveTypeFromContainer<T, Container>::type;

    // Given two Containers, Transfer the Contained Types from `From` to `To`
    template <typename From, template <typename...> class To>
    struct TransferTo;

    template <template <typename...> class From, template <typename...> class To, typename... Args>
    struct TransferTo<From<Args...>, To>
    {
        using type = To<Args...>;
    };
    /**
     * @brief Given two Containers, Transfer the Contained Types from `From` to `To`
     * @tparam From: the source Container
     * @tparam To: the target Container
     */
    template <typename From, template <typename...> class To>
    using TransferTo_t = typename TransferTo<From, To>::type;

    template <typename From, template <typename...> class To, template <typename> class Operator>
    struct TransformTo;

    template <template <typename...> class From,
        template <typename...> class To,
        template <typename> class Operator,
        typename... Args>
    struct TransformTo<From<Args...>, To, Operator>
    {
        using type = To<typename Operator<Args>::type...>;
    };
    /**
     * @brief Firser transform the types in the From Container, then transfer the transformed types to the To Container
     * @tparam From : source container
     * @tparam To : target container
     * @tparam Operator : the transform Operator, the Operator should have a type member
     */
    template <typename From, template <typename...> class To, template <typename> class Operator>
    using TransformTo_t = typename TransformTo<From, To, Operator>::type;

    template <template <typename...> class Container, typename... Types>
    struct Unique
    {
        using type = Container<Types...>;
    };

    template <template <typename...> class Container, typename Cur, typename... Rest>
    struct Unique<Container, Cur, Rest...>
    {
    private:
        using ResType = RemoveTypeFromContainer_t<Cur, typename Unique<Container, Rest...>::type>;

    public:
        using type = ConcatContainer_t<Container<Cur>, ResType>;
    };
    /**
     * @brief remove all duplicate types in the Types, then put the result in the Container
     * @tparam Container : result container
     * @tparam ...Types : target types list
     */
    template <template <typename...> class Container, typename... Types>
    using Unique_t = typename Unique<Container, Types...>::type;


    template <typename Container>
    struct UniqueFromContainer;

    template <template <typename...> class Container, typename... Types>
    struct UniqueFromContainer<Container<Types...>>
    {
        using type = Unique_t<Container, Types...>;
    };
    /**
     * @brief given a container containing any types, return a container that has unique types
     * @tparam Container
     */
    template <typename Container>
    using UniqueFromContainer_t = typename UniqueFromContainer<Container>::type;

    template <template <typename, typename> class Operator, typename... Types>
    struct IsGeneralUnique : std::true_type
    {
    };

    template <template <typename, typename> class Operator, typename Cur, typename... Types>
    struct IsGeneralUnique<Operator, Cur, Types...>
    {
        constexpr static bool value = !Any_v<Cur, Operator, Types...>&& IsGeneralUnique<Operator, Types...>::value;
    };
    /**
     * @brief A General Unique Checker, if Operator is std::is_same, it will check if all types are unique
     * @tparam ...Types : target types list
     */
    template <template <typename, typename> class Operator, typename... Types>
    constexpr bool IsGeneralUnique_v = IsGeneralUnique<Operator, Types...>::value;


    template <template <typename, typename> class Operator, class Container>
    struct IsGeneralUniqueFromContainer;
    template <template <typename, typename> class Operator, template <typename...> class Container, typename...Types>
    struct IsGeneralUniqueFromContainer<Operator, Container<Types...>>
    {
        constexpr static bool value = IsGeneralUnique_v<Operator, Types...>;
    };
    /**
     * @brief A General Unique Checker, if Operator is std::is_same, it will check if all types are unique
     * @tparam Container : target container
     */
    template <template <typename, typename> class Operator, typename Container>
    constexpr bool IsGeneralUniqueFromContainer_v = IsGeneralUniqueFromContainer<Operator, Container>::value;


    template <typename... Types>
    struct IsUnique : IsGeneralUnique<std::is_same, Types...>
    {
    };
    /**
     * @brief Check if all types are unique
     * @tparam ...Types
     */
    template <typename... Types>
    constexpr bool IsUnique_v = IsUnique<Types...>::value;

    template <typename Container>
    struct IsUniqueFromContainer;

    template <template <typename...> class Container, typename... Types>
    struct IsUniqueFromContainer<Container<Types...>> : IsUnique<Types...>
    {
    };
    /**
     * @brief Check if all types in the Container are unique
     * @tparam Container
     */
    template <typename Container>
    constexpr bool IsUniqueFromContainer_v = IsUniqueFromContainer<Container>::value;

    template <typename SingleContainer>
    struct InnerType;

    template <template <typename> class Container, typename T>
    struct InnerType<Container<T>>
    {
        using type = T;
    };
    /**
     * @brief extract the inner type of a single container
     * @tparam SingleContainer
     */
    template <typename SingleContainer>
    using InnerType_t = typename InnerType<SingleContainer>::type;

    template <typename Container>
    struct OutterType;
    template <template <typename...> class Container, typename... Args>
    struct OutterType<Container<Args...>>
    {
        template <typename...>
        using type = Container<Args...>;
    };


    template <bool keepTrue, template <typename> class Pred,
        template <typename...> class Container, typename... Args>
    struct Filter
    {
        using type = Container<>;
    };

    template <bool keepTrue, template <typename> class Pred,
        template <typename...> class Container, typename Cur, typename... Args>
    struct Filter<keepTrue, Pred, Container, Cur, Args...>
    {
    private:
        using ResType = typename Filter<keepTrue, Pred, Container, Args...>::type;

    public:
        using type = std::conditional_t<Pred<Cur>::value == keepTrue,
            ConcatContainer_t<
            Container<Cur>, ResType>,
            ResType>;
    };

    /**
     * @brief Filter the types list by Pred, then put the result in the Container
     * @tparam keepTrue : if true, keep the types that Pred is true; if false, keep the types that Pred is false
     * @tparam Pred : the Predicate
     * @tparam Container : result container
     * @tparam ...Args type lists
     */
    template <bool keepTrue, template <typename> class Pred,
        template <typename...> class Container, typename... Args>
    using Filter_t = typename Filter<keepTrue, Pred, Container, Args...>::type;

    template <bool keepTrue, template <typename> class Pred, typename Container>
    struct FilterFromContainer;

    template <bool keepTrue, template <typename> class Pred,
        template <typename...> class Container, typename... Args>
    struct FilterFromContainer<keepTrue, Pred, Container<Args...>>
    {
        using type = Filter_t<keepTrue, Pred, Container, Args...>;
    };

    /**
     * @brief Filter the types in the Container
     * @tparam Container: Container that containes types
     * @tparam keepTrue: if true, keep the types that Pred is true; if false, keep the types that Pred is false
     * @tparam Pred: the Predicate
     */
    template <bool keepTrue, template <typename> class Pred, typename Container>
    using FilterFromContainer_t = typename FilterFromContainer<keepTrue, Pred, Container>::type;

    /**
     * @brief Put the True Type into the TrueContain, and put the False Type into the FalseContain
     * @tparam Pred: The Predicate
     * @tparam TrueContain: the result container that contains the True Type
     * @tparam FalseContain: the result container that contains the False Type
     * @return TrueType: the true types list
     * @return FalseType: the false types list
     */
    template <template <typename> class Pred,
        template <typename...> class TrueContain,
        template <typename...> class FalseContain,
        typename... Args>
    struct SeparateBy
    {
        using TrueType = Filter_t<true, Pred, TrueContain, Args...>;
        using FalseType = Filter_t<false, Pred, FalseContain, Args...>;
    };

    /**
     * @brief Put the True Type into the TrueContain, and put the False Type into the FalseContain
     * @tparam Pred: The Predicate
     * @tparam TrueContain: the result container that contains the True Type
     * @tparam FalseContain: the result container that contains the False Type
     * @tparam TargetContain: the Container that contains the types
     * @return TrueType: the true types list
     * @return FalseType: the false types list
     */
    template <template <typename> class Pred,
        template <typename...> class TrueContain,
        template <typename...> class FalseContain,
        typename TargetContain>
    struct SeparateByFromContainer;

    template <template <typename> class Pred,
        template <typename...> class TrueContain,
        template <typename...> class FalseContain,
        template <typename...> class TargetContain,
        typename... Args>
    struct SeparateByFromContainer<Pred, TrueContain, FalseContain, TargetContain<Args...>>
    {
        using TrueType = Filter_t<true, Pred, TrueContain, Args...>;
        using FalseType = Filter_t<false, Pred, FalseContain, Args...>;
    };

    // Given a type list and a target type, find the index of the type in type list
    // if the target type is not in the type list, static assert will fail
    template <typename T, typename... Types>
    struct Key2ID
    {
        // ReSharper disable once CppStaticAssertFailure
        static_assert(false, "T is not in the types");
        // static constexpr std::size_t value = 0;
    };

    template <typename T, typename U, typename... Types>
    struct Key2ID<T, U, Types...>
    {
        static constexpr std::size_t value = Key2ID<T, Types...>::value + 1;
    };

    template <typename T, typename... Types>
    struct Key2ID<T, T, Types...>
    {
        static constexpr std::size_t value = 0;
    };
    /**
     * @brief return the index of the type in the type list
     * @tparam T : target type
     * @tparam ...Types : type list
     */
    template <typename T, typename... Types>
    constexpr std::size_t Key2ID_v = Key2ID<T, Types...>::value;

    template <typename T, typename Cont>
    struct Key2IDFromContainer;

    template <typename T, template <typename> class Container, typename... Args>
    struct Key2IDFromContainer<T, Container<Args...>> : Key2ID<T, Args...>
    {
    };
    /**
     * @brief return the index of the type in the Container
     * @tparam T
     * @tparam Cont
     */
    template <typename T, typename Cont>
    constexpr std::size_t Key2IDFromContainer_v = Key2IDFromContainer<T, Cont>::value;



    // given an index and type list, return the type corresponding to the index
    template <std::size_t N, typename... Types>
    struct ID2Type;

    template <typename T, typename... Types>
    struct ID2Type<0, T, Types...>
    {
        using type = T;
    };

    template <std::size_t N, typename T, typename... Types>
    struct ID2Type<N, T, Types...>
    {
        using type = typename ID2Type<N - 1, Types...>::type;
    };

    /**
     * @brief Given an index and type list, return the type corresponding to the index
     * @tparam N : the index
     * @tparam ...Types : the type list
     */
    template <std::size_t N, typename... Types>
    using ID2Type_t = typename ID2Type<N, Types...>::type;

    template <std::size_t N, typename Cont>
    struct ID2TypeFromContainer;

    template <std::size_t N, template <typename> class Container, typename... Args>
    struct ID2TypeFromContainer<N, Container<Args...>> : ID2Type<N, Args...>
    {
    };
    /**
     * @brief Given an index and a Container, return the type corresponding to the index
     * @tparam N : the index
     * @tparam Cont : the Container
     */
    template <std::size_t N, typename Cont>
    using ID2TypeFromContainer_t = typename ID2TypeFromContainer<N, Cont>::type;


    template <typename Container>
    struct ContainerHead;

    template <template <typename...> class Container, typename Head, typename... Args>
    struct ContainerHead<Container<Head, Args...>>
    {
        using type = Head;
    };

    /**
     * @brief return the first type in the Container
     * @tparam Container
     */
    template <typename Container>
    using ContainerHead_t = typename ContainerHead<Container>::type;

    template <typename Container>
    struct ContainerTail;

    template <template <typename...> class Container, typename Cur, typename... Args>
    struct ContainerTail<Container<Cur, Args...>>
    {
        using type = ID2Type_t<sizeof...(Args), Cur, Args...>;
    };

    /**
     * @brief return the last type of the Container
     * @tparam Container
     */
    template <typename Container>
    using ContainerTail_t = typename ContainerTail<Container>::type;


    template <typename Container>
    struct PopFrontFromContainer;

    template <template <typename...> class Container, typename Head, typename... Args>
    struct PopFrontFromContainer<Container<Head, Args...>>
    {
        using type = Container<Args...>;
    };

    /**
     * @brief return the container without the first type
     * @tparam Container
     */
    template <typename Container>
    using PopFrontFromContainer_t = typename PopFrontFromContainer<Container>::type;
} // namespace metann

#endif // TYPE_TRAITS_HPP
