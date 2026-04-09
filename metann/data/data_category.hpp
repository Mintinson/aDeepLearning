//
// Created by asus on 2025/1/7.
//

#ifndef DATA_CATEGORY_HPP
#define DATA_CATEGORY_HPP

#include <type_traits>
#include "allocator.hpp"

namespace metann
{
    template <typename T>
    concept CategoryConcept = true;  // do nothing but telling you that template parameter is a CategoryTags
    struct CategoryTags
    {
        struct Scalar;
        struct Matrix;
        struct BatchScalar;
        struct BatchMatrix;
    };

    /// An auxiliary variable used to extend the custom type of the data,
    /// specializing to the category to give it a value of true when a class belongs to a category
    template <typename T>
    constexpr bool IsScalarHelper_v = false;
    template <typename T>
    constexpr bool IsScalar_v = IsScalarHelper_v<std::remove_cvref_t<T>>;

    template <typename T>
    struct IsScalar
    {
        static constexpr bool value = IsScalar_v<T>;
    };

    /// An auxiliary variable used to extend the custom type of the data,
    /// specializing to the category to give it a value of true when a class belongs to a category
    template <typename T>
    constexpr bool IsMatrixHelper_v = false;
    template <typename T>
    constexpr bool IsMatrix_v = IsMatrixHelper_v<std::remove_cvref_t<T>>;

    template <typename T>
    struct IsMatrix
    {
        static constexpr bool value = IsMatrix_v<T>;
    };

    /// An auxiliary variable used to extend the custom type of the data,
    /// specializing to the category to give it a value of true when a class belongs to a category
    template <typename T>
    constexpr bool IsBatchScalarHelper_v = false;
    template <typename T>
    constexpr bool IsBatchScalar_v = IsBatchScalarHelper_v<std::remove_cvref_t<T>>;

    template <typename T>
    struct IsBatchScalar
    {
        static constexpr bool value = IsBatchScalar_v<T>;
    };

    /// An auxiliary variable used to extend the custom type of the data,
    /// specializing to the category to give it a value of true when a class belongs to a category
    template <typename T>
    constexpr bool IsBatchMatrixHelper_v = false;
    template <typename T>
    constexpr bool IsBatchMatrix_v = IsBatchMatrixHelper_v<std::remove_cvref_t<T>>;

    template <typename T>
    struct IsBatchMatrix
    {
        static constexpr bool value = IsBatchMatrix_v<T>;
    };

    namespace details
    {
        template <bool isScalar, bool isMatrix, bool isBatchScalar, bool isBatchMatrix>
        struct DataCategoryHelper;

        template <>
        struct DataCategoryHelper<true, false, false, false>
        {
            using type = CategoryTags::Scalar;
        };

        template <>
        struct DataCategoryHelper<false, true, false, false>
        {
            using type = CategoryTags::Matrix;
        };

        template <>
        struct DataCategoryHelper<false, false, true, false>
        {
            using type = CategoryTags::BatchScalar;
        };

        template <>
        struct DataCategoryHelper<false, false, false, true>
        {
            using type = CategoryTags::BatchMatrix;
        };
    } // namespace details
    template <DataConcept T>
    struct DataCategory
    {
    private:
        template <bool isScalar, bool isMatrix, bool isBatchScalar, bool isBatchMatrix,
                  typename Dummy = void>
        struct helper;

        template <typename Dummy>
        struct helper<true, false, false, false, Dummy>
        {
            using type = CategoryTags::Scalar;
        };

        template <typename Dummy>
        struct helper<false, true, false, false, Dummy>
        {
            using type = CategoryTags::Matrix;
        };

        template <typename Dummy>
        struct helper<false, false, true, false, Dummy>
        {
            using type = CategoryTags::BatchScalar;
        };

        template <typename Dummy>
        struct helper<false, false, false, true, Dummy>
        {
            using type = CategoryTags::BatchMatrix;
        };

    public:
        // using type = typename helper<
        //     IsScalar_v<T>, IsMatrix_v<T>,
        //     IsBatchScalar_v<T>, IsBatchMatrix_v<T>>::type;
        using type = typename details::DataCategoryHelper<
            IsScalar_v<T>, IsMatrix_v<T>,
            IsBatchScalar_v<T>, IsBatchMatrix_v<T>>::type;
    };

    template <typename T>
    using DataCategory_t = typename DataCategory<T>::type;

    template <typename T>
    concept MatrixConcept = IsMatrix_v<std::remove_cvref_t<T>> || IsBatchMatrix_v<std::remove_cvref_t<T>>;
}

#endif //DATA_CATEGORY_HPP
