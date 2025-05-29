//
// Created by asus on 2025/1/12.
//

#ifndef TERNARY_OPERATORS_HPP
#define TERNARY_OPERATORS_HPP
#include "../data/data_category.hpp"
#include "operator_category.hpp"
#include "operator_helper.hpp"

namespace metann {
template <TernaryOperConcept OperTag, DataConcept DataType1,
    DataConcept DataType2, DataConcept DataType3>
class TernaryOperator;

template <TernaryOperConcept OperTag, DataConcept DataType1,
    DataConcept DataType2, DataConcept DataType3>
constexpr bool IsScalarHelper_v<TernaryOperator<OperTag, DataType1, DataType2, DataType3>>
    = std::is_same_v<OperaCateCal_t<OperTag, DataType1, DataType2, DataType3>, CategoryTags::Scalar>;
template <TernaryOperConcept OperTag, DataConcept DataType1,
    DataConcept DataType2, DataConcept DataType3>
constexpr bool IsMatrixHelper_v<TernaryOperator<OperTag, DataType1, DataType2, DataType3>>
    = std::is_same_v<OperaCateCal_t<OperTag, DataType1, DataType2, DataType3>, CategoryTags::Matrix>;
template <TernaryOperConcept OperTag, DataConcept DataType1,
    DataConcept DataType2, DataConcept DataType3>
constexpr bool IsBatchScalarHelper_v<TernaryOperator<OperTag, DataType1, DataType2, DataType3>>
    = std::is_same_v<OperaCateCal_t<OperTag, DataType1, DataType2, DataType3>, CategoryTags::BatchScalar>;
template <TernaryOperConcept OperTag, DataConcept DataType1,
    DataConcept DataType2, DataConcept DataType3>
constexpr bool IsBatchMatrixHelper_v<TernaryOperator<OperTag, DataType1, DataType2, DataType3>>
    = std::is_same_v<OperaCateCal_t<OperTag, DataType1, DataType2, DataType3>, CategoryTags::BatchMatrix>;

template <TernaryOperConcept OperTag, DataConcept DataType1,
    DataConcept DataType2, DataConcept DataType3>
class TernaryOperator
    : public OperOrganizer<OperTag, OperaCateCal_t<OperTag, DataType1, DataType2, DataType3>> {
public:
    using ElementType = OperElementType_t<OperTag, DataType1, DataType2, DataType3>;
    using DeviceType = OperDeviceType_t<OperTag, DataType1, DataType2, DataType3>;

    explicit TernaryOperator(DataType1 data1, DataType2 data2, DataType3 data3)
        : OperOrganizer<OperTag,
              OperaCateCal_t<OperTag, DataType1,
                  DataType2, DataType3>>(data1, data2, data3)
        , m_data1(std::move(data1))
        , m_data2(std::move(data2))
        , m_data3(std::move(data3))
    {
    }

    [[nodiscard]] const DataType1& operand1() const { return m_data1; }
    [[nodiscard]] const DataType1& operand2() const { return m_data2; }
    [[nodiscard]] const DataType3& operand3() const { return m_data3; }
    // TODO: for evaluation

private:
    DataType1 m_data1;
    DataType2 m_data2;
    DataType3 m_data3;
};

/*************************************** Interpolate operator ***************************************/
namespace details {
    template <DataConcept P1, DataConcept P2, DataConcept P3>
        requires(IsMatrix_v<P1> && IsMatrix_v<P2> && IsMatrix_v<P3>)
        || (IsBatchMatrix_v<P1> && IsBatchMatrix_v<P2> && IsBatchMatrix_v<P3>)
    class OperatorInterpolate {
    private:
        using RawM1 = std::remove_cvref_t<P1>;
        using RawM2 = std::remove_cvref_t<P2>;
        using RawM3 = std::remove_cvref_t<P3>;

    public:
        template <CategoryConcept T, CategoryConcept U, CategoryConcept V>
            requires std::is_same_v<T, U> && std::is_same_v<T, V>
        static auto eval(T&& m1, U&& m2, V&& m3)
        {
            static_assert(std::is_same_v<typename RawM1::DeviceType, typename RawM2::DeviceType>
                    && std::is_same_v<typename RawM1::DeviceType, typename RawM3::DeviceType>,
                "Matrices with different device types cannot interpolate directly");
            static_assert(std::is_same_v<typename RawM1::ElementType, typename RawM2::ElementType>
                    && std::is_same_v<typename RawM1::ElementType, typename RawM3::ElementType>,
                "Matrices with different element types cannot interpolate directly");
            using ResType = TernaryOperator<TernaryOperTags::Interpolate, RawM1, RawM2, RawM3>;
            return ResType(std::forward<T>(m1), std::forward<U>(m2), std::forward<V>(m3));
        }
    };
} // namespace details
template <DataConcept P1, DataConcept P2, DataConcept P3>
    requires(IsMatrix_v<P1> && IsMatrix_v<P2> && IsMatrix_v<P3>)
    || (IsBatchMatrix_v<P1> && IsBatchMatrix_v<P2> && IsBatchMatrix_v<P3>)
auto interpolate(P1&& m1, P2&& m2, P3&& m3)
{
    using Cate1 = DataCategory_t<P1>;
    using Cate2 = DataCategory_t<P2>;
    using Cate3 = DataCategory_t<P3>;

    return details::OperatorInterpolate<P1, P2, P3>::template eval<Cate1, Cate2, Cate3>(std::forward<P1>(m1), std::forward<P2>(m2));
}

/*************************** NegativeLogLikelihoodDerivative operator ***************************/
// change the default behavior
template <>
struct OperCategory<TernaryOperTags::NegativeLogLikelihoodDerivative,
    CategoryTags::Scalar,
    CategoryTags::Matrix,
    CategoryTags::Matrix> {
    using type = CategoryTags::Matrix;
};

template <>
struct OperCategory<TernaryOperTags::NegativeLogLikelihoodDerivative,
    CategoryTags::BatchScalar,
    CategoryTags::BatchMatrix,
    CategoryTags::BatchMatrix> {
    using type = CategoryTags::BatchMatrix;
};

template <>
class OperOrganizer<TernaryOperTags::NegativeLogLikelihoodDerivative, CategoryTags::Matrix> {
public:
    template <DataConcept TD1, DataConcept TD2, DataConcept TD3>
    OperOrganizer(const TD1& data1, const TD2& data2, const TD3& data3)
        : m_rowNum(data2.rowNum())
        , m_colNum(data2.colNum())
    {
        assert(data2.rowNum() == data3.rowNum());
        assert(data2.colNum() == data3.colNum());
    }

    [[nodiscard]] std::size_t rowNum() const { return m_rowNum; }
    [[nodiscard]] std::size_t colNum() const { return m_colNum; }

private:
    std::size_t m_rowNum;
    std::size_t m_colNum;
};

template <>
class OperOrganizer<TernaryOperTags::NegativeLogLikelihoodDerivative, CategoryTags::BatchMatrix> {
public:
    template <DataConcept TD1, DataConcept TD2, DataConcept TD3>
    OperOrganizer(const TD1& data1, const TD2& data2, const TD3& data3)
        : m_rowNum(data2.rowNum())
        , m_colNum(data2.colNum())
        , m_batchNum(data2.batchNum())
    {
        assert(data2.rowNum() == data3.rowNum());
        assert(data2.colNum() == data3.colNum());
        assert(data2.batchNum() == data3.batchNum());
    }

    [[nodiscard]] std::size_t rowNum() const { return m_rowNum; }
    [[nodiscard]] std::size_t colNum() const { return m_colNum; }
    [[nodiscard]] std::size_t batchNum() const { return m_batchNum; }

private:
    std::size_t m_rowNum;
    std::size_t m_colNum;
    std::size_t m_batchNum;
};

template <DataConcept Op1, DataConcept Op2, DataConcept Op3>
struct OperElementType<TernaryOperTags::NegativeLogLikelihoodDerivative,
    Op1, Op2, Op3> {
    using type = typename Op2::ElementType;
};

template <DataConcept Op1, DataConcept Op2, DataConcept Op3>
struct OperDeviceType<TernaryOperTags::NegativeLogLikelihoodDerivative,
    Op1, Op2, Op3> {
    using type = typename Op2::DeviceType;
};

namespace details {
    template <DataConcept P1, DataConcept P2, DataConcept P3>
        requires(IsScalar_v<P1> && IsMatrix_v<P2> && IsMatrix_v<P3>)
        || (IsBatchScalar_v<P1> && IsBatchMatrix_v<P2> && IsBatchMatrix_v<P3>)
    class OperatorNegativeLogLikelihoodDerivative {
    private:
        using RawM1 = std::remove_cvref_t<P1>;
        using RawM2 = std::remove_cvref_t<P2>;
        using RawM3 = std::remove_cvref_t<P3>;

    public:
        template <CategoryConcept T, CategoryConcept U, CategoryConcept V>
            requires std::is_same_v<T, U> && std::is_same_v<T, V>
        static auto eval(T&& m1, U&& m2, V&& m3)
        {
            static_assert(
                std::is_same_v<typename RawM2::DeviceType, typename RawM3::DeviceType>,
                "Matrices with different device types cannot NegativeLogLikelihoodDerivative directly");
            static_assert(
                std::is_same_v<typename RawM2::ElementType, typename RawM3::ElementType>,
                "Matrices with different element types cannot NegativeLogLikelihoodDerivative directly");
            using ResType = TernaryOperator<TernaryOperTags::NegativeLogLikelihoodDerivative, RawM1, RawM2, RawM3>;
            return ResType(std::forward<T>(m1), std::forward<U>(m2), std::forward<V>(m3));
        }
    };
} // namespace details
template <DataConcept P1, DataConcept P2, DataConcept P3>
    requires(IsScalar_v<P1> && IsMatrix_v<P2> && IsMatrix_v<P3>)
    || (IsBatchScalar_v<P1> && IsBatchMatrix_v<P2> && IsBatchMatrix_v<P3>)
auto neg_log_likelihood_derivative(P1&& m1, P2&& m2, P3&& m3)
{
    using Cate1 = DataCategory_t<P1>;
    using Cate2 = DataCategory_t<P2>;
    using Cate3 = DataCategory_t<P3>;

    return details::OperatorNegativeLogLikelihoodDerivative<P1, P2, P3>::template eval<Cate1, Cate2, Cate3>(std::forward<P1>(m1), std::forward<P2>(m2));
}
} // namespace metann

#endif // TERNARY_OPERATORS_H
