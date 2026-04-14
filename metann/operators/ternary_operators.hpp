//
// Created by asus on 2025/1/12.
//

#ifndef TERNARY_OPERATORS_HPP
#define TERNARY_OPERATORS_HPP
#include "../data/data_category.hpp"
#include "../utils/type_traits.hpp"
#include "operator_category.hpp"
#include "operator_helper.hpp"

namespace metann {
template <TernaryOperConcept OperTag, DataConcept DataType1, DataConcept DataType2, DataConcept DataType3>
class TernaryOperator;

template <TernaryOperConcept OperTag, DataConcept DataType1, DataConcept DataType2, DataConcept DataType3>
constexpr bool IsScalarHelper_v<TernaryOperator<OperTag, DataType1, DataType2, DataType3>> =
    std::is_same_v<OperaCateCal_t<OperTag, DataType1, DataType2, DataType3>, CategoryTags::Scalar>;
template <TernaryOperConcept OperTag, DataConcept DataType1, DataConcept DataType2, DataConcept DataType3>
constexpr bool IsMatrixHelper_v<TernaryOperator<OperTag, DataType1, DataType2, DataType3>> =
    std::is_same_v<OperaCateCal_t<OperTag, DataType1, DataType2, DataType3>, CategoryTags::Matrix>;
template <TernaryOperConcept OperTag, DataConcept DataType1, DataConcept DataType2, DataConcept DataType3>
constexpr bool IsBatchScalarHelper_v<TernaryOperator<OperTag, DataType1, DataType2, DataType3>> =
    std::is_same_v<OperaCateCal_t<OperTag, DataType1, DataType2, DataType3>, CategoryTags::BatchScalar>;
template <TernaryOperConcept OperTag, DataConcept DataType1, DataConcept DataType2, DataConcept DataType3>
constexpr bool IsBatchMatrixHelper_v<TernaryOperator<OperTag, DataType1, DataType2, DataType3>> =
    std::is_same_v<OperaCateCal_t<OperTag, DataType1, DataType2, DataType3>, CategoryTags::BatchMatrix>;

template <TernaryOperConcept OperTag, DataConcept DataType1, DataConcept DataType2, DataConcept DataType3>
class TernaryOperator : public OperOrganizer<OperTag, OperaCateCal_t<OperTag, DataType1, DataType2, DataType3>> {
public:
    using ElementType = OperElementType_t<OperTag, DataType1, DataType2, DataType3>;
    using DeviceType = OperDeviceType_t<OperTag, DataType1, DataType2, DataType3>;

    explicit TernaryOperator(DataType1 data1, DataType2 data2, DataType3 data3)
        : OperOrganizer<OperTag, OperaCateCal_t<OperTag, DataType1, DataType2, DataType3>>(data1, data2, data3)
        , m_data1(std::move(data1))
        , m_data2(std::move(data2))
        , m_data3(std::move(data3)) {}

    [[nodiscard]] const DataType1& operand1() const { return m_data1; }

    [[nodiscard]] const DataType1& operand2() const { return m_data2; }

    [[nodiscard]] const DataType3& operand3() const { return m_data3; }

    auto evalRegister() const {
        if (!m_evalBuf.isEvaluated()) {
            using OperSeqContainer = typename OperSeq<OperTag>::type;

            using HeadType = ContainerHead_t<OperSeqContainer>;
            using TailType = PopFrontFromContainer_t<OperSeqContainer>;
            HeadType::template evalRegister<TailType>(m_evalBuf, m_data1, m_data2, m_data3);
        }
        return m_evalBuf.constHandle();
    }

    bool operator==(const TernaryOperator& rhs) const {
        return (m_data1 == rhs.m_data1) && (m_data2 == rhs.m_data2) && (m_data3 == rhs.m_data3);
    }

    template <typename OtherDataType>
    bool operator==(const OtherDataType& val) const {
        return false;
    }

    template <typename OtherDataType>
    bool operator!=(const OtherDataType& val) const {
        return !(operator==(val));
    }

private:
    DataType1 m_data1;
    DataType2 m_data2;
    DataType3 m_data3;

    using Cate = OperaCateCal_t<OperTag, DataType1, DataType2, DataType3>;
    using Principal = PrincipleDataType_t<Cate, ElementType, DeviceType>;
    EvalBuffer<Principal> m_evalBuf;
};

/*************************************** Interpolate operator ***************************************/
namespace details {
template <DataConcept P1, DataConcept P2, DataConcept P3>
    requires(IsMatrix_v<P1> && IsMatrix_v<P2> && IsMatrix_v<P3>) ||
            (IsBatchMatrix_v<P1> && IsBatchMatrix_v<P2> && IsBatchMatrix_v<P3>)
class OperatorInterpolate {
private:
    using RawM1 = std::remove_cvref_t<P1>;
    using RawM2 = std::remove_cvref_t<P2>;
    using RawM3 = std::remove_cvref_t<P3>;

public:
    template <CategoryConcept T, CategoryConcept U, CategoryConcept V>
        requires std::is_same_v<T, U> && std::is_same_v<T, V>
    static auto eval(T&& m1, U&& m2, V&& m3) {
        static_assert(std::is_same_v<typename RawM1::DeviceType, typename RawM2::DeviceType> &&
                          std::is_same_v<typename RawM1::DeviceType, typename RawM3::DeviceType>,
                      "Matrices with different device types cannot interpolate directly");
        static_assert(std::is_same_v<typename RawM1::ElementType, typename RawM2::ElementType> &&
                          std::is_same_v<typename RawM1::ElementType, typename RawM3::ElementType>,
                      "Matrices with different element types cannot interpolate directly");
        using ResType = TernaryOperator<TernaryOperTags::Interpolate, RawM1, RawM2, RawM3>;
        return ResType(std::forward<T>(m1), std::forward<U>(m2), std::forward<V>(m3));
    }
};
}  // namespace details

template <DataConcept P1, DataConcept P2, DataConcept P3>
    requires(IsMatrix_v<P1> && IsMatrix_v<P2> && IsMatrix_v<P3>) ||
            (IsBatchMatrix_v<P1> && IsBatchMatrix_v<P2> && IsBatchMatrix_v<P3>)
auto interpolate(P1&& m1, P2&& m2, P3&& m3) {
    using Cate1 = DataCategory_t<P1>;
    using Cate2 = DataCategory_t<P2>;
    using Cate3 = DataCategory_t<P3>;

    return details::OperatorInterpolate<P1, P2, P3>::template eval<Cate1, Cate2, Cate3>(std::forward<P1>(m1),
                                                                                        std::forward<P2>(m2));
}

/*************************** NegativeLogLikelihoodDerivative operator ***************************/

// template<>
// struct OperSeq<TernaryOperTags::Interpolate> {
//     using type = OperSeqContainer<eval::BinaryCalculator<eval::TrivialBinaryOperatorTag<std::plus<> > > >;
// };
//
// template<>
// struct OperSeq<TernaryOperTags::NegativeLogLikelihoodDerivative> {
//     using type = OperSeqContainer<eval::BinaryCalculator<eval::TrivialBinaryOperatorTag<std::plus<> > > >;
// };

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
        , m_colNum(data2.colNum()) {
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
        , m_batchNum(data2.batchNum()) {
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
struct OperElementType<TernaryOperTags::NegativeLogLikelihoodDerivative, Op1, Op2, Op3> {
    using type = typename Op2::ElementType;
};

template <DataConcept Op1, DataConcept Op2, DataConcept Op3>
struct OperDeviceType<TernaryOperTags::NegativeLogLikelihoodDerivative, Op1, Op2, Op3> {
    using type = typename Op2::DeviceType;
};

namespace details {
template <DataConcept P1, DataConcept P2, DataConcept P3>
    requires(IsScalar_v<P1> && IsMatrix_v<P2> && IsMatrix_v<P3>) ||
            (IsBatchScalar_v<P1> && IsBatchMatrix_v<P2> && IsBatchMatrix_v<P3>)
class OperatorNegativeLogLikelihoodDerivative {
private:
    using RawM1 = std::remove_cvref_t<P1>;
    using RawM2 = std::remove_cvref_t<P2>;
    using RawM3 = std::remove_cvref_t<P3>;

public:
    // template<CategoryConcept T, CategoryConcept U, CategoryConcept V>
    // requires std::is_same_v<T, U> && std::is_same_v<T, V>
    static auto eval(P1&& m1, P2&& m2, P3&& m3) {
        static_assert(std::is_same_v<typename RawM2::DeviceType, typename RawM3::DeviceType>,
                      "Matrices with different device types cannot NegativeLogLikelihoodDerivative directly");
        static_assert(std::is_same_v<typename RawM2::ElementType, typename RawM3::ElementType>,
                      "Matrices with different element types cannot NegativeLogLikelihoodDerivative directly");
        using ResType = TernaryOperator<TernaryOperTags::NegativeLogLikelihoodDerivative, RawM1, RawM2, RawM3>;
        return ResType(std::forward<P1>(m1), std::forward<P2>(m2), std::forward<P3>(m3));
    }
};
}  // namespace details

template <DataConcept P1, DataConcept P2, DataConcept P3>
    requires(IsScalar_v<P1> && IsMatrix_v<P2> && IsMatrix_v<P3>) ||
            (IsBatchScalar_v<P1> && IsBatchMatrix_v<P2> && IsBatchMatrix_v<P3>)
auto neg_log_likelihood_derivative(P1&& m1, P2&& m2, P3&& m3) {
    using Cate1 = DataCategory_t<P1>;
    using Cate2 = DataCategory_t<P2>;
    using Cate3 = DataCategory_t<P3>;

    // return details::OperatorNegativeLogLikelihoodDerivative<P1, P2, P3>::template eval<Cate1, Cate2, Cate3>(
    //     std::forward<P1>(m1), std::forward<P2>(m2), std::forward<P3>(m3));
    return details::OperatorNegativeLogLikelihoodDerivative<P1, P2, P3>::eval(
        std::forward<P1>(m1), std::forward<P2>(m2), std::forward<P3>(m3));
}

namespace eval {
template <>
struct UnitWrapper<TernaryOperTags::Interpolate> {
    template <typename OperHandle1,
              typename OperHandle2,
              typename OperHandle3,
              typename TElem,
              DeviceConcept Device,
              CategoryConcept Category>
    class EvalUnit;

    template <typename OperHandle1, typename OperHandle2, typename OperHandle3, typename Element>
    class EvalUnit<OperHandle1, OperHandle2, OperHandle3, Element, CPU, CategoryTags::Matrix>
        : public BaseEvalUnit<CPU> {
    public:
        using ElementType = Element;
        using DeviceType = CPU;

        EvalUnit(OperHandle1 oper1,
                 OperHandle2 oper2,
                 OperHandle3 oper3,
                 EvalHandle<Matrix<ElementType, DeviceType>> evalOutput)
            : m_oper1(std::move(oper1))
            , m_oper2(std::move(oper2))
            , m_oper3(std::move(oper3))
            , m_evalOutput(std::move(evalOutput)) {}

        void eval() override {
            const auto& pV1 = m_oper1.data();
            const auto& pV2 = m_oper2.data();
            const auto& pV3 = m_oper3.data();
            const size_t rowNum = pV1.rowNum();
            const size_t colNum = pV1.colNum();
            assert(pV2.rowNum() == rowNum);
            assert(pV2.colNum() == colNum);
            assert(pV3.rowNum() == rowNum);
            assert(pV3.colNum() == colNum);

            m_evalOutput.allocate(rowNum, colNum);
            auto& res = m_evalOutput.mutableData();

            auto mem_v1 = lower_access(pV1);
            auto mem_v2 = lower_access(pV2);
            auto mem_v3 = lower_access(pV3);
            auto mem_res = lower_access(res);

            const size_t src1PackNum = mem_v1.rowLen();
            const size_t src2PackNum = mem_v2.rowLen();
            const size_t src3PackNum = mem_v3.rowLen();
            const size_t tgtPackNum = mem_res.rowLen();

            const ElementType* r1 = mem_v1.rawMemory();
            const ElementType* r2 = mem_v2.rawMemory();
            const ElementType* r3 = mem_v3.rawMemory();
            ElementType* r = mem_res.mutableRawMemory();

            for (size_t i = 0; i < rowNum; ++i) {
                for (size_t j = 0; j < colNum; ++j) {
                    r[j] = r1[j] * r3[j] + r2[j] * (1 - r3[j]);
                }
                r1 += src1PackNum;
                r2 += src2PackNum;
                r3 += src3PackNum;
                r += tgtPackNum;
            }
            m_evalOutput.setEval();
        }

    private:
        OperHandle1 m_oper1;
        OperHandle2 m_oper2;
        OperHandle3 m_oper3;

        EvalHandle<Matrix<ElementType, DeviceType>> m_evalOutput;
    };

    template <typename OperHandle1, typename OperHandle2, typename OperHandle3, typename Element>
    class EvalUnit<OperHandle1, OperHandle2, OperHandle3, Element, CPU, CategoryTags::BatchMatrix>
        : public BaseEvalUnit<CPU> {
    public:
        using ElementType = Element;
        using DeviceType = CPU;

        EvalUnit(OperHandle1 oper1,
                 OperHandle2 oper2,
                 OperHandle3 oper3,
                 EvalHandle<Batch<ElementType, DeviceType, CategoryTags::Matrix>> evalOutput)
            : m_oper1(std::move(oper1))
            , m_oper2(std::move(oper2))
            , m_oper3(std::move(oper3))
            , m_evalOutput(std::move(evalOutput)) {}

        void eval() override {
            const auto& pV1 = m_oper1.data();
            const auto& pV2 = m_oper2.data();
            const auto& pV3 = m_oper3.data();
            const size_t rowNum = pV1.rowNum();
            const size_t colNum = pV1.colNum();
            const size_t batchNum = pV1.batchNum();

            assert(pV2.rowNum() == rowNum);
            assert(pV2.colNum() == colNum);
            assert(pV2.batchNum() == batchNum);
            assert(pV3.batchNum() == batchNum);
            assert(pV3.rowNum() == rowNum);
            assert(pV3.colNum() == colNum);

            m_evalOutput.allocate(batchNum, rowNum, colNum);
            auto& res = m_evalOutput.mutableData();

            for (std::size_t curBatch = 0; curBatch < batchNum; ++curBatch) {
                auto mem_v1 = lower_access(pV1[curBatch]);
                auto mem_v2 = lower_access(pV1[curBatch]);
                auto mem_v3 = lower_access(pV1[curBatch]);
                auto mem_res = lower_access(res[curBatch]);

                const size_t src1PackNum = mem_v1.rowLen();
                const size_t src2PackNum = mem_v2.rowLen();
                const size_t src3PackNum = mem_v3.rowLen();
                const size_t tgtPackNum = mem_res.rowLen();

                const ElementType* r1 = mem_v1.rawMemory();
                const ElementType* r2 = mem_v2.rawMemory();
                const ElementType* r3 = mem_v3.rawMemory();
                ElementType* r = mem_res.mutableRawMemory();

                for (size_t i = 0; i < rowNum; ++i) {
                    for (size_t j = 0; j < colNum; ++j) {
                        r[j] = r1[j] * r3[j] + r2[j] * (1 - r3[j]);
                    }
                    r1 += src1PackNum;
                    r2 += src2PackNum;
                    r3 += src3PackNum;
                    r += tgtPackNum;
                }
            }

            m_evalOutput.setEval();
        }

    private:
        OperHandle1 m_oper1;
        OperHandle2 m_oper2;
        OperHandle3 m_oper3;

        EvalHandle<Batch<ElementType, DeviceType, CategoryTags::Matrix>> m_evalOutput;
    };
};

template <>
struct UnitWrapper<TernaryOperTags::NegativeLogLikelihoodDerivative> {
    template <typename OperHandle1,
              typename OperHandle2,
              typename OperHandle3,
              typename TElem,
              DeviceConcept Device,
              CategoryConcept Category>
    class EvalUnit;

    template <typename OperHandle1, typename OperHandle2, typename OperHandle3, typename Element>
    class EvalUnit<OperHandle1, OperHandle2, OperHandle3, Element, CPU, CategoryTags::Matrix>
        : public BaseEvalUnit<CPU> {
    public:
        using ElementType = Element;
        using DeviceType = CPU;

        EvalUnit(OperHandle1 grad,
                 OperHandle2 operTgt,
                 OperHandle3 operPred,
                 EvalHandle<Matrix<ElementType, DeviceType>> evalOutput)
            : m_grad(std::move(grad))
            , m_operTgt(std::move(operTgt))
            , m_operPred(std::move(operPred))
            , m_evalOutput(std::move(evalOutput)) {}

        void eval() override {
            const auto& pGrad = m_grad.data();
            const auto& pTgt = m_operTgt.data();
            const auto& pPred = m_operPred.data();

            const size_t rowNum = pTgt.rowNum();
            const size_t colNum = pTgt.colNum();
            assert(pPred.rowNum() == rowNum);
            assert(pPred.colNum() == colNum);
            // assert(p_v3.rowNum() == rowNum);
            // assert(p_v3.colNum() == colNum);

            m_evalOutput.allocate(rowNum, colNum);
            auto& res = m_evalOutput.mutableData();

            auto memTgt = lower_access(pTgt);
            auto memPred = lower_access(pPred);
            // auto mem_v3 = lower_access(pV3);
            auto memRes = lower_access(res);

            const size_t src1PackNum = memTgt.rowLen();
            const size_t src2PackNum = memPred.rowLen();
            // const size_t src3PackNum = mem_v3.rowLen();
            const size_t resPackNum = memRes.rowLen();

            const ElementType* r1 = memTgt.rawMemory();
            const ElementType* r2 = memPred.rawMemory();
            // const ElementType *r3 = mem_v3.rawMemory();
            ElementType* r = memRes.mutableRawMemory();

            for (size_t i = 0; i < rowNum; ++i) {
                for (size_t j = 0; j < colNum; ++j) {
                    r[j] = pGrad.value() * (-r1[j] / r2[j]);
                }
                r1 += src1PackNum;
                r2 += src2PackNum;
                // r3 += src3PackNum;
                r += resPackNum;
            }
            m_evalOutput.setEval();
        }

    private:
        OperHandle1 m_grad;
        OperHandle2 m_operTgt;
        OperHandle3 m_operPred;

        EvalHandle<Matrix<ElementType, DeviceType>> m_evalOutput;
    };

    template <typename OperHandle1, typename OperHandle2, typename OperHandle3, typename Element>
    class EvalUnit<OperHandle1, OperHandle2, OperHandle3, Element, CPU, CategoryTags::BatchMatrix>
        : public BaseEvalUnit<CPU> {
    public:
        using ElementType = Element;
        using DeviceType = CPU;

        EvalUnit(OperHandle1 grad,
                 OperHandle2 operTgt,
                 OperHandle3 operPred,
                 EvalHandle<Batch<ElementType, DeviceType, CategoryTags::Matrix>> evalOutput)
            : m_grad(std::move(grad))
            , m_operTgt(std::move(operTgt))
            , m_operPred(std::move(operPred))
            , m_evalOutput(std::move(evalOutput)) {}

        void eval() override {
            const auto& pGrad = m_grad.data();
            const auto& pTgt = m_operTgt.data();
            const auto& pPred = m_operPred.data();

            const size_t rowNum = pTgt.rowNum();
            const size_t colNum = pTgt.colNum();
            const size_t batchNum = pTgt.batchNum();
            assert(pPred.rowNum() == rowNum);
            assert(pPred.colNum() == colNum);
            assert(pPred.batchNum() == batchNum);
            assert(pGrad.batchNum() == batchNum);
            // assert(p_v3.rowNum() == rowNum);
            // assert(p_v3.colNum() == colNum);

            m_evalOutput.allocate(batchNum, rowNum, colNum);
            auto& res = m_evalOutput.mutableData();

            for (size_t curBatch = 0; curBatch < batchNum; ++curBatch) {
                auto memTgt = lower_access(pTgt[curBatch]);
                auto memPred = lower_access(pPred[curBatch]);
                // auto mem_v3 = lower_access(pV3);
                auto memRes = lower_access(res[curBatch]);

                const size_t src1PackNum = memTgt.rowLen();
                const size_t src2PackNum = memPred.rowLen();
                // const size_t src3PackNum = mem_v3.rowLen();
                const size_t resPackNum = memRes.rowLen();

                const ElementType* r1 = memTgt.rawMemory();
                const ElementType* r2 = memPred.rawMemory();
                // const ElementType *r3 = mem_v3.rawMemory();
                ElementType* r = memRes.mutableRawMemory();

                for (size_t i = 0; i < rowNum; ++i) {
                    for (size_t j = 0; j < colNum; ++j) {
                        r[j] = pGrad[curBatch] * (-r1[j] / r2[j]);
                    }
                    r1 += src1PackNum;
                    r2 += src2PackNum;
                    // r3 += src3PackNum;
                    r += resPackNum;
                }
            }

            m_evalOutput.setEval();
        }

    private:
        OperHandle1 m_grad;
        OperHandle2 m_operTgt;
        OperHandle3 m_operPred;

        EvalHandle<Batch<ElementType, DeviceType, CategoryTags::Matrix>> m_evalOutput;
    };
};

template <TernaryOperConcept OperatorEvalTag>
struct TernaryCalculator {
    template <typename CaseTail, typename EvalRes, typename Operator1, typename Operator2, typename Operator3>
    static void evalRegister(EvalRes& evalRes, const Operator1& oper1, const Operator2& oper2, const Operator3& oper3) {
        static_assert(std::is_same_v<CaseTail, OperSeqContainer<>>, "General Case is not the last one");

        using ElementType = typename EvalRes::DataType::ElementType;
        using DeviceType = typename EvalRes::DataType::DeviceType;
        using CategoryType = DataCategory_t<typename EvalRes::DataType>;

        auto handle1 = oper1.evalRegister();
        auto handle2 = oper2.evalRegister();
        auto handle3 = oper3.evalRegister();
        using UnitType = typename UnitWrapper<OperatorEvalTag>::template EvalUnit<
            decltype(handle1), decltype(handle2), decltype(handle3), ElementType, DeviceType, CategoryType>;
        using GroupType = TrivialEvalGroup<UnitType>;

        auto outHandle = evalRes.handle();
        const void* dataPtr = outHandle.dataPtr();
        auto depVec = {handle1.dataPtr(), handle2.dataPtr(), handle3.dataPtr()};

        UnitType unit(std::move(handle1), std::move(handle2), std::move(handle3), std::move(outHandle));
        EvalPlan<DeviceType>::template registerFun<GroupType>(std::move(unit), dataPtr, std::move(depVec));
    }
};
}  // namespace eval

template <>
struct OperSeq<TernaryOperTags::Interpolate> {
    using type = OperSeqContainer<eval::TernaryCalculator<TernaryOperTags::Interpolate>>;
};

template <>
struct OperSeq<TernaryOperTags::NegativeLogLikelihoodDerivative> {
    using type = OperSeqContainer<eval::TernaryCalculator<TernaryOperTags::NegativeLogLikelihoodDerivative>>;
};
}  // namespace metann

#endif  // TERNARY_OPERATORS_HPP
