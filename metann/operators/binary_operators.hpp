//
// Created by asus on 2025/1/11.
//

#ifndef BINARY_OPERATORS_HPP
#define BINARY_OPERATORS_HPP

#include "../data/data_category.hpp"
#include "../data/duplicate.hpp"
#include "../data/matrix.hpp"
#include "../utils/type_traits.hpp"
#include "operator_category.hpp"
#include "operator_helper.hpp"
#include <type_traits>
#include <cmath>
#include <functional>

namespace metann
{
    template <BinaryOperConcept OperTag, DataConcept DataType1, DataConcept DataType2>
    class BinaryOperator;

    template <BinaryOperConcept OperTag, DataConcept DataType1, DataConcept DataType2>
    constexpr bool IsScalarHelper_v<BinaryOperator<OperTag, DataType1, DataType2>>
        = std::is_same_v<OperaCateCal_t<OperTag, DataType1, DataType2>, CategoryTags::Scalar>;
    template <BinaryOperConcept OperTag, DataConcept DataType1, DataConcept DataType2>
    constexpr bool IsMatrixHelper_v<BinaryOperator<OperTag, DataType1, DataType2>>
        = std::is_same_v<OperaCateCal_t<OperTag, DataType1, DataType2>, CategoryTags::Matrix>;
    template <BinaryOperConcept OperTag, DataConcept DataType1, DataConcept DataType2>
    constexpr bool IsBatchScalarHelper_v<BinaryOperator<OperTag, DataType1, DataType2>>
        = std::is_same_v<OperaCateCal_t<OperTag, DataType1, DataType2>, CategoryTags::BatchScalar>;
    template <BinaryOperConcept OperTag, DataConcept DataType1, DataConcept DataType2>
    constexpr bool IsBatchMatrixHelper_v<BinaryOperator<OperTag, DataType1, DataType2>>
        = std::is_same_v<OperaCateCal_t<OperTag, DataType1, DataType2>, CategoryTags::BatchMatrix>;

    template <BinaryOperConcept OperTag, DataConcept DataType1, DataConcept DataType2>
    class BinaryOperator : public OperOrganizer<OperTag, OperaCateCal_t<OperTag, DataType1, DataType2>>
    {
    public:
        using ElementType = OperElementType_t<OperTag, DataType1, DataType2>;
        using DeviceType = OperDeviceType_t<OperTag, DataType1, DataType2>;
        using Cate = OperaCateCal_t<OperTag, DataType1, DataType2>;

        explicit BinaryOperator(DataType1 data1, DataType2 data2)
            : OperOrganizer<OperTag, OperaCateCal_t<OperTag, DataType1, DataType2>>(data1, data2)
              , m_data1(std::move(data1))
              , m_data2(std::move(data2))
        {
        }

        [[nodiscard]] const DataType1& operand1() const { return m_data1; }
        [[nodiscard]] const DataType1& operand2() const { return m_data2; }


        // TODO: for evaluation
        auto evalRegister() const
        {
            if (!m_evalBuf.isEvaluated())
            {
                using OperSeqCont = typename OperSeq<OperTag>::type;
                using Head = ContainerHead_t<OperSeqCont>;
                using Tail = PopFrontFromContainer_t<OperSeqCont>;
                Head::template evalRegister<Tail>(m_evalBuf, m_data1, m_data2);
            }
            return m_evalBuf.constHandle();
        }

        bool operator==(const BinaryOperator& val) const
        {
            return (m_data1 == val.m_data1) && (m_data2 == val.m_data2);
        }

        template <typename OtherData>
        bool operator==(const OtherData& val) const
        {
            return false;
        }

        template <typename OtherData>
        bool operator!=(const OtherData& val) const
        {
            return !(operator==(val));
        }

    private:
        DataType1 m_data1;
        DataType2 m_data2;
        using PrincipalType = PrincipleDataType_t<Cate, ElementType, DeviceType>;
        EvalBuffer<PrincipalType> m_evalBuf;
    };

    /*************************************** + - * / operator ***************************************/
    template <typename T, typename U, template <typename> class Pred1, template <typename> class Pred2>
    constexpr bool Commutative_v = (Pred1<T>::value && Pred2<U>::value) || (Pred1<U>::value && Pred2<T>::value);
    // T and U can be const or reference
    template <typename T, typename U>
    concept ElemOperrable = (IsMatrix_v<T> && IsMatrix_v<U>)
        || Commutative_v<T, U, IsMatrix, IsScalar>
        || Commutative_v<T, U, IsMatrix, IsBatchMatrix>
        || Commutative_v<T, U, IsScalar, IsBatchMatrix>
        || (IsBatchMatrix_v<T> && IsBatchMatrix_v<U>);

    namespace details
    {
        template <BinaryOperConcept OperTag, typename P1, typename P2>
            requires ElemOperrable<P1, P2>
        struct OperatorTrivial // +, -, *, /
        {
        private:
            using RawM1 = std::remove_cvref_t<P1>;
            using RawM2 = std::remove_cvref_t<P2>;

        public:
            template <CategoryConcept T, CategoryConcept U>
                requires std::is_same_v<T, U>
            static auto eval(P1&& m1, P2&& m2)
            {
                static_assert(std::is_same_v<typename RawM1::DeviceType, typename RawM2::DeviceType>,
                              "Matrices with different device types cannot be operated directly");
                static_assert(std::is_same_v<typename RawM1::ElementType, typename RawM2::ElementType>,
                              "Matrices with different element types cannot be operated directly");
                using ResType = BinaryOperator<OperTag, RawM1, RawM2>;
                return ResType(std::forward<P1>(m1), std::forward<P2>(m2));
            }

            template <CategoryConcept T, CategoryConcept U>
                requires std::is_same_v<T, CategoryTags::Matrix> && std::is_same_v<U, CategoryTags::Scalar>
            static auto eval(P1&& m, P2&& s)
            {
                using ElementType = typename RawM1::ElementType;
                using DeviceType = typename RawM1::DeviceType;

                auto tempMatrix = make_trivial_matrix<ElementType, DeviceType>(m.rowNum(), m.colNum(),
                                                                               std::forward<P2>(s));
                using ResType = BinaryOperator<OperTag, RawM1, std::remove_cvref_t<decltype(tempMatrix)>>;
                return ResType(std::forward<P1>(m), std::move(tempMatrix));
            }

            template <CategoryConcept T, CategoryConcept U>
                requires std::is_same_v<T, CategoryTags::Scalar> && std::is_same_v<U, CategoryTags::Matrix>
            static auto eval(P1&& s, P2&& m)
            {
                if constexpr (IsInPack_v<OperTag, BinaryOperTags::Sub, BinaryOperTags::Div>)
                {
                    using ElementType = typename RawM2::ElementType;
                    using DeviceType = typename RawM2::DeviceType;

                    auto tempMatrix = make_trivial_matrix<ElementType, DeviceType>(m.rowNum(), m.colNum(),
                        std::forward<P1>(s));
                    using ResType = BinaryOperator<OperTag, std::remove_cvref_t<decltype(tempMatrix)>, RawM2>;
                    return ResType(std::move(tempMatrix), std::forward<P2>(m));
                }
                else
                {
                    return OperatorTrivial<OperTag, P2, P1>::template eval<U, T>(std::forward<P2>(m),
                        std::forward<P1>(s));
                }
            }

            template <CategoryConcept T, CategoryConcept U>
                requires std::is_same_v<T, CategoryTags::BatchMatrix> && std::is_same_v<U, CategoryTags::Matrix>
            static auto eval(P1&& m1, P2&& m2)
            {
                static_assert(std::is_same_v<typename RawM1::DeviceType, typename RawM2::DeviceType>,
                              "Matrices with different device types cannot be operated directly");
                static_assert(std::is_same_v<typename RawM1::ElementType, typename RawM2::ElementType>,
                              "Matrices with different element types cannot be operated directly");
                Duplicate<RawM2> tmp{std::forward<P2>(m2), m1.batchNum()};
                using ResType = BinaryOperator<OperTag, RawM1, Duplicate<RawM2>>;
                return ResType(std::forward<P1>(m1), std::move(tmp));
            }

            template <CategoryConcept T, CategoryConcept U>
                requires std::is_same_v<T, CategoryTags::Matrix> && std::is_same_v<U, CategoryTags::BatchMatrix>
            static auto eval(P1&& m1, P2&& m2)
            {
                if constexpr (IsInPack_v<OperTag, BinaryOperTags::Sub, BinaryOperTags::Div>)
                {
                    static_assert(std::is_same_v<typename RawM1::DeviceType, typename RawM2::DeviceType>,
                                  "Matrices with different device types cannot be operated directly");
                    static_assert(std::is_same_v<typename RawM1::ElementType, typename RawM2::ElementType>,
                                  "Matrices with different element types cannot be operated directly");
                    Duplicate<RawM1> tmp{std::forward<P1>(m1), m2.batchNum()};
                    using ResType = BinaryOperator<OperTag, Duplicate<RawM1>, RawM2>;
                    return ResType(std::move(tmp), std::forward<P2>(m2));
                }
                else
                {
                    return OperatorTrivial<OperTag, P2, P1>::template eval<U, T>(std::forward<P2>(m2),
                        std::forward<P1>(m1));
                }
            }

            template <CategoryConcept T, CategoryConcept U>
                requires std::is_same_v<T, CategoryTags::BatchMatrix> && std::is_same_v<U, CategoryTags::Scalar>
            static auto eval(P1&& m, P2&& s)
            {
                // static_assert(std::is_same_v<typename RawM1::DeviceType, typename RawM2::DeviceType>,
                //               "Matrices with different device types cannot be operated directly");
                // static_assert(std::is_same_v<typename RawM1::ElementType, typename RawM2::ElementType>,
                //               "Matrices with different element types cannot be operated directly");
                // Duplicate<RawM2> tmp{std::forward<U>(m2), m2.batchNum()};
                auto tmpTrivial = make_trivial_matrix<typename RawM1::ElementType, typename RawM1::DeviceType>(
                    m.rowNum(), m.colNum(), std::forward<P2>(s));
                auto duplicate = make_duplicate(m.batchNum(), std::move(tmpTrivial));
                using ResType = BinaryOperator<OperTag, RawM1, decltype(duplicate)>;
                return ResType(std::forward<P1>(m), std::move(duplicate));
            }

            template <CategoryConcept T, CategoryConcept U>
                requires std::is_same_v<T, CategoryTags::Scalar> && std::is_same_v<U, CategoryTags::BatchMatrix>
            static auto eval(P1&& s, P2&& m)
            {
                if constexpr (IsInPack_v<OperTag, BinaryOperTags::Sub, BinaryOperTags::Div>)
                {
                    // static_assert(std::is_same_v<typename RawM1::DeviceType, typename RawM2::DeviceType>,
                    //               "Matrices with different device types cannot be operated directly");
                    // static_assert(std::is_same_v<typename RawM1::ElementType, typename RawM2::ElementType>,
                    //               "Matrices with different element types cannot be operated directly");
                    // Duplicate<RawM2> tmp{std::forward<U>(m2), m2.batchNum()};
                    auto tmpTrivial = make_trivial_matrix<typename RawM2::ElementType, typename RawM2::DeviceType>(
                        m.rowNum(), m.colNum(), std::forward<P1>(s));
                    auto duplicate = make_duplicate(m.batchNum(), std::move(tmpTrivial));
                    using ResType = BinaryOperator<OperTag, decltype(duplicate), RawM2>;
                    return ResType(std::move(duplicate), std::forward<P2>(m));
                }
                else
                {
                    return OperatorTrivial<OperTag, P2, P1>::template eval<U, T>(std::forward<P2>(m),
                        std::forward<P1>(s));
                }
            }
        };
    } // namespace details
    namespace eval
    {
        template <typename Operator>
        struct TrivialBinaryOperatorTag
        {
            using Oper = Operator;
        };

        template <typename Operator>
        struct UnitWrapper<TrivialBinaryOperatorTag<Operator>>
        {
            template <typename OperHandle1, typename OperHandle2, typename Element,
                      DeviceConcept Device, CategoryConcept Category>
            class EvalUnit;

            template <typename OperHandle1, typename OperHandle2, typename Element>
            class EvalUnit<OperHandle1, OperHandle2, Element, CPU, CategoryTags::Matrix>
                : public BaseEvalUnit<CPU>
            {
            public:
                using ElementType = Element;
                using DeviceType = CPU;

                EvalUnit(OperHandle1 op1, OperHandle2 op2,
                         EvalHandle<Matrix<Element, CPU>> evalOutput):
                    m_oper1(std::move(op1)), m_oper2(std::move(op2)), m_evalOutput(std::move(evalOutput))
                {
                }

                void eval() override
                {
                    const auto& pV1 = m_oper1.data();
                    const auto& pV2 = m_oper2.data();
                    const std::size_t rowNum = pV1.rowNum();
                    const std::size_t colNum = pV1.colNum();
                    assert(pV2.rowNum() == rowNum);
                    assert(pV2.colNum() == colNum);

                    m_evalOutput.allocate(rowNum, colNum);
                    auto& res = m_evalOutput.mutableData();

                    const auto memV1 = lower_access(pV1);
                    const auto memV2 = lower_access(pV2);
                    auto memRes = lower_access(res);

                    const std::size_t src1PackNum = memV1.rowLen();
                    const std::size_t src2PackNum = memV2.rowLen();
                    const std::size_t tgtPackNum = memRes.rowLen();

                    const Element* r1 = memV1.rawMemory();
                    const Element* r2 = memV2.rawMemory();
                    Element* r = memRes.mutableRawMemory();
                    auto op = Operator{};
                    for (size_t i = 0; i < rowNum; ++i)
                    {
                        for (size_t j = 0; j < colNum; ++j)
                        {
                            r[j] = op(r1[j], r2[j]);
                        }
                        r1 += src1PackNum;
                        r2 += src2PackNum;
                        r += tgtPackNum;
                    }
                    m_evalOutput.setEval();
                }

            private:
                OperHandle1 m_oper1;
                OperHandle2 m_oper2;
                EvalHandle<Matrix<Element, CPU>> m_evalOutput;
            };

            template <typename OperHandle1, typename OperHandle2, typename Element>
            class EvalUnit<OperHandle1, OperHandle2, Element, CPU, CategoryTags::BatchMatrix>
                : public BaseEvalUnit<CPU>
            {
            public:
                using ElementType = Element;
                using DeviceType = CPU;

                EvalUnit(OperHandle1 op1, OperHandle2 op2,
                         EvalHandle<Batch<ElementType, DeviceType, CategoryTags::Matrix>> evalOutput):
                    m_oper1(std::move(op1)), m_oper2(std::move(op2)), m_evalOutput(std::move(evalOutput))
                {
                }

                void eval() override
                {
                    const auto& pV1 = m_oper1.data();
                    const auto& pV2 = m_oper2.data();
                    const std::size_t rowNum = pV1.rowNum();
                    const std::size_t colNum = pV1.colNum();
                    const std::size_t batchNum = pV1.batchNum();
                    assert(pV2.rowNum() == rowNum);
                    assert(pV2.colNum() == colNum);
                    assert(pV2.batchNum() == batchNum);

                    m_evalOutput.allocate(batchNum, rowNum, colNum);
                    auto& res = m_evalOutput.mutableData();
                    auto op = Operator{};
                    for (std::size_t batch = 0; batch < batchNum; ++batch)
                    {
                        const auto memV1 = lower_access(pV1[batch]);
                        const auto memV2 = lower_access(pV2[batch]);
                        auto memRes = lower_access(res[batch]);

                        const std::size_t src1PackNum = memV1.rowLen();
                        const std::size_t src2PackNum = memV2.rowLen();
                        const std::size_t tgtPackNum = memRes.rowLen();

                        const Element* r1 = memV1.rawMemory();
                        const Element* r2 = memV2.rawMemory();
                        Element* r = memRes.mutableRawMemory();
                        for (size_t i = 0; i < rowNum; ++i)
                        {
                            for (size_t j = 0; j < colNum; ++j)
                            {
                                r[j] = op(r1[j], r2[j]);
                            }
                            r1 += src1PackNum;
                            r2 += src2PackNum;
                            r += tgtPackNum;
                        }
                    }
                    m_evalOutput.setEval();
                }

            private:
                OperHandle1 m_oper1;
                OperHandle2 m_oper2;
                EvalHandle<Batch<ElementType, DeviceType, CategoryTags::Matrix>> m_evalOutput;
            };
        };

        template <BinaryOperConcept OperatorEvalTag>
        struct BinaryCalculator
        {
            template <typename CaseTail, typename EvalRes, typename Operator1, typename Operator2>
            static void evalRegister(EvalRes& evalRes, const Operator1& oper1, const Operator2& oper2)
            {
                static_assert(std::is_same_v<CaseTail, OperSeqContainer<>>,
                              "General Case is not the last one");

                using ElementType = typename EvalRes::DataType::ElementType;
                using DeviceType = typename EvalRes::DataType::DeviceType;
                using CategoryType = DataCategory_t<typename EvalRes::DataType>;

                auto handle1 = oper1.evalRegister();
                auto handle2 = oper2.evalRegister();
                using UnitType = typename UnitWrapper<OperatorEvalTag>::template EvalUnit<
                    decltype(handle1), decltype(handle2),
                    ElementType, DeviceType, CategoryType>;
                using GroupType = TrivialEvalGroup<UnitType>;

                auto outHandle = evalRes.handle();
                const void* dataPtr = outHandle.dataPtr();
                auto depVec = {handle1.dataPtr(), handle2.dataPtr()};

                UnitType unit(std::move(handle1), std::move(handle2), std::move(outHandle));
                EvalPlan<DeviceType>::template registerFun<GroupType>
                    (std::move(unit), dataPtr, std::move(depVec));
            }
        };
    }


    template <>
    struct OperSeq<BinaryOperTags::Add>
    {
        using type = OperSeqContainer<eval::BinaryCalculator<eval::TrivialBinaryOperatorTag<std::plus<>>>>;
    };

    template <>
    struct OperSeq<BinaryOperTags::Sub>
    {
        using type = OperSeqContainer<eval::BinaryCalculator<eval::TrivialBinaryOperatorTag<std::minus<>>>>;
    };

    template <>
    struct OperSeq<BinaryOperTags::Mul>
    {
        using type = OperSeqContainer<eval::BinaryCalculator<eval::TrivialBinaryOperatorTag<std::multiplies<>>>>;
    };

    template <>
    struct OperSeq<BinaryOperTags::Div>
    {
        using type = OperSeqContainer<eval::BinaryCalculator<eval::TrivialBinaryOperatorTag<std::divides<>>>>;
    };

    template <typename P1, typename P2>
        requires ElemOperrable<P1, P2>
    auto operator+(P1&& m1, P2&& m2)
    {
        using Cate1 = DataCategory_t<P1>;
        using Cate2 = DataCategory_t<P2>;

        return details::OperatorTrivial<BinaryOperTags::Add, P1, P2>::template eval<Cate1, Cate2>(
            std::forward<P1>(m1), std::forward<P2>(m2));
    }

    template <typename P1, typename P2>
        requires ElemOperrable<P1, P2>
    auto operator-(P1&& m1, P2&& m2)
    {
        using Cate1 = DataCategory_t<P1>;
        using Cate2 = DataCategory_t<P2>;

        return details::OperatorTrivial<BinaryOperTags::Sub, P1, P2>::template eval<Cate1, Cate2>(
            std::forward<P1>(m1), std::forward<P2>(m2));
    }

    template <typename P1, typename P2>
        requires ElemOperrable<P1, P2>
    auto operator*(P1&& m1, P2&& m2)
    {
        using Cate1 = DataCategory_t<P1>;
        using Cate2 = DataCategory_t<P2>;

        return details::OperatorTrivial<BinaryOperTags::Mul, P1, P2>::template eval<Cate1, Cate2>(
            std::forward<P1>(m1), std::forward<P2>(m2));
    }

    template <typename P1, typename P2>
        requires ElemOperrable<P1, P2>
    auto operator/(P1&& m1, P2&& m2)
    {
        using Cate1 = DataCategory_t<P1>;
        using Cate2 = DataCategory_t<P2>;

        return details::OperatorTrivial<BinaryOperTags::Div, P1, P2>::template eval<Cate1, Cate2>(
            std::forward<P1>(m1), std::forward<P2>(m2));
    }

    /*************************************** Dot operator *******************************************/
    template <typename T, typename U>
    concept Dottable = (IsMatrix_v<T> && IsMatrix_v<U>)
        || Commutative_v<T, U, IsMatrix, IsBatchMatrix>
        || (IsBatchMatrix_v<T> && IsBatchMatrix_v<U>);

    // change the default behavior
    template <>
    class OperOrganizer<BinaryOperTags::Dot, CategoryTags::Matrix>
    {
    public:
        template <DataConcept DataType1, DataConcept DataType2>
        OperOrganizer(const DataType1& m1, const DataType2& m2)
            : m_rowNum(m1.rowNum())
              , m_colNum(m2.colNum())
        {
            assert(m1.colNum() == m2.rowNum());
        }

        [[nodiscard]] std::size_t rowNum() const { return m_rowNum; }
        [[nodiscard]] std::size_t colNum() const { return m_colNum; }

    private:
        std::size_t m_rowNum;
        std::size_t m_colNum;
    };

    template <>
    class OperOrganizer<BinaryOperTags::Dot, CategoryTags::BatchMatrix>
    {
    public:
        template <DataConcept DataType1, DataConcept DataType2>
        OperOrganizer(const DataType1& m1, const DataType2& m2)
            : m_rowNum(m1.rowNum())
              , m_colNum(m2.colNum())
              , m_batchNum(m1.batchNum())
        {
            assert(m1.colNum() == m2.rowNum() && m1.batchNum() == m2.batchNum());
        }

        [[nodiscard]] std::size_t rowNum() const { return m_rowNum; }
        [[nodiscard]] std::size_t colNum() const { return m_colNum; }
        [[nodiscard]] std::size_t batchNum() const { return m_batchNum; }

    private:
        std::size_t m_rowNum;
        std::size_t m_colNum;
        std::size_t m_batchNum;
    };

    namespace details
    {
        template <typename P1, typename P2>
            requires ElemOperrable<P1, P2>
        struct OperatorDot
        {
        private:
            using RawM1 = std::remove_cvref_t<P1>;
            using RawM2 = std::remove_cvref_t<P2>;

        public:
            template <CategoryConcept T, CategoryConcept U>
                requires std::is_same_v<T, U>
            static auto eval(P1&& m1, P2&& m2)
            {
                static_assert(std::is_same_v<typename RawM1::DeviceType, typename RawM2::DeviceType>,
                              "Matrices with different device types cannot dot directly");
                static_assert(std::is_same_v<typename RawM1::ElementType, typename RawM2::ElementType>,
                              "Matrices with different element types cannot dot directly");
                using ResType = BinaryOperator<BinaryOperTags::Dot, RawM1, RawM2>;
                return ResType(std::forward<P1>(m1), std::forward<P2>(m2));
            }

            template <CategoryConcept T, CategoryConcept U>
                requires std::is_same_v<T, CategoryTags::BatchMatrix> && std::is_same_v<U, CategoryTags::Matrix>
            static auto eval(P1&& m1, P2&& m2)
            {
                static_assert(std::is_same_v<typename RawM1::DeviceType, typename RawM2::DeviceType>,
                              "Matrices with different device types cannot add directly");
                static_assert(std::is_same_v<typename RawM1::ElementType, typename RawM2::ElementType>,
                              "Matrices with different element types cannot add directly");
                Duplicate<RawM2> tmp{std::forward<P2>(m2), m1.batchNum()};
                using ResType = BinaryOperator<BinaryOperTags::Dot, RawM1, Duplicate<RawM2>>;
                return ResType(std::forward<P1>(m1), std::move(tmp));
            }

            template <CategoryConcept T, CategoryConcept U>
                requires std::is_same_v<T, CategoryTags::Matrix> && std::is_same_v<U, CategoryTags::BatchMatrix>
            static auto eval(P1&& m1, P2&& m2)
            {
                static_assert(std::is_same_v<typename RawM1::DeviceType, typename RawM2::DeviceType>,
                              "Matrices with different device types cannot add directly");
                static_assert(std::is_same_v<typename RawM1::ElementType, typename RawM2::ElementType>,
                              "Matrices with different element types cannot add directly");
                Duplicate<RawM1> tmp{std::forward<P1>(m1), m2.batchNum()};
                using ResType = BinaryOperator<BinaryOperTags::Dot, Duplicate<RawM1>, RawM2>;
                return ResType(std::move(tmp), std::forward<P2>(m2));
            }

        public:
        };
    } // namespace details
    template <>
    struct OperSeq<BinaryOperTags::Dot>
    {
        using type = OperSeqContainer<eval::BinaryCalculator<BinaryOperTags::Dot>>;
    };

    template <>
    struct eval::UnitWrapper<BinaryOperTags::Dot>
    {
        template <typename OperHandle1, typename OperHandle2, typename Element,
                  DeviceConcept Device, CategoryConcept Category>
        class EvalUnit;

        template <typename OperHandle1, typename OperHandle2, typename Element>
        class EvalUnit<OperHandle1, OperHandle2, Element, CPU, CategoryTags::Matrix>
            : public BaseEvalUnit<CPU>
        {
        public:
            using ElementType = Element;
            using DeviceType = CPU;

            EvalUnit(OperHandle1 op1, OperHandle2 op2,
                     EvalHandle<Matrix<Element, CPU>> evalOutput):
                m_oper1(std::move(op1)), m_oper2(std::move(op2)), m_evalOutput(std::move(evalOutput))
            {
            }

            void eval() override
            {
                const auto& pV1 = m_oper1.data();
                const auto& pV2 = m_oper2.data();
                const std::size_t rowNum = pV1.rowNum();
                const std::size_t colNum = pV2.colNum();
                const std::size_t midNum = pV1.colNum();
                assert(pV2.rowNum() == midNum);

                m_evalOutput.allocate(rowNum, colNum);
                auto& res = m_evalOutput.mutableData();
                auto mem_res = lower_access(res);
                const std::size_t tgtPackNum = mem_res.rowLen();
                auto* r = mem_res.mutableRawMemory();
                for (std::size_t i = 0; i < rowNum; ++i)
                {
                    for (std::size_t j = 0; j < colNum; ++j)
                    {
                        *r = Element{};
                        for (std::size_t k = 0; k < midNum; ++k)
                        {
                            *r += pV1(i, k) * pV2(k, j);
                        }
                        ++r;
                    }
                    r += tgtPackNum - colNum;
                }
                m_evalOutput.setEval();
            }

        private:
            OperHandle1 m_oper1;
            OperHandle2 m_oper2;
            EvalHandle<Matrix<Element, CPU>> m_evalOutput;
        };

        template <typename OperHandle1, typename OperHandle2, typename Element>
        class EvalUnit<OperHandle1, OperHandle2, Element, CPU, CategoryTags::BatchMatrix>
            : public BaseEvalUnit<CPU>
        {
        public:
            using ElementType = Element;
            using DeviceType = CPU;

            EvalUnit(OperHandle1 op1, OperHandle2 op2,
                     EvalHandle<Batch<ElementType, DeviceType, CategoryTags::Matrix>> evalOutput):
                m_oper1(std::move(op1)), m_oper2(std::move(op2)), m_evalOutput(std::move(evalOutput))
            {
            }

            void eval() override
            {
                const auto& pV1 = m_oper1.data();
                const auto& pV2 = m_oper2.data();
                const std::size_t rowNum = pV1.rowNum();
                const std::size_t colNum = pV2.colNum();
                const std::size_t midNum = pV1.colNum();
                const std::size_t batchNum = pV1.batchNum();
                assert(pV2.rowNum() == midNum);
                assert(pV2.batchNum() == batchNum);

                m_evalOutput.allocate(batchNum, rowNum, colNum);
                auto& res = m_evalOutput.mutableData();
                for (std::size_t batch = 0; batch < batchNum; ++batch)
                {
                    auto mem_res = lower_access(res[batch]);
                    const std::size_t tgtPackNum = mem_res.rowLen();
                    auto* r = mem_res.mutableRawMemory();
                    const auto& curV1 = pV1[batch];
                    const auto& curV2 = pV2[batch];
                    for (std::size_t i = 0; i < rowNum; ++i)
                    {
                        for (std::size_t j = 0; j < colNum; ++j)
                        {
                            *r = Element{};
                            for (std::size_t k = 0; k < midNum; ++k)
                            {
                                *r += curV1(i, k) * curV2(k, j);
                            }
                            ++r;
                        }
                        r += tgtPackNum - colNum;
                    }
                }
                m_evalOutput.setEval();
            }

        private:
            OperHandle1 m_oper1;
            OperHandle2 m_oper2;
            EvalHandle<Batch<ElementType, DeviceType, CategoryTags::Matrix>> m_evalOutput;
        };
    };

    template <typename P1, typename P2>
        requires Dottable<P1, P2>
    auto dot(P1&& m1, P2&& m2)
    {
        using Cate1 = DataCategory_t<P1>;
        using Cate2 = DataCategory_t<P2>;

        return details::OperatorDot<P1, P2>::template eval<Cate1, Cate2>(std::forward<P1>(m1), std::forward<P2>(m2));
    }

    /************************************** NegativeLogLikelihood Operator **************************************/
    template <>
    struct OperCategory<BinaryOperTags::NegativeLogLikelihood, CategoryTags::Matrix, CategoryTags::Matrix>
    {
        using type = CategoryTags::Scalar;
    };

    template <>
    struct OperCategory<BinaryOperTags::NegativeLogLikelihood, CategoryTags::BatchMatrix, CategoryTags::BatchMatrix>
    {
        using type = CategoryTags::BatchScalar;
    };

    namespace details
    {
        template <BinaryOperConcept OperTag, typename P1, typename P2>
        struct BinaryOperatorForSameDataType
        {
        private:
            using RawM1 = std::remove_cvref_t<P1>;
            using RawM2 = std::remove_cvref_t<P2>;

        public:
            template <CategoryConcept T, CategoryConcept U>
                requires std::is_same_v<T, U>
            static auto eval(P1&& m1, P2&& m2)
            {
                static_assert(std::is_same_v<typename RawM1::DeviceType, typename RawM2::DeviceType>,
                              "Matrices with different device types cannot add directly");
                static_assert(std::is_same_v<typename RawM1::ElementType, typename RawM2::ElementType>,
                              "Matrices with different element types cannot add directly");
                using ResType = BinaryOperator<OperTag, RawM1, RawM2>;
                return ResType(std::forward<P1>(m1), std::forward<P2>(m2));
            }
        };

        // template <typename P1, typename P2>
        //     requires (IsMatrix_v<P1> && IsMatrix_v<P2>) || (IsBatchMatrix_v<P1> && IsBatchMatrix_v<P2>)
        // struct OperatorNegLogLikelihood
        // {
        // private:
        //     using RawM1 = std::remove_cvref_t<P1>;
        //     using RawM2 = std::remove_cvref_t<P2>;
        //
        // public:
        //     template <CategoryConcept T, CategoryConcept U>
        //         requires std::is_same_v<T, U>
        //     static auto eval(T&& m1, U&& m2)
        //     {
        //         static_assert(std::is_same_v<typename RawM1::DeviceType, typename RawM2::DeviceType>,
        //                       "Matrices with different device types cannot add directly");
        //         static_assert(std::is_same_v<typename RawM1::ElementType, typename RawM2::ElementType>,
        //                       "Matrices with different element types cannot add directly");
        //         using ResType = BinaryOperator<BinaryOperTags::NegativeLogLikelihood, RawM1, RawM2>;
        //         return ResType(std::forward<T>(m1), std::forward<U>(m2));
        //     }
        //
        // public:
        // };
    } // namespace details

    template <DataConcept P1, DataConcept P2>
        requires(IsMatrix_v<P1> && IsMatrix_v<P2>) || (IsBatchMatrix_v<P1> && IsBatchMatrix_v<P2>)
    auto neg_log_likelihood(P1&& m1, P2&& m2)
    {
        using Cate1 = DataCategory_t<P1>;
        using Cate2 = DataCategory_t<P2>;

        return details::BinaryOperatorForSameDataType<
            BinaryOperTags::NegativeLogLikelihood, P1, P2>::template eval<Cate1, Cate2>(
            std::forward<P1>(m1), std::forward<P2>(m2));
    }

    template <>
    struct eval::UnitWrapper<BinaryOperTags::NegativeLogLikelihood>
    {
        template <typename OperHandle1, typename OperHandle2, typename Element,
                  DeviceConcept Device, CategoryConcept Category>
        class EvalUnit;

        template <typename OperHandle1, typename OperHandle2, typename Element>
        class EvalUnit<OperHandle1, OperHandle2, Element, CPU, CategoryTags::Scalar>
            : public BaseEvalUnit<CPU>
        {
        public:
            using ElementType = Element;
            using DeviceType = CPU;

            EvalUnit(OperHandle1 op1, OperHandle2 op2,
                     EvalHandle<Scalar<Element, CPU>> evalOutput):
                m_oper1(std::move(op1)), m_oper2(std::move(op2)), m_evalOutput(std::move(evalOutput))
            {
            }

            void eval() override
            {
                const auto& pV1 = m_oper1.data();
                const auto& pV2 = m_oper2.data();
                const std::size_t rowNum = pV1.rowNum();
                const std::size_t colNum = pV1.colNum();
                assert(pV2.rowNum() == rowNum);
                assert(pV2.colNum() == colNum);

                m_evalOutput.allocate();
                auto res = Element{};
                // auto mem_res = lower_access(res);
                auto mem_v1 = lower_access(pV1);
                auto mem_v2 = lower_access(pV2);
                const std::size_t src1PackNum = pV1.rowLen();
                const std::size_t src2PackNum = pV2.rowLen();
                const auto* r1 = mem_v1.rawMemory();
                const auto* r2 = mem_v2.rawMemory();
                for (std::size_t i = 0; i < rowNum; ++i)
                {
                    for (std::size_t j = 0; j < colNum; ++j)
                    {
                        res -= r1[j] * std::log(r2[j]);
                    }
                    r1 += src1PackNum;
                    r2 += src2PackNum;
                }
                m_evalOutput.mutableData().value() = res;
                m_evalOutput.setEval();
            }

        private:
            OperHandle1 m_oper1;
            OperHandle2 m_oper2;
            EvalHandle<Scalar<Element, CPU>> m_evalOutput;
        };

        template <typename OperHandle1, typename OperHandle2, typename Element>
        class EvalUnit<OperHandle1, OperHandle2, Element, CPU, CategoryTags::BatchScalar>
            : public BaseEvalUnit<CPU>
        {
        public:
            using ElementType = Element;
            using DeviceType = CPU;

            EvalUnit(OperHandle1 op1, OperHandle2 op2,
                     EvalHandle<Batch<ElementType, DeviceType, CategoryTags::Scalar>> evalOutput):
                m_oper1(std::move(op1)), m_oper2(std::move(op2)), m_evalOutput(std::move(evalOutput))
            {
            }

            void eval() override
            {
                const auto& pTar = m_oper1.data();
                const auto& pPre = m_oper2.data();

                const size_t rowNum = pTar.rowNum();
                const size_t colNum = pTar.colNum();
                const size_t batchNum = pTar.batchNum();
                assert(pPre.rowNum() == rowNum);
                assert(pPre.colNum() == colNum);
                assert(pPre.batchNum() == batchNum);

                m_evalOutput.allocate(batchNum);
                auto& aim = m_evalOutput.mutableData();

                for (std::size_t curBatch = 0; curBatch < batchNum; ++curBatch)
                {
                    auto res = ElementType{};

                    auto mem_v1 = lower_access(pTar[curBatch]);
                    auto mem_v2 = lower_access(pPre[curBatch]);

                    const size_t src1PackNum = mem_v1.rowLen();
                    const size_t src2PackNum = mem_v2.rowLen();

                    const ElementType* r1 = mem_v1.rawMemory();
                    const ElementType* r2 = mem_v2.rawMemory();

                    for (size_t i = 0; i < rowNum; ++i)
                    {
                        for (size_t j = 0; j < colNum; ++j)
                        {
                            res -= r1[j] * log(r2[j]);
                        }
                        r1 += src1PackNum;
                        r2 += src2PackNum;
                    }

                    aim.setValue(curBatch, res);
                }
                m_evalOutput.setEval();
            }

        private:
            OperHandle1 m_oper1;
            OperHandle2 m_oper2;
            EvalHandle<Batch<ElementType, DeviceType, CategoryTags::Scalar>> m_evalOutput;
        };
    };

    template <>
    struct OperSeq<BinaryOperTags::NegativeLogLikelihood>
    {
        using type = OperSeqContainer<eval::BinaryCalculator<BinaryOperTags::NegativeLogLikelihood>>;
    };

    /************************************** VecSoftmaxDerivative Operator **************************************/
    namespace details
    {
        // template <typename P1, typename P2>
        //     requires (IsMatrix_v<P1> && IsMatrix_v<P2>) || (IsBatchMatrix_v<P1> && IsBatchMatrix_v<P2>)
        // struct OperatorSoftmaxDerivative
        // {
        // private:
        //     using RawM1 = std::remove_cvref_t<P1>;
        //     using RawM2 = std::remove_cvref_t<P2>;
        //
        // public:
        //     template <CategoryConcept T, CategoryConcept U>
        //         requires std::is_same_v<T, U>
        //     static auto eval(T&& m1, U&& m2)
        //     {
        //         static_assert(std::is_same_v<typename RawM1::DeviceType, typename RawM2::DeviceType>,
        //                       "Matrices with different device types cannot add directly");
        //         static_assert(std::is_same_v<typename RawM1::ElementType, typename RawM2::ElementType>,
        //                       "Matrices with different element types cannot add directly");
        //         using ResType = BinaryOperator<BinaryOperTags::VecSoftmaxDerivative, RawM1, RawM2>;
        //         return ResType(std::forward<T>(m1), std::forward<U>(m2));
        //     }
        //
        // public:
        // };
    } // namespace details
    // ! Need to carefully coding
    template <DataConcept P1, DataConcept P2>
        requires(IsMatrix_v<P1> && IsMatrix_v<P2>) || (IsBatchMatrix_v<P1> && IsBatchMatrix_v<P2>)
    auto softmax_derivative(P1&& m1, P2&& m2)
    {
        using Cate1 = DataCategory_t<P1>;
        using Cate2 = DataCategory_t<P2>;

        return details::BinaryOperatorForSameDataType<
            BinaryOperTags::VecSoftmaxDerivative, P1, P2>::template eval<Cate1, Cate2>(
            std::forward<P1>(m1), std::forward<P2>(m2));
    }

    template <>
    struct eval::UnitWrapper<BinaryOperTags::VecSoftmaxDerivative>
    {
        template <typename OperHandle1, typename OperHandle2, typename Element,
                  DeviceConcept Device, CategoryConcept Category>
        class EvalUnit;

        template <typename OperHandle1, typename OperHandle2, typename Element>
        class EvalUnit<OperHandle1, OperHandle2, Element, CPU, CategoryTags::Matrix>
            : public BaseEvalUnit<CPU>
        {
        public:
            using ElementType = Element;
            using DeviceType = CPU;

            EvalUnit(OperHandle1 op1, OperHandle2 op2,
                     EvalHandle<Matrix<Element, CPU>> evalOutput):
                m_oper1(std::move(op1)), m_oper2(std::move(op2)), m_evalOutput(std::move(evalOutput))
            {
            }

            void eval() override
            {
                const auto& pV1 = m_oper1.data();
                const auto& pV2 = m_oper2.data();
                // const std::size_t rowNum = pV1.rowNum();
                const std::size_t colNum = pV1.colNum();
                assert(pV1.rowNum() == 1);
                assert(pV2.rowNum() == 1);
                assert(pV2.colNum() == colNum);
                Matrix<ElementType, DeviceType> tmp{colNum, colNum};
                for (std::size_t i = 0; i < colNum; ++i)
                {
                    for (std::size_t j = 0; j < colNum; ++j)
                    {
                        tmp.setValue(i, j, -1 * pV2(0, i) * pV2(0, j));
                    }
                    auto value = tmp(i, i);
                    tmp.setValue(i, i, pV2(0, i) + value);
                }
                auto tempHandle = tmp.evalRegister();
                using EvalUnit = UnitWrapper<BinaryOperTags::Dot>::EvalUnit<
                    decltype(m_oper1), decltype(tempHandle),
                    ElementType, DeviceType,
                    CategoryTags::Matrix
                >;
                using GroupType = TrivialEvalGroup<EvalUnit>;
                const void* dataPtr = m_evalOutput.dataPtr();
                auto depVec = {m_oper1.dataPtr(), tempHandle.dataPtr()};
                EvalUnit unit{m_oper1, std::move(tempHandle), std::move(m_evalOutput)};
                EvalPlan<DeviceType>::registerFun<GroupType>(std::move(unit), dataPtr, std::move(depVec));
            }

        private:
            OperHandle1 m_oper1;
            OperHandle2 m_oper2;
            EvalHandle<Matrix<Element, CPU>> m_evalOutput;
        };

        template <typename OperHandle1, typename OperHandle2, typename Element>
        class EvalUnit<OperHandle1, OperHandle2, Element, CPU, CategoryTags::BatchMatrix>
            : public BaseEvalUnit<CPU>
        {
        public:
            using ElementType = Element;
            using DeviceType = CPU;

            EvalUnit(OperHandle1 op1, OperHandle2 op2,
                     EvalHandle<Batch<ElementType, DeviceType, CategoryTags::Matrix>> evalOutput):
                m_oper1(std::move(op1)), m_oper2(std::move(op2)), m_evalOutput(std::move(evalOutput))
            {
            }

            void eval() override
            {
                const auto& pV1 = m_oper1.data();
                const auto& pV2 = m_oper2.data();
                // const std::size_t rowNum = pV1.rowNum();
                const std::size_t colNum = pV2.colNum();
                const std::size_t batchNum = pV1.batchNum();
                assert(pV2.rowNum() == 1);
                assert(pV1.rowNum() == 1);
                assert(pV2.colNum() == colNum);
                assert(pV2.batchNum() == batchNum);
                Batch<ElementType, DeviceType, CategoryTags::Matrix> tmp{
                    batchNum, colNum, colNum
                };
                for (std::size_t curBatch = 0; curBatch < batchNum; ++curBatch)
                {
                    for (std::size_t i = 0; i < colNum; ++i)
                    {
                        for (std::size_t j = 0; j < colNum; ++j)
                        {
                            tmp.setValue(curBatch, i, j,
                                         -1 * pV2[curBatch](0, i) * pV2[curBatch](0, j));
                        }
                        auto value = tmp[curBatch](i, i);
                        tmp.setValue(curBatch, i, i, pV2[curBatch](0, i) + value);
                    }
                }
                auto tempHandle = tmp.evalRegister();
                using EvalUnit = UnitWrapper<BinaryOperTags::Dot>::EvalUnit<
                decltype(m_oper1), decltype(m_oper2),ElementType, DeviceType,
                CategoryTags::BatchMatrix
                    >;
                using GroupType = TrivialEvalGroup<EvalUnit>;
                const void* dataPtr = m_evalOutput.dataPtr();
                auto depVec = {m_oper1.dataPtr(), tempHandle.dataPtr()};
                EvalUnit unit(m_oper1, std::move(tempHandle), std::move(m_evalOutput));
                EvalPlan<DeviceType>::template registerFun<GroupType>(std::move(unit), dataPtr, std::move(depVec));
            }

        private:
            OperHandle1 m_oper1;
            OperHandle2 m_oper2;
            EvalHandle<Batch<ElementType, DeviceType, CategoryTags::Matrix>> m_evalOutput;
        };
    };

    template <>
    struct OperSeq<BinaryOperTags::VecSoftmaxDerivative>
    {
        using type = OperSeqContainer<eval::BinaryCalculator<BinaryOperTags::VecSoftmaxDerivative>>;
    };


    /************************************** SigmoidDerivative Operator **************************************/
    namespace details
    {
        // template <typename P1, typename P2>
        //     requires (IsMatrix_v<P1> && IsMatrix_v<P2>) || (IsBatchMatrix_v<P1> && IsBatchMatrix_v<P2>)
        // struct OperatorSigmoidDerivative
        // {
        // private:
        //     using RawM1 = std::remove_cvref_t<P1>;
        //     using RawM2 = std::remove_cvref_t<P2>;
        //
        // public:
        //     template <CategoryConcept T, CategoryConcept U>
        //         requires std::is_same_v<T, U>
        //     static auto eval(T&& m1, U&& m2)
        //     {
        //         static_assert(std::is_same_v<typename RawM1::DeviceType, typename RawM2::DeviceType>,
        //                       "Matrices with different device types cannot add directly");
        //         static_assert(std::is_same_v<typename RawM1::ElementType, typename RawM2::ElementType>,
        //                       "Matrices with different element types cannot add directly");
        //         using ResType = BinaryOperator<BinaryOperTags::SigmoidDerivative, RawM1, RawM2>;
        //         return ResType(std::forward<T>(m1), std::forward<U>(m2));
        //     }
        //
        // public:
        // };
    } // namespace details

    template <DataConcept P1, DataConcept P2>
        requires(IsMatrix_v<P1> && IsMatrix_v<P2>) || (IsBatchMatrix_v<P1> && IsBatchMatrix_v<P2>)
    auto sigmoid_derivative(P1&& m1, P2&& m2)
    {
        using Cate1 = DataCategory_t<P1>;
        using Cate2 = DataCategory_t<P2>;

        return details::BinaryOperatorForSameDataType<
            BinaryOperTags::SigmoidDerivative, P1, P2>::template eval<Cate1, Cate2>(
            std::forward<P1>(m1), std::forward<P2>(m2));
    }

    template <>
    struct eval::UnitWrapper<BinaryOperTags::SigmoidDerivative>
    {
        template <typename OperHandle1, typename OperHandle2, typename Element,
                  DeviceConcept Device, CategoryConcept Category>
        class EvalUnit;

        template <typename OperHandle1, typename OperHandle2, typename Element>
        class EvalUnit<OperHandle1, OperHandle2, Element, CPU, CategoryTags::Matrix>
            : public BaseEvalUnit<CPU>
        {
        public:
            using ElementType = Element;
            using DeviceType = CPU;

            EvalUnit(OperHandle1 op1, OperHandle2 op2,
                     EvalHandle<Matrix<Element, CPU>> evalOutput):
                m_oper1(std::move(op1)), m_oper2(std::move(op2)), m_evalOutput(std::move(evalOutput))
            {
            }

            void eval() override
            {
                const auto& pV1 = m_oper1.data();
                const auto& pV2 = m_oper2.data();
                const std::size_t rowNum = pV1.rowNum();
                const std::size_t colNum = pV1.colNum();
                assert(pV2.rowNum() == rowNum);
                assert(pV2.colNum() == colNum);

                m_evalOutput.allocate(rowNum, colNum);
                auto& res = m_evalOutput.mutableData();
                auto mem_res = lower_access(res);
                auto mem_out = lower_access(pV2);
                auto mem_grad = lower_access(pV1);
                const std::size_t src1PackNum = pV1.rowLen();
                const std::size_t src2PackNum = pV2.rowLen();
                const std::size_t tgtPackNum = mem_res.rowLen();
                auto* r = mem_res.mutableRawMemory();
                const auto* r1 = mem_grad.rawMemory();
                const auto* r2 = mem_out.rawMemory();

                // TODO: maybe some mistake
                for (std::size_t i = 0; i < rowNum; ++i)
                {
                    for (std::size_t j = 0; j < colNum; ++j)
                    {
                        r[j] = r1[j] * (1 - r2[j]) * r2[j];
                    }
                    r += tgtPackNum;
                    r1 += src1PackNum;
                    r2 += src2PackNum;
                }
                m_evalOutput.setEval();
            }

        private:
            OperHandle1 m_oper1;
            OperHandle2 m_oper2;
            EvalHandle<Matrix<Element, CPU>> m_evalOutput;
        };

        template <typename OperHandle1, typename OperHandle2, typename Element>
        class EvalUnit<OperHandle1, OperHandle2, Element, CPU, CategoryTags::BatchMatrix>
            : public BaseEvalUnit<CPU>
        {
        public:
            using ElementType = Element;
            using DeviceType = CPU;

            EvalUnit(OperHandle1 op1, OperHandle2 op2,
                     EvalHandle<Batch<ElementType, DeviceType, CategoryTags::Matrix>> evalOutput):
                m_oper1(std::move(op1)), m_oper2(std::move(op2)), m_evalOutput(std::move(evalOutput))
            {
            }

            void eval() override
            {
                const auto& pV1 = m_oper1.data();
                const auto& pV2 = m_oper2.data();
                const std::size_t rowNum = pV1.rowNum();
                const std::size_t colNum = pV2.colNum();
                const std::size_t batchNum = pV1.batchNum();
                assert(pV2.rowNum() == rowNum);
                assert(pV2.colNum() == colNum);
                assert(pV2.batchNum() == batchNum);

                m_evalOutput.allocate(batchNum, rowNum, colNum);
                auto& res = m_evalOutput.mutableData();
                for (std::size_t batch = 0; batch < batchNum; ++batch)
                {
                    auto mem_res = lower_access(res[batch]);
                    auto mem_out = lower_access(pV2[batch]);
                    auto mem_grad = lower_access(pV1[batch]);
                    const std::size_t src1PackNum = mem_grad.rowLen();
                    const std::size_t src2PackNum = mem_out.rowLen();
                    const std::size_t tgtPackNum = mem_res.rowLen();
                    auto* r = mem_res.mutableRawMemory();
                    const auto* r1 = mem_grad.rawMemory();
                    const auto* r2 = mem_out.rawMemory();
                    for (std::size_t i = 0; i < rowNum; ++i)
                    {
                        for (std::size_t j = 0; j < colNum; ++j)
                        {
                            r[j] = r1[j] * (1 - r2[j]) * r2[j];
                        }
                        r += tgtPackNum;
                        r1 += src1PackNum;
                        r2 += src2PackNum;
                    }
                }
                m_evalOutput.setEval();
            }

        private:
            OperHandle1 m_oper1;
            OperHandle2 m_oper2;
            EvalHandle<Batch<ElementType, DeviceType, CategoryTags::Matrix>> m_evalOutput;
        };
    };

    template <>
    struct OperSeq<BinaryOperTags::SigmoidDerivative>
    {
        using type = OperSeqContainer<eval::BinaryCalculator<BinaryOperTags::SigmoidDerivative>>;
    };

    /************************************** TanhDerivative Operator **************************************/
    namespace details
    {
        // template <typename P1, typename P2>
        //     requires (IsMatrix_v<P1> && IsMatrix_v<P2>) || (IsBatchMatrix_v<P1> && IsBatchMatrix_v<P2>)
        // struct OperatorTanhDerivative
        // {
        // private:
        //     using RawM1 = std::remove_cvref_t<P1>;
        //     using RawM2 = std::remove_cvref_t<P2>;
        //
        // public:
        //     template <CategoryConcept T, CategoryConcept U>
        //         requires std::is_same_v<T, U>
        //     static auto eval(T&& m1, U&& m2)
        //     {
        //         static_assert(std::is_same_v<typename RawM1::DeviceType, typename RawM2::DeviceType>,
        //                       "Matrices with different device types cannot add directly");
        //         static_assert(std::is_same_v<typename RawM1::ElementType, typename RawM2::ElementType>,
        //                       "Matrices with different element types cannot add directly");
        //         using ResType = BinaryOperator<BinaryOperTags::TanhDerivative, RawM1, RawM2>;
        //         return ResType(std::forward<T>(m1), std::forward<U>(m2));
        //     }
        //
        // public:
        // };
    } // namespace details
    template <DataConcept P1, DataConcept P2>
        requires(IsMatrix_v<P1> && IsMatrix_v<P2>) || (IsBatchMatrix_v<P1> && IsBatchMatrix_v<P2>)
    auto tanh_derivative(P1&& m1, P2&& m2)
    {
        using Cate1 = DataCategory_t<P1>;
        using Cate2 = DataCategory_t<P2>;

        return details::BinaryOperatorForSameDataType<
            BinaryOperTags::TanhDerivative, P1, P2>::template eval<Cate1, Cate2>(
            std::forward<P1>(m1), std::forward<P2>(m2));
    }

    template <>
    struct eval::UnitWrapper<BinaryOperTags::TanhDerivative>
    {
        template <typename OperHandle1, typename OperHandle2, typename Element,
                  DeviceConcept Device, CategoryConcept Category>
        class EvalUnit;

        template <typename OperHandle1, typename OperHandle2, typename Element>
        class EvalUnit<OperHandle1, OperHandle2, Element, CPU, CategoryTags::Matrix>
            : public BaseEvalUnit<CPU>
        {
        public:
            using ElementType = Element;
            using DeviceType = CPU;

            EvalUnit(OperHandle1 op1, OperHandle2 op2,
                     EvalHandle<Matrix<Element, CPU>> evalOutput):
                m_oper1(std::move(op1)), m_oper2(std::move(op2)), m_evalOutput(std::move(evalOutput))
            {
            }

            void eval() override
            {
                const auto& pV1 = m_oper1.data();
                const auto& pV2 = m_oper2.data();
                const std::size_t rowNum = pV1.rowNum();
                const std::size_t colNum = pV1.colNum();
                assert(pV2.rowNum() == rowNum);
                assert(pV2.colNum() == colNum);

                m_evalOutput.allocate(rowNum, colNum);
                auto& res = m_evalOutput.mutableData();
                auto mem_res = lower_access(res);
                auto mem_out = lower_access(pV2);
                auto mem_grad = lower_access(pV1);
                const std::size_t src1PackNum = pV1.rowLen();
                const std::size_t src2PackNum = pV2.rowLen();
                const std::size_t tgtPackNum = mem_res.rowLen();
                auto* r = mem_res.mutableRawMemory();
                const auto* r1 = mem_grad.rawMemory();
                const auto* r2 = mem_out.rawMemory();
                for (std::size_t i = 0; i < rowNum; ++i)
                {
                    for (std::size_t j = 0; j < colNum; ++j)
                    {
                        r[j] = r1[j] * (1 - r2[j] * r2[j]);
                    }
                    r += tgtPackNum;
                    r1 += src1PackNum;
                    r2 += src2PackNum;
                }
                m_evalOutput.setEval();
            }

        private:
            OperHandle1 m_oper1;
            OperHandle2 m_oper2;
            EvalHandle<Matrix<Element, CPU>> m_evalOutput;
        };

        template <typename OperHandle1, typename OperHandle2, typename Element>
        class EvalUnit<OperHandle1, OperHandle2, Element, CPU, CategoryTags::BatchMatrix>
            : public BaseEvalUnit<CPU>
        {
        public:
            using ElementType = Element;
            using DeviceType = CPU;

            EvalUnit(OperHandle1 op1, OperHandle2 op2,
                     EvalHandle<Batch<ElementType, DeviceType, CategoryTags::Matrix>> evalOutput):
                m_oper1(std::move(op1)), m_oper2(std::move(op2)), m_evalOutput(std::move(evalOutput))
            {
            }

            void eval() override
            {
                const auto& pV1 = m_oper1.data();
                const auto& pV2 = m_oper2.data();
                const std::size_t rowNum = pV1.rowNum();
                const std::size_t colNum = pV2.colNum();
                const std::size_t batchNum = pV1.batchNum();
                assert(pV2.rowNum() == rowNum);
                assert(pV2.colNum() == colNum);
                assert(pV2.batchNum() == batchNum);

                m_evalOutput.allocate(batchNum, rowNum, colNum);
                auto& res = m_evalOutput.mutableData();
                for (std::size_t batch = 0; batch < batchNum; ++batch)
                {
                    auto mem_res = lower_access(res[batch]);
                    auto mem_out = lower_access(pV2[batch]);
                    auto mem_grad = lower_access(pV1[batch]);
                    const std::size_t src1PackNum = mem_grad.rowLen();
                    const std::size_t src2PackNum = mem_out.rowLen();
                    const std::size_t tgtPackNum = mem_res.rowLen();
                    auto* r = mem_res.mutableRawMemory();
                    const auto* r1 = mem_grad.rawMemory();
                    const auto* r2 = mem_out.rawMemory();
                    for (std::size_t i = 0; i < rowNum; ++i)
                    {
                        for (std::size_t j = 0; j < colNum; ++j)
                        {
                            r[j] = r1[j] * (1 - r2[j] * r2[j]);
                        }
                        r += tgtPackNum;
                        r1 += src1PackNum;
                        r2 += src2PackNum;
                    }
                }
                m_evalOutput.setEval();
            }

        private:
            OperHandle1 m_oper1;
            OperHandle2 m_oper2;
            EvalHandle<Batch<ElementType, DeviceType, CategoryTags::Matrix>> m_evalOutput;
        };
    };

    template <>
    struct OperSeq<BinaryOperTags::TanhDerivative>
    {
        using type = OperSeqContainer<eval::BinaryCalculator<BinaryOperTags::TanhDerivative>>;
    };
} // namespace metann

#endif // BINARY_OPERATORS_HPP
