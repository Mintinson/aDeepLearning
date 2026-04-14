//
// Created by asus on 2025/1/11.
//

#ifndef UNARY_OPERATORS_HPP
#define UNARY_OPERATORS_HPP

#include <algorithm>
#include <cmath>

#include "../data/allocator.hpp"
#include "../data/batch.hpp"
#include "../data/data_category.hpp"
#include "../data/matrix.hpp"
#include "../utils/type_traits.hpp"
#include "operator_category.hpp"
#include "operator_helper.hpp"

namespace metann {
template <OperTagConcept OperTag, DataConcept DataType>
class UnaryOperator;

template <OperTagConcept OperTag, DataConcept DataType>
constexpr bool IsScalarHelper_v<UnaryOperator<OperTag, DataType>> =
    std::is_same_v<OperaCateCal_t<OperTag, DataType>, CategoryTags::Scalar>;
template <OperTagConcept OperTag, DataConcept DataType>
constexpr bool IsMatrixHelper_v<UnaryOperator<OperTag, DataType>> =
    std::is_same_v<OperaCateCal_t<OperTag, DataType>, CategoryTags::Matrix>;
template <OperTagConcept OperTag, DataConcept DataType>
constexpr bool IsBatchScalarHelper_v<UnaryOperator<OperTag, DataType>> =
    std::is_same_v<OperaCateCal_t<OperTag, DataType>, CategoryTags::BatchScalar>;
template <OperTagConcept OperTag, DataConcept DataType>
constexpr bool IsBatchMatrixHelper_v<UnaryOperator<OperTag, DataType>> =
    std::is_same_v<OperaCateCal_t<OperTag, DataType>, CategoryTags::BatchMatrix>;

template <OperTagConcept OperTag, DataConcept DataType>
class UnaryOperator : public OperOrganizer<OperTag, OperaCateCal_t<OperTag, DataType>> {
public:
    using ElementType = OperElementType_t<OperTag, DataType>;
    using DeviceType = OperDeviceType_t<OperTag, DataType>;

    UnaryOperator(DataType data)
        : OperOrganizer<OperTag, OperaCateCal_t<OperTag, DataType>>(data)
        , m_data(std::move(data)) {}

    [[nodiscard]] const DataType& operand() const { return m_data; }

    // TODO: for evaluation
    auto evalRegister() const {
        if (!m_evalBuf.isEvaluated()) {
            using OperSeqCont = typename OperSeq<OperTag>::type;
            using Head = ContainerHead_t<OperSeqCont>;
            using Tail = PopFrontFromContainer_t<OperSeqCont>;
            Head::template evalRegister<Tail>(m_evalBuf, m_data);
        }
        return m_evalBuf.constHandle();
    }

    bool operator==(const UnaryOperator& val) const { return m_data == val.m_data; }

    template <typename OtherData>
    bool operator==(const OtherData& val) const {
        return false;
    }

    template <typename OtherData>
    bool operator!=(const OtherData& val) const {
        return !(operator==(val));
    }

private:
    DataType m_data;
    using Cate = OperaCateCal_t<OperTag, DataType>;
    using Principal = PrincipleDataType_t<Cate, ElementType, DeviceType>;
    EvalBuffer<Principal> m_evalBuf;
};

/******************************* transpose operator *******************************/
template <>
class OperOrganizer<UnaryOperTags::Transpose, CategoryTags::Matrix> {
public:
    template <DataConcept DataType>
    explicit OperOrganizer(const DataType& data) : m_rowNum(data.colNum())
                                                 , m_colNum(data.rowNum()) {}

    [[nodiscard]] std::size_t rowNum() const { return m_rowNum; }

    [[nodiscard]] std::size_t colNum() const { return m_colNum; }

private:
    std::size_t m_rowNum;
    std::size_t m_colNum;
};

template <>
class OperOrganizer<UnaryOperTags::Transpose, CategoryTags::BatchMatrix> {
public:
    template <DataConcept DataType>
    explicit OperOrganizer(const DataType& data)
        : m_rowNum(data.colNum())
        , m_colNum(data.rowNum())
        , m_batchNum(data.batchNum()) {}

    [[nodiscard]] std::size_t rowNum() const { return m_rowNum; }

    [[nodiscard]] std::size_t colNum() const { return m_colNum; }

    [[nodiscard]] std::size_t batchNum() const { return m_batchNum; }

private:
    std::size_t m_rowNum;
    std::size_t m_colNum;
    std::size_t m_batchNum;
};

namespace details {
template <MatrixConcept MatType>
struct OperatorTranspose {
    static auto eval(MatType&& mat) {
        using rawType = std::remove_cvref_t<MatType>;
        using ResType = UnaryOperator<UnaryOperTags::Transpose, rawType>;
        return ResType{std::forward<MatType>(mat)};
    }
};
}  // namespace details

template <MatrixConcept MatType>
auto transpose(MatType&& mat) {
    return details::OperatorTranspose<MatType>::eval(std::forward<MatType>(mat));
}

namespace eval {
template <UnaryOperConcept OperTag>
struct UnaryCalculator {
    template <typename CaseTail, typename EvalRes, typename Operand>
    static void evalRegister(EvalRes& evalRes, const Operand& oper) {
        static_assert(std::is_same_v<CaseTail, OperSeqContainer<>>, "General Case is not the last one");
        using ElementType = typename EvalRes::DataType::ElementType;
        using DeviceType = typename EvalRes::DataType::DeviceType;
        using CateType = DataCategory_t<typename EvalRes::DataType>;

        auto handle = oper.evalRegister();
        using UnitType =
            typename UnitWrapper<OperTag>::template EvalUnit<decltype(handle), ElementType, DeviceType, CateType>;
        using GroupType = TrivialEvalGroup<UnitType>;

        auto outHandle = evalRes.handle();
        const void* dataPtr = outHandle.dataPtr();
        const void* depVec = handle.dataPtr();
        UnitType unit{std::move(handle), std::move(outHandle)};
        EvalPlan<DeviceType>::template registerFun<GroupType>(std::move(unit), dataPtr, {depVec});
    }
};

template <>
struct UnitWrapper<UnaryOperTags::Transpose> {
    template <typename OperHandle, typename Elem, DeviceConcept Device, CategoryConcept Cate>
    class EvalUnit;

    template <typename OperHandle, typename Element>
    class EvalUnit<OperHandle, Element, CPU, CategoryTags::Matrix> : public BaseEvalUnit<CPU> {
    public:
        using ElementType = Element;
        using DeviceType = CPU;

        EvalUnit(OperHandle oper, EvalHandle<Matrix<ElementType, DeviceType>> evalOutput)
            : m_oper(std::move(oper))
            , m_evalOutput(evalOutput) {}

        void eval() override {
            const auto& pV = m_oper.data();
            const std::size_t rowNum = pV.rowNum();
            const std::size_t colNum = pV.colNum();

            m_evalOutput.allocate(colNum, rowNum);
            auto& res = m_evalOutput.mutableData();

            auto memV1 = lower_access(pV);
            const std::size_t src1PackNum = memV1.rowLen();
            const ElementType* r1 = memV1.rawMemory();

            auto memRes = lower_access(res);
            const std::size_t resPackNum = memRes.rowLen();
            ElementType* r = memRes.mutableRawMemory();

            for (std::size_t i = 0; i < rowNum; ++i) {
                for (std::size_t j = 0; j < colNum; ++j) {
                    r[j * resPackNum + i] = r1[j];
                }
                r1 += src1PackNum;
            }
            m_evalOutput.setEval();
        }

    private:
        OperHandle m_oper;
        EvalHandle<Matrix<ElementType, DeviceType>> m_evalOutput;
    };

    template <typename OperHandle, typename Element>
    class EvalUnit<OperHandle, Element, CPU, CategoryTags::BatchMatrix> : public BaseEvalUnit<CPU> {
    public:
        using ElementType = Element;
        using DeviceType = CPU;

        EvalUnit(OperHandle oper, EvalHandle<Batch<ElementType, DeviceType, CategoryTags::Matrix>> evalOutput)
            : m_oper(std::move(oper))
            , m_evalOutput(evalOutput) {}

        void eval() override {
            const auto& pV = m_oper.data();
            const std::size_t rowNum = pV.rowNum();
            const std::size_t colNum = pV.colNum();
            const std::size_t batchNum = pV.batchNum();

            m_evalOutput.allocate(batchNum, colNum, rowNum);
            auto& res = m_evalOutput.mutableData();

            for (std::size_t batch = 0; batch < batchNum; ++batch) {
                auto memV1 = lower_access(pV[batch]);
                const std::size_t src1PackNum = memV1.rowLen();
                const ElementType* r1 = memV1.rawMemory();

                auto memRes = lower_access(res[batch]);
                const std::size_t resPackNum = memRes.rowLen();
                ElementType* r = memRes.mutableRawMemory();

                for (std::size_t i = 0; i < rowNum; ++i) {
                    for (std::size_t j = 0; j < colNum; ++j) {
                        r[j * resPackNum + i] = r1[j];
                    }
                    r1 += src1PackNum;
                }
            }
            m_evalOutput.setEval();
        }

    private:
        OperHandle m_oper;
        EvalHandle<Batch<ElementType, DeviceType, CategoryTags::Matrix>> m_evalOutput;
    };
};
}  // namespace eval

template <>
struct OperSeq<UnaryOperTags::Transpose> {
    using type = OperSeqContainer<eval::UnaryCalculator<UnaryOperTags::Transpose>>;
};

/********************************** Collapse Operator ********************************************/
template <>
struct OperCategory<UnaryOperTags::Collapse, CategoryTags::BatchMatrix>  // change the default behavior
{
    using type = CategoryTags::Matrix;
};

namespace details {
template <typename BatchMat>
    requires IsBatchMatrix_v<BatchMat>
struct OperatorCollapse {
private:
    using RawType = std::remove_cvref_t<BatchMat>;

public:
    static auto eval(BatchMat&& mat) {
        using ResType = UnaryOperator<UnaryOperTags::Collapse, RawType>;
        return ResType{std::forward<BatchMat>(mat)};
    }
};
}  // namespace details

template <typename BatchMat>
    requires IsBatchMatrix_v<BatchMat>
auto collapse(BatchMat&& mat) {
    return details::OperatorCollapse<BatchMat>::eval(std::forward<BatchMat>(mat));
}

template <>
struct eval::UnitWrapper<UnaryOperTags::Collapse> {
    template <typename OperHandle, typename Elem, DeviceConcept Device>
    class EvalUnit;

    template <typename OperHandle, typename Element>
    class EvalUnit<OperHandle, Element, CPU> : public BaseEvalUnit<CPU> {
    public:
        using ElementType = Element;
        using DeviceType = CPU;

        EvalUnit(OperHandle oper, EvalHandle<Matrix<ElementType, DeviceType>> evalOutput)
            : m_oper(std::move(oper))
            , m_evalOutput(std::move(evalOutput)) {}

        void eval() override {
            const auto& pV = m_oper.data();
            const std::size_t rowNum = pV.rowNum();
            const std::size_t colNum = pV.colNum();
            const std::size_t batchNum = pV.batchNum();
            m_evalOutput.allocate(rowNum, colNum);
            auto& res = m_evalOutput.mutableData();

            for (std::size_t i = 0; i < rowNum; ++i) {
                for (std::size_t j = 0; j < colNum; ++j) {
                    ElementType tmp{};
                    for (std::size_t k = 0; k < batchNum; ++k) {
                        tmp += pV[k](i, j);
                    }
                    res.setValue(i, j, tmp);
                }
            }
            m_evalOutput.setEval();
        }

    private:
        OperHandle m_oper;
        EvalHandle<Matrix<ElementType, DeviceType>> m_evalOutput;
    };
};

template <>
struct OperSeq<UnaryOperTags::Collapse> {
    using type = OperSeqContainer<eval::UnaryCalculator<UnaryOperTags::Collapse>>;
};

namespace eval {
template <>
struct UnaryCalculator<UnaryOperTags::Collapse> {
    template <typename CaseTail, typename EvalRes, typename Operand>
    static void evalRegister(EvalRes& evalRes, const Operand& oper) {
        static_assert(std::is_same_v<CaseTail, OperSeqContainer<>>, "General Case is not the last one");
        using ElementType = typename EvalRes::DataType::ElementType;
        using DeviceType = typename EvalRes::DataType::DeviceType;
        using CateType = DataCategory_t<typename EvalRes::DataType>;
        using OperTag = typename UnaryOperTags::Collapse;

        auto handle = oper.evalRegister();
        using UnitType = typename UnitWrapper<OperTag>::template EvalUnit<decltype(handle), ElementType, DeviceType>;
        using GroupType = TrivialEvalGroup<UnitType>;

        auto outHandle = evalRes.handle();
        const void* dataPtr = outHandle.dataPtr();
        const void* depVec = handle.dataPtr();
        UnitType unit{std::move(handle), std::move(outHandle)};
        EvalPlan<DeviceType>::template registerFun<GroupType>(std::move(unit), dataPtr, {depVec});
    }
};
}  // namespace eval

namespace details {
template <UnaryOperConcept UnaryOperTag, MatrixConcept MatType>
struct UnaryOperatorForMatrix {
private:
    using RawType = std::remove_cvref_t<MatType>;

public:
    static auto eval(MatType&& mat) {
        using ResType = UnaryOperator<UnaryOperTag, RawType>;
        return ResType{std::forward<MatType>(mat)};
    }
};
}  // namespace details

/********************************** Abs Operator ********************************************/
// namespace details
// {
//     template <MatrixConcept MatType>
//     struct OperatorAbs
//     {
//     private:
//         using RawType = std::remove_cvref_t<MatType>;
//
//     public:
//         static auto eval(MatType&& mat)
//         {
//             using ResType = UnaryOperator<UnaryOperTags::Abs, RawType>;
//             return ResType{std::forward<MatType>(mat)};
//         }
//     };
// }
namespace eval {
template <typename Operator>
struct TrivialUnaryOperatorTag {
    using Oper = Operator;
};

template <typename Operator>
struct UnitWrapper<TrivialUnaryOperatorTag<Operator>> {
    template <typename OperHandle, typename Element, DeviceConcept Device, CategoryConcept Category>
    class EvalUnit;

    template <typename OperHandle, typename Element>
    class EvalUnit<OperHandle, Element, CPU, CategoryTags::Matrix> : public BaseEvalUnit<CPU> {
    public:
        using ElementType = Element;
        using DeviceType = CPU;

        // EvalHandle
        EvalUnit(OperHandle oper, EvalHandle<Matrix<ElementType, DeviceType>> evalOutput)
            : m_oper(std::move(oper))
            , m_evalOutput(evalOutput) {}

        void eval() override {
            const auto& pV = m_oper.data();
            const std::size_t rowNum = pV.rowNum();
            const std::size_t colNum = pV.colNum();

            m_evalOutput.allocate(rowNum, colNum);
            auto& res = m_evalOutput.mutableData();

            auto memV1 = lower_access(pV);
            auto memRes = lower_access(res);

            const std::size_t src1PackNum = memV1.rowLen();
            const std::size_t tgtPackNum = memRes.rowLen();

            const ElementType* r1 = memV1.rawMemory();
            ElementType* r = memRes.mutableRawMemory();
            auto oper = Operator{};
            for (std::size_t i = 0; i < rowNum; ++i) {
                for (std::size_t j = 0; j < colNum; ++j) {
                    r[j] = oper(r1[j]);
                }
                r1 += src1PackNum;
                r += tgtPackNum;
            }
            m_evalOutput.setEval();
        }

    private:
        OperHandle m_oper;
        EvalHandle<Matrix<ElementType, DeviceType>> m_evalOutput;
    };

    template <typename OperHandle, typename Element>
    class EvalUnit<OperHandle, Element, CPU, CategoryTags::BatchMatrix> : public BaseEvalUnit<CPU> {
    public:
        using ElementType = Element;
        using DeviceType = CPU;

        // EvalHandle
        EvalUnit(OperHandle oper, EvalHandle<Batch<ElementType, DeviceType, CategoryTags::Matrix>> evalOutput)
            : m_oper(std::move(oper))
            , m_evalOutput(evalOutput) {}

        void eval() override {
            const auto& pV = m_oper.data();
            const std::size_t rowNum = pV.rowNum();
            const std::size_t colNum = pV.colNum();
            const std::size_t batchNum = pV.batchNum();

            m_evalOutput.allocate(batchNum, rowNum, colNum);
            auto& res = m_evalOutput.mutableData();

            for (std::size_t batch = 0; batch < batchNum; ++batch) {
                auto memV1 = lower_access(pV[batch]);
                auto memRes = lower_access(res[batch]);

                const std::size_t src1PackNum = memV1.rowLen();
                const std::size_t tgtPackNum = memRes.rowLen();

                const ElementType* r1 = memV1.rawMemory();
                ElementType* r = memRes.mutableRawMemory();
                auto oper = Operator{};
                for (std::size_t i = 0; i < rowNum; ++i) {
                    for (std::size_t j = 0; j < colNum; ++j) {
                        r[j] = oper(r1[j]);
                    }
                    r1 += src1PackNum;
                    r += tgtPackNum;
                }
            }
            m_evalOutput.setEval();
        }

    private:
        OperHandle m_oper;
        EvalHandle<Batch<ElementType, DeviceType, CategoryTags::Matrix>> m_evalOutput;
    };
};
}  // namespace eval

template <MatrixConcept MatType>
auto abs(MatType&& mat) {
    // return details:<MatType>::eval(std::forward<MatType>(mat));
    return details::UnaryOperatorForMatrix<UnaryOperTags::Abs, MatType>::eval(std::forward<MatType>(mat));
}

template <>
struct OperSeq<UnaryOperTags::Abs> {
    using type = OperSeqContainer<
        eval::UnaryCalculator<eval::TrivialUnaryOperatorTag<decltype([](auto x) { return std::abs(x); })>>>;
};

/********************************** Sign Operator ********************************************/
// namespace details
// {
//     template <MatrixConcept MatType>
//     struct OperatorSign
//     {
//     private:
//         using RawType = std::remove_cvref_t<MatType>;
//
//     public:
//         static auto eval(MatType&& mat)
//         {
//             using ResType = UnaryOperator<UnaryOperTags::Sign, RawType>;
//             return ResType{std::forward<MatType>(mat)};
//         }
//     };
// }

template <MatrixConcept MatType>
auto sign(MatType&& mat) {
    return details::UnaryOperatorForMatrix<UnaryOperTags::Sign, MatType>::eval(std::forward<MatType>(mat));
}

template <>
struct OperSeq<UnaryOperTags::Sign> {
    using type = OperSeqContainer<eval::UnaryCalculator<eval::TrivialUnaryOperatorTag<decltype([]<typename T0>(T0&& x) {
        return (x > 0) ? static_cast<std::remove_reference_t<T0>>(1) : static_cast<std::remove_reference_t<T0>>(-1);
    })>>>;
};

/********************************** Sigmoid Operator ********************************************/
// namespace details
// {
//     template <MatrixConcept MatType>
//     struct OperatorSigmoid
//     {
//         static auto eval(MatType&& mat)
//         {
//             using rawType = std::remove_cvref_t<MatType>;
//             using ResType = UnaryOperator<UnaryOperTags::Sigmoid, rawType>;
//             return ResType{std::forward<MatType>(mat)};
//         }
//     };
// } // namespace details

template <MatrixConcept MatType>
auto sigmoid(MatType&& mat) {
    return details::UnaryOperatorForMatrix<UnaryOperTags::Sigmoid, MatType>::eval(std::forward<MatType>(mat));
}

template <>
struct OperSeq<UnaryOperTags::Sigmoid> {
    using type = OperSeqContainer<eval::UnaryCalculator<eval::TrivialUnaryOperatorTag<decltype([]<typename T0>(T0&& x) {
        return static_cast<std::remove_reference_t<T0>>(1 / (1 + exp(-x)));
    })>>>;
};

/********************************** Tanh Operator ********************************************/
// namespace details
// {
//     template <MatrixConcept MatType>
//     struct OperatorTanh
//     {
//     private:
//         using RawType = std::remove_cvref_t<MatType>;
//
//     public:
//         static auto eval(MatType&& mat)
//         {
//             using ResType = UnaryOperator<UnaryOperTags::Tanh, RawType>;
//             return ResType{std::forward<MatType>(mat)};
//         }
//     };
// }

template <MatrixConcept MatType>
auto tanh(MatType&& mat) {
    return details::UnaryOperatorForMatrix<UnaryOperTags::Tanh, MatType>::eval(std::forward<MatType>(mat));
}

template <>
struct OperSeq<UnaryOperTags::Tanh> {
    using type = OperSeqContainer<eval::UnaryCalculator<
        eval::TrivialUnaryOperatorTag<decltype([]<typename T0>(T0&& x) { return std::tanh(x); })>>>;
};

/********************************** ReLU Operator ********************************************/
template <MatrixConcept MatType>
auto relu(MatType&& mat) {
    return details::UnaryOperatorForMatrix<UnaryOperTags::ReLU, MatType>::eval(std::forward<MatType>(mat));
}

template <>
struct OperSeq<UnaryOperTags::ReLU> {
    using type = OperSeqContainer<eval::UnaryCalculator<eval::TrivialUnaryOperatorTag<decltype([]<typename T0>(T0&& x) {
        return std::max(x, static_cast<std::remove_reference_t<T0>>(0));
    })>>>;
};

/********************************** VecSoftmax Operator ********************************************/
template <MatrixConcept MatType>
auto vec_softmax(MatType&& mat) {
    return details::UnaryOperatorForMatrix<UnaryOperTags::VecSoftmax, MatType>::eval(std::forward<MatType>(mat));
}

template <>
struct OperSeq<UnaryOperTags::VecSoftmax> {
    using type = OperSeqContainer<eval::UnaryCalculator<UnaryOperTags::VecSoftmax>>;
};

template <>
struct eval::UnitWrapper<UnaryOperTags::VecSoftmax> {
    template <typename OperHandle, typename Element, DeviceConcept Device, CategoryConcept Cate>
    class EvalUnit;

    template <typename OperHandle, typename Element>
    class EvalUnit<OperHandle, Element, CPU, CategoryTags::Matrix> : public BaseEvalUnit<CPU> {
    public:
        using ElementType = Element;
        using DeviceType = CPU;

        EvalUnit(OperHandle oper, EvalHandle<Matrix<ElementType, DeviceType>> evalOutput)
            : m_oper(std::move(oper))
            , m_evalOutput(evalOutput) {}

        void eval() override {
            const auto& pV = m_oper.data();
            assert(pV.rowNum() == 1);
            const std::size_t colNum = pV.colNum();

            m_evalOutput.allocate(1, colNum);
            if (colNum == 0) {
                return;
            }
            auto& res = m_evalOutput.mutableData();
            auto memV1 = lower_access(pV);
            auto memRes = lower_access(res);

            const ElementType* r1 = memV1.rawMemory();
            ElementType* r = memRes.mutableRawMemory();

            auto maxElem = *std::max_element(r1, r1 + colNum);
            ElementType sum{};

            for (std::size_t i = 0; i < colNum; ++i) {
                r[i] = std::exp(r1[i] - maxElem);
                sum += r[i];
            }
            for (std::size_t i = 0; i < colNum; ++i) {
                r[i] /= sum;
            }
            m_evalOutput.setEval();
        }

    private:
        OperHandle m_oper;
        EvalHandle<Matrix<ElementType, DeviceType>> m_evalOutput;
    };

    template <typename OperHandle, typename Element>
    class EvalUnit<OperHandle, Element, CPU, CategoryTags::BatchMatrix> : public BaseEvalUnit<CPU> {
    public:
        using ElementType = Element;
        using DeviceType = CPU;

        EvalUnit(OperHandle oper, EvalHandle<Batch<ElementType, DeviceType, CategoryTags::Matrix>> evalOutput)
            : m_oper(std::move(oper))
            , m_evalOutput(evalOutput) {}

        void eval() override {
            const auto& pV = m_oper.data();
            assert(pV.rowNum() == 1);
            const std::size_t colNum = pV.colNum();
            const std::size_t batchNum = pV.batchNum();

            m_evalOutput.allocate(batchNum, 1, colNum);
            if (colNum == 0) {
                return;
            }
            auto& res = m_evalOutput.mutableData();
            for (std::size_t batch = 0; batch < batchNum; ++batch) {
                auto memV1 = lower_access(pV[batch]);
                auto memRes = lower_access(res[batch]);

                const ElementType* r1 = memV1.rawMemory();
                ElementType* r = memRes.mutableRawMemory();

                auto maxElem = *std::max_element(r1, r1 + colNum);
                ElementType sum{};

                for (std::size_t i = 0; i < colNum; ++i) {
                    r[i] = std::exp(r1[i] - maxElem);
                    sum += r[i];
                }
                for (std::size_t i = 0; i < colNum; ++i) {
                    r[i] /= sum;
                }
            }
            m_evalOutput.setEval();
        }

    private:
        OperHandle m_oper;
        EvalHandle<Batch<ElementType, DeviceType, CategoryTags::Matrix>> m_evalOutput;
    };
};

}  // namespace metann

#endif  // UNARY_OPERATORS_HPP
