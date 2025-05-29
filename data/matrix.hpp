//
// Created by asus on 2025/1/8.
//

#ifndef MATRIX_HPP
#define MATRIX_HPP
#include <cassert>
#include <string.h>

#include "data_category.hpp"
#include "data_device.hpp"
#include "allocator.hpp"
#include "scalar.hpp"
#include "../eval/extended.hpp"
#include "../eval/facilities.hpp"


namespace metann
{
    template <typename Element, DeviceConcept Device = CPU>
    struct Matrix;

    template <typename Element, DeviceConcept Device>
    constexpr bool IsMatrixHelper_v<Matrix<Element, Device>> = true;

    template <typename Element, DeviceConcept Device>
    struct PrincipleDataType<CategoryTags::Matrix, Element, Device>
    {
        using type = Matrix<Element, Device>;
    };

    template <typename Element>
    struct Matrix<Element, CPU> // specialization for CPU scalar
    {
        using DeviceType = CPU;
        using ElementType = Element;

        explicit Matrix(const std::size_t rows = 0, const std::size_t cols = 0)
            : m_mem{rows * cols}, m_rowNum(rows), m_colNum(cols),
              m_rowLen{cols}
        {
        }

        // rule of five
        Matrix(const Matrix&) = default;
        Matrix(Matrix&&) noexcept = default;
        Matrix& operator=(const Matrix&) = default;
        Matrix& operator=(Matrix&&) noexcept = default;
        ~Matrix() = default;

        [[nodiscard]] std::size_t rowNum() const
        {
            return m_rowNum;
        }

        [[nodiscard]] std::size_t colNum() const
        {
            return m_colNum;
        }
        [[nodiscard]] std::size_t rowLen() const
        {
            return m_rowLen;
        }

        void setValue(const std::size_t rowId, const std::size_t colId, ElementType value)
        {
            assert(availableForWrite());
            (m_mem.rawMemory())[rowId * m_rowLen + colId] = value;
        }

        [[nodiscard]] auto operator()(std::size_t rowId, std::size_t colId) const
        {
            return (m_mem.rawMemory())[rowId * m_rowLen + colId];
        }

        bool availableForWrite() const
        {
            return m_mem.useCount() == 1;
        }

        // private:
        Matrix(std::shared_ptr<ElementType> mem, ElementType* memStart,
               const std::size_t rowNum, const std::size_t colNum, const std::size_t rowLen)
            : m_mem{mem, memStart}, m_rowNum{rowNum}, m_colNum{colNum},
              m_rowLen{rowLen}
        {
        }

    public:
        [[nodiscard]] Matrix subMatrix(const std::size_t rowBeg, const std::size_t rowEnd,
                                       const std::size_t colBeg, const std::size_t colEnd) const
        {
            ElementType* pos = m_mem.rawMemory() + rowBeg * m_rowLen + colBeg;
            return Matrix{m_mem.sharedPtr(), pos, rowEnd - rowBeg, colEnd - colBeg, m_rowLen};
        }


        bool operator==(const Matrix& rhs) const = default;

        template <typename OtherType>
        bool operator==(const OtherType& rhs) const
        {
            return false;
        }

        template <typename OtherType>
        bool operator!=(const OtherType& rhs) const
        {
            return !(this->operator==(rhs));
        }

        auto evalRegister() const
        {
            return make_const_eval_handle(*this);
        }

        friend struct LowerAccessImpl<Matrix<Element, CPU>>;

    private:
        ContinuousMemory<ElementType, DeviceType> m_mem;
        std::size_t m_rowNum;
        std::size_t m_colNum;
        std::size_t m_rowLen;
    };


    template <typename Element>
    struct LowerAccessImpl<Matrix<Element, CPU>>
    {
        explicit LowerAccessImpl(Matrix<Element, CPU> mat): m_mat(mat)
        {
        }

        auto mutableRawMemory() { return m_mat.m_mem.rawMemory(); }
        const auto rawMemory() const { return m_mat.m_mem.rawMemory(); }
        size_t rowLen() const { return m_mat.m_rowLen; }

    private:
        Matrix<Element, CPU> m_mat;
    };

    /// Trivial Matrix
    template <typename Element, DeviceConcept Device, typename Scalar>
    class TrivialMatrix;

    namespace eval
    {
        struct TrivialMatrixEvalTag;

        template <>
        struct UnitWrapper<TrivialMatrixEvalTag>
        {
            template <typename Element, DeviceConcept Device>
            class EvalUnit : public BaseEvalUnit<Device>
            {
            public:
                using DeviceType = Device;

                template <typename ScalarElementType>
                EvalUnit(EvalHandle<Matrix<Element, DeviceType>> resBuf,
                         const std::size_t rowNum, const std::size_t colNum,
                         const Scalar<ScalarElementType, DeviceType>& val)
                    : BaseEvalUnit<Device>(),
                      m_resHandle(std::move(resBuf)),
                      m_rowNum(rowNum), m_colNum(colNum),
                      m_value(val.value())

                {
                }

                void eval() override
                {
                    m_resHandle.allocate(m_rowNum, m_colNum);
                    auto& mutableData = m_resHandle.mutableData();

                    auto lowLayer = lower_access(mutableData);
                    const std::size_t rowLen = lowLayer.rowLen();
                    auto mem = lowLayer.mutableRawMemory();
                    for (std::size_t i = 0; i != m_rowNum; ++i)
                    {
                        for (std::size_t j = 0; j != m_colNum; ++j)
                        {
                            mem[j] = m_value;
                        }
                        mem += rowLen;
                    }
                    m_resHandle.setEval();
                }

            private:
                EvalHandle<Matrix<Element, DeviceType>> m_resHandle;
                std::size_t m_rowNum;
                std::size_t m_colNum;
                Element m_value;
            };
        };
    }

    template <typename Element, DeviceConcept Device, typename Scalar>
    class TrivialMatrix
    {
    public:
        using ElementType = Element;
        using DeviceType = Device;

        TrivialMatrix(std::size_t rowNum, std::size_t colNum, Scalar value)
            : m_value{value}, m_rowNum{rowNum}, m_colNum{colNum}
        {
        }

        [[nodiscard]] std::size_t rowNum() const { return m_rowNum; }
        [[nodiscard]] std::size_t colNum() const { return m_colNum; }

        [[nodiscard]] auto elementValue() const { return m_value; }

        // TODO: for evaluation
        bool operator==(const TrivialMatrix& rhs) const
        {
            return m_value == rhs.m_value && m_rowNum == rhs.m_rowNum && m_colNum == rhs.m_colNum;
        }

        template <typename OtherType>
        bool operator==(const OtherType& rhs) const
        {
            return false;
        }

        template <typename OtherType>
        bool operator!=(const OtherType& rhs) const
        {
            return !(*this == rhs);
        }

        auto evalRegister() const
        {
            using EvalUnit = eval::UnitWrapper<eval::TrivialMatrixEvalTag>::EvalUnit<ElementType, DeviceType>;
            using EvalGroup = TrivialEvalGroup<EvalUnit>;
            if (!m_evalBuf.isEvaluated())
            {
                auto evalHandle = m_evalBuf.handle();
                const void* outputPtr = evalHandle.dataPtr();

                EvalUnit unit{std::move(evalHandle), m_rowNum, m_colNum, m_value};
                EvalPlan<DeviceType>::template registerFun<EvalGroup>(
                    std::move(unit), outputPtr, {}
                );
            }
            return m_evalBuf.constHandle();
        }

    private:
        Scalar m_value;
        std::size_t m_rowNum;
        std::size_t m_colNum;

        EvalBuffer<Matrix<ElementType, DeviceType>> m_evalBuf;
    };

    template <typename Element, DeviceConcept Device, typename Scalar>
    constexpr bool IsMatrixHelper_v<TrivialMatrix<Element, Device, Scalar>> = true;


    template <typename Element, DeviceConcept Device, typename ValueType>
    auto make_trivial_matrix(std::size_t rowNum, std::size_t colNum, ValueType&& value)
    {
        using RawValueType = std::remove_cvref_t<ValueType>;
        if constexpr (IsScalar_v<RawValueType>)
        {
            static_assert(std::is_same_v<Device, typename RawValueType::DeviceType> || std::is_same_v<Device, CPU>,
                          "The scalar value should be stored in the same location as matrix!");
            return TrivialMatrix<Element, Device, RawValueType>{rowNum, colNum, std::forward<ValueType>(value)};
        }
        else
        {
            auto tmpElem = static_cast<Element>(value);
            auto scalar = Scalar<Element, CPU>{std::move(tmpElem)};
            return TrivialMatrix<Element, Device, decltype(scalar)>{rowNum, colNum, scalar};
        }
    }

    /// Zero Matrix
    namespace eval
    {
        struct ZeroMatrixEvalTag;

        template <>
        struct UnitWrapper<ZeroMatrixEvalTag>
        {
            template <typename Element, DeviceConcept Device>
            class EvalUnit : public BaseEvalUnit<Device>
            {
            public:
                using DeviceType = Device;

                EvalUnit(EvalHandle<Matrix<Element, DeviceType>> resBuf,
                         const std::size_t rowNum, const std::size_t colNum)
                    : m_resHandle(std::move(resBuf)),
                      m_rowNum(rowNum), m_colNum(colNum)
                {
                }

                void eval() override
                {
                    m_resHandle.allocate(m_rowNum, m_colNum);
                    auto& mutableData = m_resHandle.mutableData();

                    auto lowLayer = lower_access(mutableData);
                    const std::size_t rowLen = lowLayer.rowLen();
                    auto mem = lowLayer.mutableRawMemory();
                    if (rowLen != m_colNum)
                    {
                        throw std::runtime_error("Gap among matrix rows");
                    }
                    memset(mem, 0, sizeof(Element) * m_colNum * m_rowNum);
                    m_resHandle.setEval();
                }

            private:
                EvalHandle<Matrix<Element, DeviceType>> m_resHandle;
                std::size_t m_rowNum;
                std::size_t m_colNum;
            };
        };
    }

    template <typename Element, DeviceConcept Device>
    class ZeroMatrix
    {
    public:
        using ElementType = Element;
        using DeviceType = Device;
        ZeroMatrix() = default;

        ZeroMatrix(const std::size_t rowNum, const std::size_t colNum)
            : m_rowNum{rowNum}, m_colNum{colNum}
        {
        }

        std::size_t rowNum() const { return m_rowNum; }
        std::size_t colNum() const { return m_colNum; }


        // TODO: for evaluation
        bool operator==(const ZeroMatrix& rhs) const
        {
            return m_rowNum == rhs.m_rowNum && m_colNum == rhs.m_colNum;
        }

        template <typename OtherType>
        bool operator==(const OtherType& rhs) const
        {
            return false;
        }

        template <typename OtherType>
        bool operator!=(const OtherType& rhs) const
        {
            return !(*this == rhs);
        }

        auto evalRegister() const
        {
            using EvalUnit = eval::UnitWrapper<eval::ZeroMatrixEvalTag>::EvalUnit<ElementType, DeviceType>;
            using EvalGroup = TrivialEvalGroup<EvalUnit>;
            if (!m_evalBuf.isEvaluated())
            {
                auto evalHandle = m_evalBuf.handle();
                decltype(auto) outPtr = evalHandle.dataPtr();
                EvalUnit unit(std::move(evalHandle), m_rowNum, m_colNum);
                EvalPlan<DeviceType>::template registerFun<EvalGroup>(std::move(unit), outPtr, {});
            }
            return m_evalBuf.constHandle();
        }

    private:
        std::size_t m_rowNum{};
        std::size_t m_colNum{};
        EvalBuffer<Matrix<ElementType, DeviceType>> m_evalBuf;
    };

    template <typename Element, DeviceConcept Device>
    constexpr bool IsMatrixHelper_v<ZeroMatrix<Element, Device>> = true;

    /// One Hot Vector
    namespace eval
    {
        struct OneHotVectorEvalTag;

        template <>
        struct UnitWrapper<OneHotVectorEvalTag>
        {
            template <typename Element, DeviceConcept Device>
            class EvalUnit : public BaseEvalUnit<Device>
            {
            public:
                using ElementType = Element;
                using DeviceType = Device;

                EvalUnit(EvalHandle<Matrix<Element, DeviceType>> resBuf,
                         const std::size_t rowNum,
                         const std::size_t colNum,
                         const std::size_t val)
                    : m_resHandle(std::move(resBuf))
                      , m_rowNum(rowNum)
                      , m_colNum(colNum)
                      , m_val(val)
                {
                    assert(m_val < (m_rowNum * m_colNum));
                }

                void eval() override
                {
                    auto& mutableData = m_resHandle.mutableData();
                    m_resHandle.allocate(m_rowNum, m_colNum);
                    auto lowLayer = lower_access(mutableData);
                    auto mem = lowLayer.mutableRawMemory();
                    memset(mem, 0, sizeof(Element) * m_rowNum * m_colNum);
                    mem[m_val] = 1;
                    m_resHandle.setEval();
                }

            private:
                EvalHandle<Matrix<Element, DeviceType>> m_resHandle;
                std::size_t m_rowNum;
                std::size_t m_colNum;
                std::size_t m_val;
            };
        };
    }

    template <typename Element, DeviceConcept Device>
    class OneHotVector
    {
    public:
        using ElementType = Element;
        using DeviceType = Device;

        OneHotVector(const std::size_t colNum, const std::size_t hotPos)
            : m_colNum{colNum}, m_hotPos(hotPos)
        {
        }

        [[nodiscard]] std::size_t rowNum() const { return 1; }
        [[nodiscard]] std::size_t colNum() const { return m_colNum; }
        [[nodiscard]] std::size_t hotPos() const { return m_hotPos; }


        // TODO: for evaluation
        [[nodiscard]] bool operator==(const OneHotVector& rhs) const
        {
            return m_colNum == rhs.m_colNum && m_hotPos == rhs.m_hotPos;
        }

        template <typename OtherType>
        [[nodiscard]] bool operator==(const OtherType& rhs) const
        {
            return false;
        }

        template <typename OtherType>
        [[nodiscard]] bool operator!=(const OtherType& rhs) const
        {
            return !(*this == rhs);
        }

        auto evalRegister() const
        {
            using EvalUnit = eval::UnitWrapper<eval::OneHotVectorEvalTag>::EvalUnit<ElementType, DeviceType>;
            using EvalGroup = TrivialEvalGroup<EvalUnit>;
            if (!m_evalBuf.isEvaluated())
            {
                auto evalHandle = m_evalBuf.handle();
                decltype(auto) outPtr = evalHandle.dataPtr();
                EvalUnit unit(std::move(evalHandle), 1, m_colNum, m_hotPos);
                EvalPlan<DeviceType>::template registerFun<EvalGroup>(std::move(unit), outPtr, {});
            }
            return m_evalBuf.constHandle();
        }

    private:
        std::size_t m_colNum{};
        std::size_t m_hotPos{};
        EvalBuffer<Matrix<ElementType, DeviceType>> m_evalBuf;
    };

    template <typename Element, DeviceConcept Device>
    constexpr bool IsMatrixHelper_v<OneHotVector<Element, Device>> = true;
}


#endif //MATRIX_HPP
