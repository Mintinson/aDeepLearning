//
// Created by asus on 2025/1/10.
//
//
/**
 * @brief When a non-batch data type and batch data type to do the calculation,
 * `Duplicate` can convert the non-batch data type to batch data type,
 * while retaining its batch data within the equal information so that compiler can be easy to optimize.
 */
#ifndef DUPLICATE_HPP
#define DUPLICATE_HPP

#include "allocator.hpp"
#include "batch.hpp"
#include "data_category.hpp"

namespace metann
{
    template <DataConcept DataType, CategoryConcept DataCate>
    class DuplicateImpl;

    template <DataConcept DataType>
    class Duplicate : public DuplicateImpl<DataType, DataCategory_t<DataType>>
    {
    public:
        using ElementType = typename DataType::ElementType;
        using DeviceType = typename DataType::DeviceType;
        using DuplicateImpl<DataType, DataCategory_t<DataType>>::DuplicateImpl;
    };

    namespace eval
    {
        struct DupScalarEvalTag;

        template <>
        struct UnitWrapper<DupScalarEvalTag>
        {
            template <typename InHandle, typename ElementType, DeviceConcept Device, CategoryConcept DataCate>
            struct EvalUnit;

            template <typename InHandle, typename Element>
            struct EvalUnit<InHandle, Element, CPU, CategoryTags::Scalar>
                : public BaseEvalUnit<CPU>
            {
            public:
                EvalUnit(InHandle oper, size_t batchNum,
                         EvalHandle<Batch<Element, CPU, CategoryTags::Scalar>> evalOutput)
                    : m_oper(std::move(oper))
                      , m_batchNum(batchNum)
                      , m_evalOutput(std::move(evalOutput))
                {
                }

                void eval() override
                {
                    const auto& p_v1 = m_oper.data();

                    m_evalOutput.allocate(m_batchNum);
                    auto& res = m_evalOutput.mutableData();

                    for (size_t i = 0; i < m_batchNum; ++i)
                    {
                        res.setValue(i, p_v1.value());
                    }
                    m_evalOutput.setEval();
                }

            private:
                InHandle m_oper;
                size_t m_batchNum;
                EvalHandle<Batch<Element, CPU, CategoryTags::Scalar>> m_evalOutput;
            };
        };

        struct DupMatrixEvalTag;

        template <>
        struct UnitWrapper<DupMatrixEvalTag>
        {
            template <typename InHandle, typename ElementType, DeviceConcept Device, CategoryConcept DataCate>
            struct EvalUnit;

            template <typename InHandle, typename Element>
            struct EvalUnit<InHandle, Element, CPU, CategoryTags::Matrix>
                : public BaseEvalUnit<CPU>
            {
            public:
                EvalUnit(InHandle oper, size_t batchNum,
                         EvalHandle<Batch<Element, CPU, CategoryTags::Matrix>> evalOutput)
                    : m_oper(std::move(oper))
                      , m_batchNum(batchNum)
                      , m_evalOutput(std::move(evalOutput))
                {
                }

                void eval() override
                {
                    const auto& p_v1 = m_oper.data();
                    const std::size_t rowNum = p_v1.rowNum();
                    const std::size_t colNum = p_v1.colNum();

                    m_evalOutput.allocate(m_batchNum, rowNum, colNum);
                    auto& res = m_evalOutput.mutableData();

                    for (size_t i = 0; i < m_batchNum; ++i)
                    {
                        for (size_t j = 0; j < rowNum; ++j)
                        {
                            for (size_t k = 0; k < colNum; ++k)
                            {
                                res.setValue(i, j, k, p_v1(j, k));
                            }
                        }
                    }
                    m_evalOutput.setEval();
                }

            private:
                InHandle m_oper;
                size_t m_batchNum;
                EvalHandle<Batch<Element, CPU, CategoryTags::Matrix>> m_evalOutput;
            };
        };
    }

    template <DataConcept DataType>
    class DuplicateImpl<DataType, CategoryTags::Scalar>
    {
    public:
        using ElementType = typename DataType::ElementType;
        using DeviceType = typename DataType::DeviceType;

        DuplicateImpl(DataType data, const std::size_t batchNum)
            : m_data(std::move(data))
              , m_batchNum(batchNum)
        {
        }

        std::size_t batchNum() const { return m_batchNum; }
        std::size_t size() const { return m_batchNum; }

        const DataType& element() const { return m_data; }

        bool operator==(const Duplicate<DataType>& rhs) const
        {
            const DuplicateImpl& tmp = static_cast<const DuplicateImpl&>(rhs);
            return (tmp.m_data == m_data) && (tmp.m_batchNum == m_batchNum);
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
            if (!m_evalBuf.isEvaluated())
            {
                auto inHandle = m_data.evalRegister();
                auto outHandle = m_evalBuf.handle();

                using EvalUnit = eval::UnitWrapper<eval::DupScalarEvalTag>::EvalUnit<decltype(inHandle),
                    ElementType, DeviceType, CategoryTags::Scalar>;
                using GroupType = TrivialEvalGroup<EvalUnit>;

                const void* dataPtr = outHandle.dataPtr();
                const void* depPtr = inHandle.dataPtr();
                EvalUnit unit(std::move(inHandle), m_batchNum, std::move(outHandle));
                EvalPlan<DeviceType>::template registerFun<GroupType>
                    (std::move(unit), dataPtr, {depPtr});
            }
            return m_evalBuf.constHandle();
        }

    private:
        DataType m_data;
        std::size_t m_batchNum;
        EvalBuffer<Batch<ElementType, DeviceType, CategoryTags::Scalar>> m_evalBuf;
    };

    template <DataConcept DataType>
    class DuplicateImpl<DataType, CategoryTags::Matrix>
    {
    public:
        using ElementType = typename DataType::ElementType;
        using DeviceType = typename DataType::DeviceType;

        DuplicateImpl(DataType data, const std::size_t batchNum)
            : m_data(std::move(data))
              , m_batchNum(batchNum)
        {
        }

        std::size_t rowNum() const { return m_data.rowNum(); }
        std::size_t colNum() const { return m_data.colNum(); }
        std::size_t batchNum() const { return m_batchNum; }
        const DataType& element() const { return m_data; }

        bool operator==(const Duplicate<DataType>& rhs) const
        {
            const DuplicateImpl& tmp = static_cast<const DuplicateImpl&>(rhs);
            return (tmp.m_data == m_data) && (tmp.m_batchNum == m_batchNum);
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
            if (!m_evalBuf.isEvaluated())
            {
                auto inHandle = m_data.evalRegister();
                auto outHandle = m_evalBuf.handle();

                using EvalUnit = eval::UnitWrapper<eval::DupMatrixEvalTag>::EvalUnit<decltype(inHandle),
                    ElementType, DeviceType, CategoryTags::Matrix>;
                using GroupType = TrivialEvalGroup<EvalUnit>;

                const void* dataPtr = outHandle.dataPtr();
                const void* depPtr = inHandle.dataPtr();
                EvalUnit unit(std::move(inHandle), m_batchNum, std::move(outHandle));
                EvalPlan<DeviceType>::template registerFun<GroupType>
                    (std::move(unit), dataPtr, {depPtr});
            }
            return m_evalBuf.constHandle();
        }

    private:
        DataType m_data;
        std::size_t m_batchNum;
        EvalBuffer<Batch<ElementType, DeviceType, CategoryTags::Matrix>> m_evalBuf;
    };

    // template <DataConcept DataType>
    // class Duplicate : public DuplicateImpl<DataType, DataCategory_t<DataType>>
    // {
    // public:
    //     using ElementType = typename DataType::ElementType;
    //     using DeviceType = typename DataType::DeviceType;
    //
    //     using DuplicateImpl<DataType, DataCategory_t<DataType>>::DuplicateImpl;
    // };

    template <DataConcept DataType>
    constexpr bool IsBatchMatrixHelper_v<Duplicate<DataType>> = IsMatrixHelper_v<DataType>;
    template <DataConcept DataType>
    constexpr bool IsBatchScalarHelper_v<Duplicate<DataType>> = IsScalarHelper_v<DataType>;

    template <DataConcept DataType>
    auto make_duplicate(std::size_t batchNum, DataType&& data)
    {
        using RawDataType = std::remove_cvref_t<DataType>;
        return Duplicate<RawDataType>{std::move(data), batchNum};
    }

    template <DataConcept DataType, typename... Args>
    auto make_duplicate(std::size_t batchNum, Args&&... args)
    {
        using RawDataType = std::remove_cvref_t<DataType>;
        RawDataType data{std::forward<Args>(args)...};
        return Duplicate<RawDataType>{std::move(data), batchNum};
    }
}

#endif // DUPLICATE_HPP
