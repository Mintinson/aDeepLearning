//
// Created by asus on 2025/1/10.
//
/**
 * @brief array is a dynamic batch type that can dynamically change the size of a batch like an STL vector.
 **/

#ifndef ARRAY_HPP
#define ARRAY_HPP
#include <vector>
#include <memory>
#include <iterator>
#include <stdexcept>

#include "batch.hpp"
#include "data_category.hpp"


namespace metann
{
    template <DataConcept Data, typename DataCate>
    class ArrayImpl;

    template <DataConcept DataType>
    class Array : public ArrayImpl<DataType, DataCategory_t<DataType>>
    {
    public:
        using ElementType = typename DataType::ElementType;
        using DeviceType = typename DataType::DeviceType;
        using ArrayImpl<DataType, DataCategory_t<DataType>>::ArrayImpl;
    };

    // template <DataConcept DataType>
    // constexpr bool IsBatchScalar_v<Array<DataType>> = IsScalar_v<DataType>;
    // template <DataConcept DataType>
    // constexpr bool IsBatchMatrix_v<Array<DataType>> = IsMatrix_v<DataType>;

    template <DataConcept DataType>
    constexpr bool IsBatchScalarHelper_v<Array<DataType>> = IsScalar_v<DataType>;
    template <DataConcept DataType>
    constexpr bool IsBatchMatrixHelper_v<Array<DataType>> = IsMatrix_v<DataType>;

    namespace eval
    {
        struct ScalarArrayEvalTag;
        struct MatrixArrayEvalTag;

        template <>
        struct UnitWrapper<ScalarArrayEvalTag>
        {
            template <typename InputELem, typename Elem, DeviceConcept Device, CategoryConcept Cate>
            struct EvalUnit;

            template <typename InputELem, typename Elem>
            struct EvalUnit<InputELem, Elem, CPU, CategoryTags::Scalar>
                : public BaseEvalUnit<CPU>
            {
                using ElementType = Elem;
                using DeviceType = CPU;

                EvalUnit(std::vector<InputELem> input,
                         EvalHandle<Batch<ElementType, CPU, CategoryTags::Scalar>> output)
                    : m_inputs(std::move(input))
                      , m_output(std::move(output))
                {
                }

                void eval() override
                {
                    if (m_inputs.empty())
                    {
                        m_output.allocate(0);
                    }
                    else
                    {
                        std::size_t tbn = m_inputs.size();
                        m_output.allocate(tbn);
                        auto& res = m_output.mutableData();

                        for (std::size_t bn = 0; bn < tbn; ++bn)
                        {
                            res.setValue(bn, m_inputs[bn].data().value());
                        }
                    }
                    m_output.setEval();
                }

            private:
                std::vector<InputELem> m_inputs;
                EvalHandle<Batch<ElementType, CPU, CategoryTags::Scalar>> m_output;
            };
        };

        template <>
        struct UnitWrapper<MatrixArrayEvalTag>
        {
            template <typename InputELem, typename Elem, DeviceConcept Device, CategoryConcept Cate>
            struct EvalUnit;

            template <typename InputELem, typename Elem>
            struct EvalUnit<InputELem, Elem, CPU, CategoryTags::Matrix>
                : public BaseEvalUnit<CPU>
            {
                using ElementType = Elem;
                using DeviceType = CPU;

                EvalUnit(std::vector<InputELem> input,
                         EvalHandle<Batch<ElementType, CPU, CategoryTags::Matrix>> output)
                    : m_inputs(std::move(input))
                      , m_output(std::move(output))
                {
                }

                void eval() override
                {
                    if (m_inputs.empty())
                    {
                        m_output.allocate(0, 0, 0);
                    }
                    else
                    {
                        std::size_t tbn = m_inputs.size();
                        std::size_t trn = m_inputs[0].data().rowNum();
                        std::size_t tcn = m_inputs[0].data().colNum();
                        m_output.allocate(tbn, trn, tcn);
                        auto& res = m_output.mutableData();

                        for (std::size_t bn = 0; bn < tbn; ++bn)
                        {
                            const auto& input = m_inputs[bn].data();

                            for (size_t i = 0; i < trn; ++i)
                            {
                                for (size_t j = 0; j < tcn; ++j)
                                {
                                    res.setValue(bn, i, j, input(i, j));
                                }
                            }
                        }
                    }
                    m_output.setEval();
                }

            private:
                std::vector<InputELem> m_inputs;
                EvalHandle<Batch<ElementType, CPU, CategoryTags::Matrix>> m_output;
            };
        };
    }

    template <DataConcept DataType>
    class ArrayImpl<DataType, CategoryTags::Scalar>
    {
    public:
        using ElementType = typename DataType::ElementType;
        using DeviceType = typename DataType::DeviceType;
        ArrayImpl() = default;

        template <typename IterType>
            requires requires { typename std::iterator_traits<IterType>::iterator_category; }
        ArrayImpl(IterType begin, IterType end)
            : m_buffer(new std::vector<ElementType>(begin, end))
        {
        }

        [[nodiscard]] std::size_t batchNum() const { return m_buffer->size(); }
        [[nodiscard]] std::size_t size() const { return m_buffer->size(); }

        void push_back(DataType data)
        {
            assert(availableForWrite());
            m_buffer->emplace_back(std::move(data));
        }

        [[nodiscard]] bool availableForWrite() const
        {
            return !m_evalBuf.isEvaluated() && m_buffer.use_count() == 1;
        }

        template <typename... Args>
        void emplace_back(Args&&... args)
        {
            assert(availableForWrite());
            DataType tmp(std::forward<Args>(args)...);
            m_buffer->emplace_back(std::move(tmp));
        }

        void reserve(std::size_t num)
        {
            assert(availableForWrite());
            m_buffer->reserve(num);
        }

        void clear()
        {
            assert(availableForWrite());
            m_buffer->clear();
        }

        [[nodiscard]] bool empty() const
        {
            return m_buffer->empty();
        }

        [[nodiscard]] const auto& operator[](std::size_t id) const
        {
            return (*m_buffer)[id];
        }

        [[nodiscard]] auto& operator[](std::size_t id)
        {
            return (*m_buffer)[id];
        }

        [[nodiscard]] auto begin() { return m_buffer->begin(); }
        [[nodiscard]] auto begin() const { return m_buffer->begin(); }
        [[nodiscard]] auto end() { return m_buffer->end(); }
        [[nodiscard]] auto end() const { return m_buffer->end(); }


        bool operator==(const Array<DataType>& rhs) const
        {
            const ArrayImpl& tmp = static_cast<const ArrayImpl&>(rhs);
            return m_buffer == tmp.m_buffer;
        }

        template <typename OtherType>
        bool operator==(const OtherType& rhs) const
        {
            return false;
        }

        template <typename OtherType>
        bool operator!=(const OtherType& rhs) const
        {
            return !(operator==(rhs));
        }

        auto evalRegister() const
        {
            if (!m_evalBuf.isEvaluated())
            {
                using OpEvalHandle = std::decay_t<decltype(std::declval<DataType>().evalRegister())>;
                std::vector<OpEvalHandle> handleBuf;
                std::vector<const void*> depVec;
                handleBuf.reserve(this->size());
                depVec.reserve(this->size());
                for (std::size_t i = 0; i < this->size(); ++i)
                {
                    handleBuf.push_back((*this)[i].evalRegister());
                    depVec.push_back(handleBuf.back().dataPtr());
                }

                auto outHandle = m_evalBuf.handle();

                using EvalUnit = eval::UnitWrapper<eval::ScalarArrayEvalTag>::EvalUnit<
                    OpEvalHandle, ElementType, DeviceType, CategoryTags::Scalar>;
                using GroupType = TrivialEvalGroup<EvalUnit>;

                const void* dataPtr = outHandle.dataPtr();
                EvalUnit unit(std::move(handleBuf), std::move(outHandle));
                EvalPlan<DeviceType>::template registerFun<GroupType>(
                    std::move(unit), dataPtr, std::move(depVec));
            }
            return m_evalBuf.constHandle();
        }

    private:
        std::shared_ptr<std::vector<DataType>> m_buffer{new std::vector<DataType>()};
        EvalBuffer<Batch<ElementType, DeviceType, CategoryTags::Scalar>> m_evalBuf;
    };

    template <DataConcept DataType>
    class ArrayImpl<DataType, CategoryTags::Matrix>
    {
    public:
        using ElementType = typename DataType::ElementType;
        using DeviceType = typename DataType::DeviceType;
        ArrayImpl() = default;

        ArrayImpl(const std::size_t rowNum, const std::size_t colNum):
            m_rowNum(rowNum), m_colNum{colNum}, m_buffer{std::make_shared<std::vector<DataType>>()}
        {
        }

        template <typename IterType> requires requires { typename std::iterator_traits<IterType>::iterator_category; }
        ArrayImpl(IterType begin, IterType end)
            : m_buffer(new std::vector<DataType>(begin, end))
        {
            if (!m_buffer->empty())
            {
                m_rowNum = m_buffer->at(0).rowNum();
                m_colNum = m_buffer->at(0).colNum();
                for (size_t i = 1; i < m_buffer->size(); ++i)
                {
                    if ((m_buffer->operator[](i).rowNum() != m_rowNum) ||
                        (m_buffer->operator[](i).colNum() != m_colNum))
                    {
                        throw std::runtime_error("Dimension mismatch");
                    }
                }
            }
        }

        [[nodiscard]] std::size_t rowNum() const { return m_rowNum; }
        [[nodiscard]] std::size_t colNum() const { return m_colNum; }
        [[nodiscard]] std::size_t batchNum() const { return m_buffer->size(); }
        [[nodiscard]] std::size_t size() const { return m_buffer->size(); }

        void push_back(DataType data)
        {
            assert(availableForWrite());
            assert(data.rowNum() == rowNum() && data.colNum() == colNum());
            m_buffer->emplace_back(std::move(data));
        }

        [[nodiscard]] bool availableForWrite() const
        {
            return !m_evalBuf.isEvaluated() && m_buffer.use_count() == 1;
        }

        template <typename... Args>
        void emplace_back(Args&&... args)
        {
            assert(availableForWrite());
            DataType tmp(std::forward<Args>(args)...);
            if ((tmp.rowNum() != m_rowNum) || (tmp.colNum() != m_colNum))
            {
                throw std::runtime_error("Dimension mismatch");
            }
            m_buffer->emplace_back(std::move(tmp));
        }

        void reserve(std::size_t num)
        {
            assert(availableForWrite());
            m_buffer->reserve(num);
        }

        void clear()
        {
            assert(availableForWrite());
            m_buffer->clear();
        }

        [[nodiscard]] bool empty() const
        {
            return m_buffer->empty();
        }

        [[nodiscard]] const auto& operator[](std::size_t id) const
        {
            return (*m_buffer)[id];
        }

        [[nodiscard]] auto& operator[](std::size_t id)
        {
            return (*m_buffer)[id];
        }

        [[nodiscard]] auto begin() { return m_buffer->begin(); }
        [[nodiscard]] auto begin() const { return m_buffer->begin(); }
        [[nodiscard]] auto end() { return m_buffer->end(); }
        [[nodiscard]] auto end() const { return m_buffer->end(); }


        bool operator==(const ArrayImpl& rhs) const
        {
            const ArrayImpl<DataType, CategoryTags::Matrix>& tmp =
                static_cast<const ArrayImpl<DataType, CategoryTags::Matrix>&>(rhs);
            return m_buffer == tmp.m_buffer;
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
                using TOpEvalHandle = std::decay_t<decltype(std::declval<DataType>().evalRegister())>;
                std::vector<TOpEvalHandle> handleBuf;
                std::vector<const void*> depVec;
                handleBuf.reserve(this->size());
                depVec.reserve(this->size());
                for (std::size_t i = 0; i < this->size(); ++i)
                {
                    handleBuf.push_back((*this)[i].evalRegister());
                    depVec.push_back(handleBuf.back().dataPtr());
                }

                auto outHandle = m_evalBuf.handle();

                using EvalUnit = eval::UnitWrapper<eval::MatrixArrayEvalTag>::EvalUnit<
                    TOpEvalHandle, ElementType, DeviceType, CategoryTags::Matrix>;
                using GroupType = TrivialEvalGroup<EvalUnit>;

                const void* dataPtr = outHandle.dataPtr();
                EvalUnit unit(std::move(handleBuf), std::move(outHandle));
                EvalPlan<DeviceType>::template registerFun<GroupType>
                    (std::move(unit), dataPtr, std::move(depVec));
            }
            return m_evalBuf.constHandle();
        }

    private:
        std::size_t m_rowNum{};
        std::size_t m_colNum{};
        std::shared_ptr<std::vector<DataType>> m_buffer;
        EvalBuffer<Batch<ElementType, DeviceType, CategoryTags::Matrix>> m_evalBuf;
    };


    template <typename IterType>
        requires requires { typename std::iterator_traits<IterType>::iterator_category; }
    auto make_array(IterType beg, IterType end)
    {
        using DataType = typename std::iterator_traits<IterType>::value_type;
        using RawDataType = std::remove_cvref_t<DataType>;
        return Array<RawDataType>(beg, end);
    }
} // namespace metann


#endif //ARRAY_HPP
