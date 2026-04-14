//
// Created by asus on 2025/1/8.
//

#ifndef BATCH_HPP
#define BATCH_HPP
#include "allocator.hpp"
#include "data_device.hpp"
#include "matrix.hpp"

namespace metann {
template <typename Element, DeviceConcept Device, typename Category>
class Batch;
template <typename Element, DeviceConcept Device>
class Batch<Element, Device, CategoryTags::Scalar>;
template <typename Element, DeviceConcept Device>
constexpr bool IsBatchScalarHelper_v<Batch<Element, Device, CategoryTags::Scalar>> = true;
template <typename Element, DeviceConcept Device>
class Batch<Element, Device, CategoryTags::Matrix>;
template <typename Element, DeviceConcept Device>
constexpr bool IsBatchMatrixHelper_v<Batch<Element, Device, CategoryTags::Matrix>> = true;

template <typename Element, DeviceConcept Device>
struct PrincipleDataType<CategoryTags::BatchMatrix, Element, Device> {
    using type = Batch<Element, Device, CategoryTags::Matrix>;
};

template <typename Element, DeviceConcept Device>
struct PrincipleDataType<CategoryTags::BatchScalar, Element, Device> {
    using type = Batch<Element, Device, CategoryTags::Scalar>;
};

/// ************** Batch Array **************
template <typename Element, DeviceConcept Device>
class Batch<Element, Device, CategoryTags::Scalar> {
public:
    using ElementType = Element;
    using DeviceType = Device;
    friend struct LowerAccessImpl<Batch>;

    explicit Batch(const std::size_t batchNum = 0) : m_mem(batchNum), m_batchNum(batchNum) {}

    [[nodiscard]] std::size_t batchNum() const { return m_batchNum; }

    [[nodiscard]] bool availableForWrite() const { return m_mem.useCount() == 1; }

    void setValue(const std::size_t batchId, ElementType value) {
        assert(availableForWrite());
        m_mem.rawMemory()[batchId] = value;
    }

    [[nodiscard]] const auto operator[](const std::size_t batchId) const { return m_mem.rawMemory()[batchId]; }

    [[nodiscard]] bool operator==(const Batch& rhs) const { return m_mem == rhs.m_mem && m_batchNum == rhs.m_batchNum; }

    template <typename OtherType>
    [[nodiscard]] bool operator==(const OtherType& rhs) const {
        return false;
    }

    template <typename OtherType>
    [[nodiscard]] bool operator!=(const OtherType& rhs) const {
        return !(*this == rhs);
    }

    auto evalRegister() const { return make_const_eval_handle(*this); }

private:
    ContinuousMemory<Element, Device> m_mem;
    std::size_t m_batchNum;
};

template <typename Element, DeviceConcept Device>
struct LowerAccessImpl<Batch<Element, Device, CategoryTags::Scalar>> {
    explicit LowerAccessImpl(Batch<Element, Device, CategoryTags::Scalar> data) : m_data(std::move(data)) {}

    [[nodiscard]] auto mutableRawMemory() { return m_data.m_mem.rawMemory(); }

    [[nodiscard]] const auto rawMemory() const { return m_data.m_mem.rawMemory(); }

private:
    Batch<Element, Device, CategoryTags::Scalar> m_data;
};

/// ************** Batch Matrix **************
template <typename Element, DeviceConcept Device>
class Batch<Element, Device, CategoryTags::Matrix> {
public:
    using ElementType = Element;
    using DeviceType = Device;
    friend struct LowerAccessImpl<Batch>;

    // Batch() = default;

    explicit Batch(const std::size_t batchNum = 0, const std::size_t rowNum = 0, const std::size_t colNum = 0)
        : m_mem(batchNum * rowNum * colNum)
        , m_rowNum(rowNum)
        , m_colNum(colNum)
        , m_batchNum(batchNum)
        , m_rowLen(colNum)
        , m_rawMatrixSize(rowNum * colNum) {}

    [[nodiscard]] std::size_t rowNum() const { return m_rowNum; }

    [[nodiscard]] std::size_t colNum() const { return m_colNum; }

    [[nodiscard]] std::size_t batchNum() const { return m_batchNum; }

    void setValue(const std::size_t batchId, const std::size_t rowId, const std::size_t colId, ElementType value) {
        assert(availableForWrite());
        (m_mem.rawMemory())[batchId * m_rawMatrixSize + rowId * m_rowLen + colId] = value;
    }

    [[nodiscard]] const auto operator[](const std::size_t batchId) const {
        auto pos = m_mem.rawMemory() + batchId * m_rawMatrixSize;
        return Matrix<ElementType, DeviceType>(m_mem.sharedPtr(), pos, m_rowNum, m_colNum, m_rowLen);
    }

    [[nodiscard]] Batch subMatrix(const std::size_t rowBeg,
                                  const std::size_t rowEnd,
                                  const std::size_t colBeg,
                                  const std::size_t colEnd) const {
        ElementType* pos = m_mem.rawMemory() + rowBeg * m_rowLen + colBeg;
        return Batch{m_mem.sharedPtr(), pos, rowEnd - rowBeg, colEnd - colBeg, m_batchNum, m_rowLen, m_rawMatrixSize};
    }

    bool availableForWrite() const { return m_mem.useCount() == 1; }

    bool operator==(const Batch& rhs) const {
        return (m_mem == rhs.m_mem) && (m_rowNum == rhs.m_rowNum) && (m_colNum == rhs.m_colNum) &&
               (m_batchNum == rhs.m_batchNum) && (m_rowLen == rhs.m_rowLen) && (m_rawMatrixSize == rhs.m_rawMatrixSize);
    }

    template <typename OtherType>
    bool operator==(const OtherType& rhs) const {
        return false;
    }

    template <typename OtherType>
    bool operator!=(const OtherType& rhs) const {
        return !(*this == rhs);
    }

    auto evalRegister() const { return make_const_eval_handle(*this); }

private:
    ContinuousMemory<ElementType, DeviceType> m_mem;
    std::size_t m_rowNum;
    std::size_t m_colNum;
    std::size_t m_batchNum;
    std::size_t m_rowLen;
    std::size_t m_rawMatrixSize;

    Batch(std::shared_ptr<ElementType> mem,
          ElementType* memStart,
          const std::size_t rowNum,
          const std::size_t colNum,
          const std::size_t batchNum,
          const std::size_t rowLen,
          const std::size_t matrixSize)
        : m_mem(mem, memStart)
        , m_rowNum(rowNum)
        , m_colNum(colNum)
        , m_batchNum(batchNum)
        , m_rowLen(rowLen)
        , m_rawMatrixSize(matrixSize) {}
};

template <typename Element, DeviceConcept Device>
struct LowerAccessImpl<Batch<Element, Device, CategoryTags::Matrix>> {
    explicit LowerAccessImpl(Batch<Element, Device, CategoryTags::Matrix> data) : m_data(std::move(data)) {}

    [[nodiscard]] auto mutableRawMemory() { return m_data.m_mem.rawMemory(); }

    [[nodiscard]] const auto rawMemory() const { return m_data.m_mem.rawMemory(); }

    [[nodiscard]] std::size_t RowLen() const { return m_data.m_rowLen; }

    [[nodiscard]] std::size_t RawMatrixSize() const { return m_data.m_rawMatrixSize; }

private:
    Batch<Element, Device, CategoryTags::Matrix> m_data;
};
}  // namespace metann

#endif  // BATCH_HPP
