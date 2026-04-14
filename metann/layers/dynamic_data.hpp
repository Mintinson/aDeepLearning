//
// Created by asus on 2025/1/13.
//

#ifndef DYNAMIC_DATA_HPP
#define DYNAMIC_DATA_HPP
#include <memory>

#include "../data/data_category.hpp"
#include "../data/data_device.hpp"
#include "../eval/facilities.hpp"

namespace metann {
/**
 * @brief : To hide all the details of the data. Only support necessary interfaces.
 * @tparam Element : Element type
 * @tparam Device : Device type
 * @tparam DataCate : Data category
 */
template <typename Element, DeviceConcept Device, CategoryConcept DataCate>
class DynamicCategory;

template <typename Element, DeviceConcept Device>
class DynamicCategory<Element, Device, CategoryTags::Matrix> {
public:
    using ElementType = Element;
    using DeviceType = Device;
    using EvalType = PrincipleDataType_t<CategoryTags::Matrix, ElementType, DeviceType>;

    template <typename Base>
    explicit DynamicCategory(const Base& base) : m_rowNum(base.rowNum())
                                               , m_colNum(base.colNum()) {}

    virtual ~DynamicCategory() = default;
    virtual bool operator==(const DynamicCategory& val) const = 0;
    virtual bool operator!=(const DynamicCategory& val) const = 0;

    [[nodiscard]] std::size_t rowNum() const { return m_rowNum; }

    [[nodiscard]] std::size_t colNum() const { return m_colNum; }

    virtual DynamicConstEvalHandle<EvalType> evalRegister() const = 0;

private:
    std::size_t m_rowNum;
    std::size_t m_colNum;
};

template <typename Element, DeviceConcept Device>
class DynamicCategory<Element, Device, CategoryTags::BatchMatrix> {
public:
    using ElementType = Element;
    using DeviceType = Device;
    using EvalType = PrincipleDataType_t<CategoryTags::BatchMatrix, ElementType, DeviceType>;

    template <typename Base>
    explicit DynamicCategory(const Base& base)
        : m_rowNum(base.rowNum())
        , m_colNum(base.colNum())
        , m_batchNum(base.batchNum()) {}

    virtual ~DynamicCategory() = default;
    virtual bool operator==(const DynamicCategory& val) const = 0;
    virtual bool operator!=(const DynamicCategory& val) const = 0;

    [[nodiscard]] std::size_t rowNum() const { return m_rowNum; }

    [[nodiscard]] std::size_t colNum() const { return m_colNum; }

    [[nodiscard]] std::size_t batchNum() const { return m_batchNum; }

    // TODO: virtual evaluation interface
    virtual DynamicConstEvalHandle<EvalType> evalRegister() const = 0;

private:
    std::size_t m_rowNum;
    std::size_t m_colNum;
    std::size_t m_batchNum;
};

/**
 * @brief A wrapper class that take into a specific data type, and provide a unified interface.
 *        it is derived from DynamicCategory, so we can use a pointer to access the real data.
 * @tparam BaseData
 */
template <DataConcept BaseData>
class DynamicWrapper
    : public DynamicCategory<typename BaseData::ElementType, typename BaseData::DeviceType, DataCategory_t<BaseData>> {
    using BaseClass =
        DynamicCategory<typename BaseData::ElementType, typename BaseData::DeviceType, DataCategory_t<BaseData>>;

public:
    DynamicWrapper(BaseData data) : BaseClass(data), m_data(std::move(data)) {}

    // TODO: evaluation
    bool operator==(const BaseClass& val) const override {
        try {
            const DynamicWrapper& real = dynamic_cast<const DynamicWrapper&>(val);
            return m_data == real.m_data;
        } catch (std::bad_cast&) {
            return false;
        }
    }

    bool operator!=(const BaseClass& val) const override { return !(operator==(val)); }

    DynamicConstEvalHandle<typename BaseClass::EvalType> evalRegister() const override {
        return DynamicConstEvalHandle<typename BaseClass::EvalType>{m_data.evalRegister()};
    }

    const BaseData& baseData() const { return m_data; }

private:
    BaseData m_data;
};

template <typename Element, DeviceConcept Device, CategoryConcept DataCate>
class DynamicData;

template <typename Element, DeviceConcept Device>
class DynamicData<Element, Device, CategoryTags::Matrix> {
    using BaseData = DynamicCategory<Element, Device, CategoryTags::Matrix>;

public:
    using ElementType = Element;
    using DeviceType = Device;
    using ResHandleType = decltype(std::declval<BaseData>().evalRegister());
    DynamicData() = default;

    template <typename OriginalData>
    DynamicData(std::shared_ptr<DynamicWrapper<OriginalData>> data) : m_baseData(std::move(data)) {}

    [[nodiscard]] std::size_t rowNum() const { return m_baseData->rowNum(); }

    [[nodiscard]] std::size_t colNum() const { return m_baseData->colNum(); }

    auto evalRegister() const { return m_baseData->evalRegister(); }

    bool operator==(const DynamicData& val) const {
        if ((!m_baseData) && (!val.m_baseData)) {
            return true;
        }
        if ((!m_baseData) || (!val.m_baseData)) {
            return false;
        }
        BaseData& val1 = *m_baseData;
        BaseData& val2 = *(val.m_baseData);
        return val1 == val2;
    }

    template <typename OtherType>
    bool operator==(const OtherType& val) const {
        return false;
    }

    template <typename OtherType>
    bool operator!=(const OtherType& val) const {
        return !(operator==(val));
    }

private:
    std::shared_ptr<BaseData> m_baseData;
};

template <typename Element, DeviceConcept Device>
class DynamicData<Element, Device, CategoryTags::BatchMatrix> {
    using BaseData = DynamicCategory<Element, Device, CategoryTags::BatchMatrix>;

public:
    using ElementType = Element;
    using DeviceType = Device;
    using ResHandleType = decltype(std::declval<BaseData>().evalRegister());
    DynamicData() = default;

    template <typename OriginalData>
    explicit DynamicData(std::shared_ptr<DynamicWrapper<OriginalData>> data) : m_baseData(std::move(data)) {}

    bool operator==(const DynamicData& val) const {
        if ((!m_baseData && (!val.m_baseData))) {
            return true;
        }
        if ((!m_baseData) || (!val.m_baseData)) {
            return false;
        }
        BaseData& val1 = *m_baseData;
        BaseData& val2 = *(val.m_baseData);
        return val1 == val2;
    }

    template <typename OtherType>
    bool operator==(const OtherType& val) const {
        return false;
    }

    template <typename OtherType>
    bool operator!=(const OtherType& val) const {
        return !(operator==(val));
    }

    [[nodiscard]] std::size_t rowNum() const { return m_baseData->rowNum(); }

    [[nodiscard]] std::size_t colNum() const { return m_baseData->colNum(); }

    [[nodiscard]] std::size_t batchNum() const { return m_baseData->batchNum(); }

    auto evalRegister() const { return m_baseData->evalRegister(); }

private:
    std::shared_ptr<BaseData> m_baseData;
};

template <typename Element, DeviceConcept Device>
constexpr bool IsMatrixHelper_v<DynamicData<Element, Device, CategoryTags::Matrix>> = true;
template <typename Element, DeviceConcept Device>
constexpr bool IsBatchMatrix_v<DynamicData<Element, Device, CategoryTags::BatchMatrix>> = true;

namespace details {
template <typename DataType>
struct IsDynamicHelper : std::false_type {};

template <typename Element, DeviceConcept Device, CategoryConcept DataCate>
struct IsDynamicHelper<DynamicData<Element, Device, DataCate>> : std::true_type {};
}  // namespace details

template <DataConcept DataType>
constexpr bool IsDynamic_v = details::IsDynamicHelper<std::remove_cvref_t<DataType>>::value;

template <DataConcept DataType>
auto make_dynamic(DataType&& data) {
    if constexpr (IsDynamic_v<DataType>) {
        return std::forward<DataType>(data);
    } else {
        using RawDataType = std::remove_cvref_t<DataType>;
        using DerivedData = DynamicWrapper<RawDataType>;
        auto baseData = std::make_shared<DerivedData>(std::forward<DataType>(data));
        return DynamicData<typename RawDataType::ElementType, typename RawDataType::DeviceType,
                           DataCategory_t<RawDataType>>(std::move(baseData));
    }
}
}  // namespace metann

#endif  // DYNAMIC_DATA_HPP
