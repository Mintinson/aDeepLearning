//
// Created by asus on 2025/1/8.
//

#ifndef ALLOCATOR_HPP
#define ALLOCATOR_HPP

#include <memory>

#include "data_device.hpp"

namespace metann
{
    template <typename T>
    concept DataConcept = true;
    // {
    //     typename T::ElementType;
    //     requires DeviceConcept<typename T::DeviceType>;
    //     // typename T::DeviceType;
    // };

    template <DeviceConcept Device>
    struct Allocator;

    template <>
    struct Allocator<CPU>
    {
        template <typename T>
        static std::shared_ptr<T> allocate(const std::size_t elemSize)
        {
            return std::shared_ptr<T>(new T[elemSize], std::default_delete<T[]>());
        }
    };

    template <typename Element, DeviceConcept Device>
        requires std::same_as<Element, std::remove_cvref_t<Element>>
    class ContinuousMemory
    {
    public:
        using ElemType = Element;
        using ElemPtr = Element*;

        explicit ContinuousMemory(std::size_t size)
            : m_mem(Allocator<Device>::template allocate<ElemType>(size)),
              m_start(m_mem.get())
        {
        }

        ContinuousMemory(std::shared_ptr<ElemType> mem, ElemPtr start):
            m_mem(std::move(mem)), m_start(start)
        {
        }

        auto rawMemory() const { return m_start; }
        std::shared_ptr<ElemType> sharedPtr() const { return m_mem; }
        bool operator==(const ContinuousMemory& other) const = default;
        bool operator!=(const ContinuousMemory& other) const = default;
        std::size_t useCount() const { return m_mem.use_count(); }

    private:
        std::shared_ptr<ElemType> m_mem;
        ElemPtr m_start;
    };

    template <DataConcept DataType>
    struct LowerAccessImpl;

    template <DataConcept DataType>
    auto lower_access(DataType&& p)
        requires DataConcept<std::remove_cvref_t<DataType>>
    {
        using RawType = std::remove_cvref_t<DataType>;
        return LowerAccessImpl<RawType>(std::forward<DataType>(p));
    }

}

#endif //ALLOCATOR_HPP
