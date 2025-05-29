//
// Created by asus on 2025/1/8.
//

#ifndef SCALAR_HPP
#define SCALAR_HPP

#include "data_category.hpp"
#include "data_device.hpp"
#include "../eval/facilities.hpp"

namespace metann
{
    template <typename Element, DeviceConcept Device = CPU>
    struct Scalar;

    template <typename Element, DeviceConcept Device>
    constexpr bool IsScalarHelper_v<Scalar<Element, Device>> = true;

    template<typename Element, DeviceConcept Device>
    struct PrincipleDataType<CategoryTags::Scalar, Element, Device>
    {
        using type = Scalar<Element, Device>;
    };


    template <typename Element>
    struct Scalar<Element, CPU>  // specialization for CPU scalar
    {
    public:
        using ElementType = Element;
        using DeviceType = CPU;

        explicit Scalar(ElementType element) : m_elem(element)
        {
        }
        // rule of six
        Scalar() = default;
        Scalar(const Scalar&) = default;
        Scalar(Scalar&&) noexcept = default;
        Scalar& operator=(const Scalar&) = default;
        Scalar& operator=(Scalar&&) noexcept = default;
        ~Scalar() = default;
    public:
        ElementType& value() {return m_elem;}
        const ElementType& value() const {return m_elem;}

        bool operator==(const Scalar& rhs) const
        {
            return m_elem == rhs.m_elem;
        }
        template<typename OtherType>
        bool operator==(const OtherType& rhs) const
        {
            return false;
        }
        template<typename OtherType>
        bool operator!=(const OtherType& rhs) const
        {
            return !(this->operator==(rhs));
        }

        auto evalRegister() const
        {
            return make_const_eval_handle(*this);
        }
    private:
        ElementType m_elem{};
    };
} // namespace metann


#endif //SCALAR_HPP
