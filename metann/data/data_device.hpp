//
// Created by asus on 2025/1/7.
//

#ifndef DATA_DEVICE_HPP
#define DATA_DEVICE_HPP
#include <concepts>

namespace metann {
struct DeviceTags {
};

template <typename T>
concept DeviceConcept = std::derived_from<T, DeviceTags>;

struct CPU : DeviceTags {
};

template <typename Category, typename Element, DeviceConcept Device>
struct PrincipleDataType;

template <typename Category, typename Element, DeviceConcept Device>
using PrincipleDataType_t = typename PrincipleDataType<Category, Element, Device>::type;

}

#endif // DATA_DEVICE_HPP
