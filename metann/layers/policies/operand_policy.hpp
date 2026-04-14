//
// Created by asus on 2025/1/14.
//

#ifndef OPERAND_POLICY_HPP
#define OPERAND_POLICY_HPP
#include "../../data/data_device.hpp"

namespace metann {
/**
 * @brief Policy class for specifying operand properties like device type and element type
 *
 * This policy defines the base configurations for operands used in neural network layers,
 * including what device they run on (e.g. CPU) and their element type (e.g. float).
 */
struct OperandPolicy {
    using MajorClass = OperandPolicy;
    struct DeviceTypeCate;
    using Device = CPU;

    struct ElementTypeCate;
    using ElementType = float;
};

/**
 * @brief Policy specialization for CPU device type
 *
 * Inherits from OperandPolicy and specifies CPU as the device type.
 */
struct CPUDevice : virtual OperandPolicy {
    using MinorClass = DeviceTypeCate;
    using Device = CPU;
};

/**
 * @brief Policy template for specifying element types
 *
 * @tparam T The element type to use (e.g. float, double)
 *
 * Inherits from OperandPolicy and allows specifying custom element types.
 */
template <typename T>
struct ElementTypeIs : virtual OperandPolicy {
    using MinorClass = ElementTypeCate;
    using ElementType = T;
};
}  // namespace metann

#endif  // OPERAND_POLICY_HPP
