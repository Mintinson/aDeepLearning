//
// Created by asus on 2025/1/14.
//

#ifndef INPUT_POLICY_HPP
#define INPUT_POLICY_HPP

namespace metann {
struct InputPolicy {
    using MajorClass = InputPolicy;
    struct BatchModelValueCate;
    static constexpr bool BatchModel = false;
};

struct BatchModelPolicy : virtual InputPolicy {
    using MinorClass = BatchModelValueCate;
    static constexpr bool BatchModel = true;
};

struct NoBatchModelPolicy : virtual InputPolicy {
    using MinorClass = InputPolicy;
    static constexpr bool BatchModel = false;
};

}  // namespace metann

#endif  // INPUT_POLICY_HPP
