//
// Created by asus on 2025/1/14.
//

#ifndef UPDATE_POLICY_HPP
#define UPDATE_POLICY_HPP

namespace metann {
struct FeedbackPolicy {
    using MajorClass = FeedbackPolicy;
    struct IsUpdateValueCate;
    struct IsFeedbackOutputValueCate;

    static constexpr bool isUpdate = false;          // whether to update
    static constexpr bool isFeedbackOutput = false;  // whether to calculate the gradient.
};

struct UpdatePolicy : virtual public FeedbackPolicy {
    using MinorClass = IsUpdateValueCate;
    static constexpr bool isUpdate = true;
};

struct NoUpdatePolicy : virtual public FeedbackPolicy {
    using MinorClass = IsUpdateValueCate;
    static constexpr bool isUpdate = false;
};

struct FeedbackOutputPolicy : virtual public FeedbackPolicy {
    using MinorClass = IsFeedbackOutputValueCate;
    static constexpr bool isFeedbackOutput = true;
};

struct NoFeedbackOutputPolicy : virtual public FeedbackPolicy {
    using MinorClass = IsFeedbackOutputValueCate;
    static constexpr bool isFeedbackOutput = false;
};

}  // namespace metann

#endif  // UPDATE_POLICY_HPP
