//
// Created by asus on 2025/1/11.
//

#ifndef OPERATOR_CATEGORY_HPP
#define OPERATOR_CATEGORY_HPP

namespace metann {
template <typename T>
concept UnaryOperConcept = true;

struct UnaryOperTags {
    struct Sigmoid;
    struct Transpose;
    struct Collapse;
    struct Abs;
    struct Sign;
    struct Tanh;
    struct VecSoftmax;
    struct ReLU;
};

template <typename T>
concept BinaryOperConcept = true;

struct BinaryOperTags {
    struct Add;
    struct Sub;
    struct Mul;
    struct Div;
    struct Dot;
    struct NegativeLogLikelihood;
    struct VecSoftmaxDerivative;
    struct SigmoidDerivative;
    struct TanhDerivative;
};

template <typename T>
concept TernaryOperConcept = true;

struct TernaryOperTags {
    struct NegativeLogLikelihoodDerivative;
    struct Interpolate;
};

template <typename T>
concept OperTagConcept = UnaryOperConcept<T> || BinaryOperConcept<T> || TernaryOperConcept<T>;
}  // namespace metann

#endif  // OPERATOR_CATEGORY_HPP
