#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>

#include <metann/data/matrix.hpp>
#include <metann/operators/binary_operators.hpp>
#include <metann/operators/unary_operators.hpp>

using namespace metann;

namespace {
Matrix<float, CPU> build_features() {
    Matrix<float, CPU> x(8, 2);

    x.setValue(0, 0, 0.0f);
    x.setValue(0, 1, 0.0f);
    x.setValue(1, 0, 0.0f);
    x.setValue(1, 1, 1.0f);
    x.setValue(2, 0, 1.0f);
    x.setValue(2, 1, 0.0f);
    x.setValue(3, 0, 1.0f);
    x.setValue(3, 1, 1.0f);
    x.setValue(4, 0, 0.1f);
    x.setValue(4, 1, 0.8f);
    x.setValue(5, 0, 0.8f);
    x.setValue(5, 1, 0.2f);
    x.setValue(6, 0, 0.9f);
    x.setValue(6, 1, 0.9f);
    x.setValue(7, 0, 0.2f);
    x.setValue(7, 1, 0.1f);

    return x;
}

Matrix<float, CPU> build_labels() {
    Matrix<float, CPU> y(8, 1);
    y.setValue(0, 0, 0.0f);
    y.setValue(1, 0, 1.0f);
    y.setValue(2, 0, 1.0f);
    y.setValue(3, 0, 1.0f);
    y.setValue(4, 0, 1.0f);
    y.setValue(5, 0, 1.0f);
    y.setValue(6, 0, 1.0f);
    y.setValue(7, 0, 0.0f);
    return y;
}

float binary_cross_entropy(const Matrix<float, CPU>& p, const Matrix<float, CPU>& y) {
    constexpr float eps = 1e-6f;
    float total = 0.0f;

    for (std::size_t i = 0; i < p.rowNum(); ++i) {
        const float pred = std::clamp(p(i, 0), eps, 1.0f - eps);
        const float label = y(i, 0);
        total += -(label * std::log(pred) + (1.0f - label) * std::log(1.0f - pred));
    }
    return total / static_cast<float>(p.rowNum());
}

void compute_logistic_grads(const Matrix<float, CPU>& x,
                            const Matrix<float, CPU>& p,
                            const Matrix<float, CPU>& y,
                            Matrix<float, CPU>& grad_w,
                            float& grad_b) {
    std::vector<float> accum(x.colNum(), 0.0f);
    float bias_sum = 0.0f;

    for (std::size_t i = 0; i < x.rowNum(); ++i) {
        const float error = p(i, 0) - y(i, 0);
        bias_sum += error;
        for (std::size_t j = 0; j < x.colNum(); ++j) {
            accum[j] += error * x(i, j);
        }
    }

    const float scale = 1.0f / static_cast<float>(x.rowNum());
    for (std::size_t j = 0; j < x.colNum(); ++j) {
        grad_w.setValue(j, 0, scale * accum[j]);
    }
    grad_b = scale * bias_sum;
}
}  // namespace

int main() {
    Matrix<float, CPU> x = build_features();
    Matrix<float, CPU> y = build_labels();

    Matrix<float, CPU> w(2, 1);
    w.setValue(0, 0, 0.0f);
    w.setValue(1, 0, 0.0f);
    float b = 0.0f;

    Matrix<float, CPU> grad_w(2, 1);
    const float learning_rate = 0.9f;
    const int epochs = 1400;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        auto logits = evaluate(dot(x, w) + Scalar<float, CPU>(b));
        auto probs = evaluate(sigmoid(logits));

        const float loss = binary_cross_entropy(probs, y);

        float grad_b = 0.0f;
        compute_logistic_grads(x, probs, y, grad_w, grad_b);

        for (std::size_t j = 0; j < w.rowNum(); ++j) {
            w.setValue(j, 0, w(j, 0) - learning_rate * grad_w(j, 0));
        }
        b -= learning_rate * grad_b;

        if (epoch % 280 == 0 || epoch == epochs - 1) {
            std::cout << "epoch=" << std::setw(4) << epoch << "  bce=" << std::fixed << std::setprecision(6) << loss
                      << '\n';
        }
    }

    std::cout << "\nLearned parameters\n";
    std::cout << "w0=" << w(0, 0) << "  w1=" << w(1, 0) << "  b=" << b << "\n\n";

    Matrix<float, CPU> eval_points(4, 2);
    eval_points.setValue(0, 0, 0.0f);
    eval_points.setValue(0, 1, 0.0f);
    eval_points.setValue(1, 0, 0.0f);
    eval_points.setValue(1, 1, 1.0f);
    eval_points.setValue(2, 0, 1.0f);
    eval_points.setValue(2, 1, 0.0f);
    eval_points.setValue(3, 0, 1.0f);
    eval_points.setValue(3, 1, 1.0f);

    auto logits = evaluate(dot(eval_points, w) + Scalar<float, CPU>(b));
    auto probs = evaluate(sigmoid(logits));

    std::cout << "OR classification demo\n";
    for (std::size_t i = 0; i < eval_points.rowNum(); ++i) {
        const int cls = probs(i, 0) >= 0.5f ? 1 : 0;
        std::cout << "x=[" << eval_points(i, 0) << ", " << eval_points(i, 1) << "]"
                  << "  p(y=1)=" << probs(i, 0) << "  pred_class=" << cls << '\n';
    }

    return 0;
}
