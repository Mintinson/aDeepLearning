#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>

#include <metann/data/matrix.hpp>
#include <metann/operators/binary_operators.hpp>
#include <metann/operators/unary_operators.hpp>

using namespace metann;

namespace {
Matrix<float, CPU> make_xor_input() {
    Matrix<float, CPU> x(4, 2);
    x.setValue(0, 0, 0.0f);
    x.setValue(0, 1, 0.0f);
    x.setValue(1, 0, 0.0f);
    x.setValue(1, 1, 1.0f);
    x.setValue(2, 0, 1.0f);
    x.setValue(2, 1, 0.0f);
    x.setValue(3, 0, 1.0f);
    x.setValue(3, 1, 1.0f);
    return x;
}

Matrix<float, CPU> make_xor_label() {
    Matrix<float, CPU> y(4, 1);
    y.setValue(0, 0, 0.0f);
    y.setValue(1, 0, 1.0f);
    y.setValue(2, 0, 1.0f);
    y.setValue(3, 0, 0.0f);
    return y;
}

Matrix<float, CPU> add_row_bias(const Matrix<float, CPU>& x, const Matrix<float, CPU>& b) {
    Matrix<float, CPU> out(x.rowNum(), x.colNum());
    for (std::size_t i = 0; i < x.rowNum(); ++i) {
        for (std::size_t j = 0; j < x.colNum(); ++j) {
            out.setValue(i, j, x(i, j) + b(0, j));
        }
    }
    return out;
}

Matrix<float, CPU> mean_rows(const Matrix<float, CPU>& x) {
    Matrix<float, CPU> out(1, x.colNum());
    const float scale = 1.0f / static_cast<float>(x.rowNum());

    for (std::size_t j = 0; j < x.colNum(); ++j) {
        float sum = 0.0f;
        for (std::size_t i = 0; i < x.rowNum(); ++i) {
            sum += x(i, j);
        }
        out.setValue(0, j, sum * scale);
    }

    return out;
}

void sgd_update(Matrix<float, CPU>& param, const Matrix<float, CPU>& grad, float learning_rate) {
    for (std::size_t i = 0; i < param.rowNum(); ++i) {
        for (std::size_t j = 0; j < param.colNum(); ++j) {
            param.setValue(i, j, param(i, j) - learning_rate * grad(i, j));
        }
    }
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
}  // namespace

int main() {
    Matrix<float, CPU> x = make_xor_input();
    Matrix<float, CPU> y = make_xor_label();

    Matrix<float, CPU> w1(2, 4);
    w1.setValue(0, 0, 0.80f);
    w1.setValue(0, 1, -0.40f);
    w1.setValue(0, 2, 0.30f);
    w1.setValue(0, 3, -0.90f);
    w1.setValue(1, 0, 0.70f);
    w1.setValue(1, 1, 0.20f);
    w1.setValue(1, 2, -0.50f);
    w1.setValue(1, 3, -0.30f);

    Matrix<float, CPU> b1(1, 4);
    b1.setValue(0, 0, 0.00f);
    b1.setValue(0, 1, 0.00f);
    b1.setValue(0, 2, 0.00f);
    b1.setValue(0, 3, 0.00f);

    Matrix<float, CPU> w2(4, 1);
    w2.setValue(0, 0, 0.60f);
    w2.setValue(1, 0, -0.70f);
    w2.setValue(2, 0, 0.50f);
    w2.setValue(3, 0, -0.60f);

    Matrix<float, CPU> b2(1, 1);
    b2.setValue(0, 0, 0.00f);

    const float learning_rate = 0.80f;
    const int epochs = 6000;
    const float batch_size = static_cast<float>(x.rowNum());

    for (int epoch = 0; epoch < epochs; ++epoch) {
        auto z1_linear = evaluate(dot(x, w1));
        auto z1 = add_row_bias(z1_linear, b1);
        auto a1 = evaluate(metann::tanh(z1));

        auto z2_linear = evaluate(dot(a1, w2));
        auto z2 = add_row_bias(z2_linear, b2);
        auto a2 = evaluate(sigmoid(z2));

        const float loss = binary_cross_entropy(a2, y);

        auto dz2 = evaluate(a2 - y);
        auto dw2 = evaluate(dot(transpose(a1), dz2) / Scalar<float, CPU>(batch_size));
        auto db2 = mean_rows(dz2);

        auto da1 = evaluate(dot(dz2, transpose(w2)));
        auto tanh_prime = evaluate(Scalar<float, CPU>(1.0f) - (a1 * a1));
        auto dz1 = evaluate(da1 * tanh_prime);

        auto dw1 = evaluate(dot(transpose(x), dz1) / Scalar<float, CPU>(batch_size));
        auto db1 = mean_rows(dz1);

        sgd_update(w2, dw2, learning_rate);
        sgd_update(b2, db2, learning_rate);
        sgd_update(w1, dw1, learning_rate);
        sgd_update(b1, db1, learning_rate);

        if (epoch % 800 == 0 || epoch == epochs - 1) {
            std::cout << "epoch=" << std::setw(4) << epoch << "  bce=" << std::fixed << std::setprecision(6) << loss
                      << '\n';
        }
    }

    auto z1_final = add_row_bias(evaluate(dot(x, w1)), b1);
    auto a1_final = evaluate(metann::tanh(z1_final));
    auto z2_final = add_row_bias(evaluate(dot(a1_final, w2)), b2);
    auto probs = evaluate(sigmoid(z2_final));

    int correct = 0;
    std::cout << "\nXOR predictions\n";
    for (std::size_t i = 0; i < x.rowNum(); ++i) {
        const int pred = probs(i, 0) >= 0.5f ? 1 : 0;
        const int label = static_cast<int>(y(i, 0));
        if (pred == label) {
            ++correct;
        }

        std::cout << "x=[" << x(i, 0) << ", " << x(i, 1) << "]"
                  << "  p(y=1)=" << probs(i, 0) << "  pred=" << pred << "  label=" << label << '\n';
    }

    const float acc = static_cast<float>(correct) / static_cast<float>(x.rowNum());
    std::cout << "accuracy=" << acc * 100.0f << "%\n";

    return 0;
}
