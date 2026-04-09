#include <metann/data/matrix.hpp>
#include <metann/operators/binary_operators.hpp>

#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace metann;

namespace
{
Matrix<float, CPU> build_features()
{
    Matrix<float, CPU> x(8, 2);

    x.setValue(0, 0, 1.0f); x.setValue(0, 1, 1.0f);
    x.setValue(1, 0, 2.0f); x.setValue(1, 1, 1.0f);
    x.setValue(2, 0, 3.0f); x.setValue(2, 1, 2.0f);
    x.setValue(3, 0, 4.0f); x.setValue(3, 1, 1.0f);
    x.setValue(4, 0, 1.0f); x.setValue(4, 1, 3.0f);
    x.setValue(5, 0, 2.0f); x.setValue(5, 1, 4.0f);
    x.setValue(6, 0, 3.0f); x.setValue(6, 1, 3.0f);
    x.setValue(7, 0, 5.0f); x.setValue(7, 1, 2.0f);

    return x;
}

Matrix<float, CPU> build_targets(const Matrix<float, CPU>& x)
{
    Matrix<float, CPU> y(x.rowNum(), 1);
    for (std::size_t i = 0; i < x.rowNum(); ++i)
    {
        const float label = 2.0f * x(i, 0) - 3.0f * x(i, 1) + 1.0f;
        y.setValue(i, 0, label);
    }
    return y;
}

float mse_loss(const Matrix<float, CPU>& prediction, const Matrix<float, CPU>& target)
{
    float total = 0.0f;
    for (std::size_t i = 0; i < prediction.rowNum(); ++i)
    {
        const float diff = prediction(i, 0) - target(i, 0);
        total += diff * diff;
    }
    return total / static_cast<float>(prediction.rowNum());
}

void compute_linear_grads(const Matrix<float, CPU>& x,
                          const Matrix<float, CPU>& prediction,
                          const Matrix<float, CPU>& target,
                          Matrix<float, CPU>& grad_w,
                          float& grad_b)
{
    std::vector<float> accum(x.colNum(), 0.0f);
    float bias_sum = 0.0f;

    for (std::size_t i = 0; i < x.rowNum(); ++i)
    {
        const float error = prediction(i, 0) - target(i, 0);
        bias_sum += error;
        for (std::size_t j = 0; j < x.colNum(); ++j)
        {
            accum[j] += error * x(i, j);
        }
    }

    const float scale = 2.0f / static_cast<float>(x.rowNum());
    for (std::size_t j = 0; j < x.colNum(); ++j)
    {
        grad_w.setValue(j, 0, scale * accum[j]);
    }
    grad_b = scale * bias_sum;
}
}

int main()
{
    Matrix<float, CPU> x = build_features();
    Matrix<float, CPU> y = build_targets(x);

    Matrix<float, CPU> w(2, 1);
    w.setValue(0, 0, 0.0f);
    w.setValue(1, 0, 0.0f);
    float b = 0.0f;

    const float learning_rate = 0.03f;
    const int epochs = 360;

    Matrix<float, CPU> grad_w(2, 1);
    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        auto prediction = evaluate(dot(x, w) + Scalar<float, CPU>(b));
        const float loss = mse_loss(prediction, y);

        float grad_b = 0.0f;
        compute_linear_grads(x, prediction, y, grad_w, grad_b);

        for (std::size_t j = 0; j < w.rowNum(); ++j)
        {
            w.setValue(j, 0, w(j, 0) - learning_rate * grad_w(j, 0));
        }
        b -= learning_rate * grad_b;

        if (epoch % 60 == 0 || epoch == epochs - 1)
        {
            std::cout << "epoch=" << std::setw(3) << epoch
                      << "  mse=" << std::fixed << std::setprecision(6) << loss << '\n';
        }
    }

    std::cout << "\nLearned parameters\n";
    std::cout << "w0=" << w(0, 0) << "  w1=" << w(1, 0) << "  b=" << b << '\n';
    std::cout << "Expected close to: w0=2, w1=-3, b=1\n\n";

    Matrix<float, CPU> probe(3, 2);
    probe.setValue(0, 0, 2.0f); probe.setValue(0, 1, 2.0f);
    probe.setValue(1, 0, 6.0f); probe.setValue(1, 1, 1.0f);
    probe.setValue(2, 0, 0.0f); probe.setValue(2, 1, 3.0f);

    auto probe_prediction = evaluate(dot(probe, w) + Scalar<float, CPU>(b));

    std::cout << "Prediction demo\n";
    for (std::size_t i = 0; i < probe.rowNum(); ++i)
    {
        const float expected = 2.0f * probe(i, 0) - 3.0f * probe(i, 1) + 1.0f;
        std::cout << "x=[" << probe(i, 0) << ", " << probe(i, 1) << "]"
                  << "  pred=" << probe_prediction(i, 0)
                  << "  expected=" << expected << '\n';
    }

    return 0;
}
