#include <cmath>
#include <cstddef>
#include <iostream>
#include <map>

#include <metann/layers/compose/single_layer.hpp>
#include <metann/layers/grad_collector.hpp>
#include <metann/layers/initializer.hpp>
#include <metann/layers/layer_io.hpp>
#include <metann/layers/policies/single_layer_policy.hpp>
#include <metann/operators/unary_operators.hpp>

using namespace metann;

namespace {
void print_matrix(const Matrix<float, CPU>& m, const char* title) {
    std::cout << title << " (" << m.rowNum() << "x" << m.colNum() << ")\n";
    for (std::size_t r = 0; r < m.rowNum(); ++r) {
        for (std::size_t c = 0; c < m.colNum(); ++c) {
            std::cout << m(r, c);
            if (c + 1 < m.colNum()) {
                std::cout << "  ";
            }
        }
        std::cout << '\n';
    }
}
}  // namespace

int main() {
    using DemoLayer = InjectPolicy_t<SingleLayer, UpdatePolicy, FeedbackOutputPolicy>;
    DemoLayer layer("demo", 2, 1);

    Matrix<float, CPU> w(2, 1);
    w.setValue(0, 0, 0.50f);
    w.setValue(1, 0, -0.25f);

    Matrix<float, CPU> b(1, 1);
    b.setValue(0, 0, 0.10f);

    auto init = make_initializer<float>();
    init.setMatrix("demo-weight", w);
    init.setMatrix("demo-bias", b);

    std::map<std::string, Matrix<float, CPU>> params;
    layer.init(init, params);

    Matrix<float, CPU> x(1, 2);
    x.setValue(0, 0, 1.40f);
    x.setValue(0, 1, -0.60f);

    const float target = 0.80f;

    auto forward_pack = layer.feedForward(LayerIO::create().set<LayerIO>(x));
    auto pred = evaluate(forward_pack.get<LayerIO>());

    const float error = pred(0, 0) - target;
    const float mse = error * error;

    Matrix<float, CPU> grad_out(1, 1);
    grad_out.setValue(0, 0, 2.0f * error);

    auto backward_pack = layer.feedBackward(LayerIO::create().set<LayerIO>(grad_out));
    auto grad_input = evaluate(backward_pack.get<LayerIO>());

    GradCollector<float, CPU> collector;
    layer.gradCollect(collector);

    print_matrix(x, "Input");
    print_matrix(pred, "Prediction");
    std::cout << "Target=" << target << "  MSE=" << mse << "\n\n";

    print_matrix(grad_input, "dLoss/dInput");

    std::cout << "\nCollected parameter gradients\n";
    for (auto& item : collector) {
        auto grad_matrix = evaluate(collapse(item.m_grad));
        print_matrix(item.m_weight, "Weight snapshot");
        print_matrix(grad_matrix, "dLoss/dWeight");
        std::cout << '\n';
    }

    return 0;
}
