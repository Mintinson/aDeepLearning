#include <metann/data/matrix.hpp>
#include <metann/layers/compose/single_layer.hpp>
#include <metann/layers/initializer.hpp>
#include <metann/layers/layer_io.hpp>
#include <metann/layers/policies/single_layer_policy.hpp>

#include <cstddef>
#include <iostream>
#include <map>

using namespace metann;

namespace
{
void print_matrix(const Matrix<float, CPU>& m, const char* title)
{
    std::cout << title << " (" << m.rowNum() << "x" << m.colNum() << ")\n";
    for (std::size_t r = 0; r < m.rowNum(); ++r)
    {
        for (std::size_t c = 0; c < m.colNum(); ++c)
        {
            std::cout << m(r, c);
            if (c + 1 < m.colNum())
            {
                std::cout << "  ";
            }
        }
        std::cout << '\n';
    }
}
}

int main()
{
    using HiddenLayer = InjectPolicy_t<SingleLayer, TanhAction>;
    using OutputLayer = InjectPolicy_t<SingleLayer>;

    HiddenLayer hidden("hidden", 2, 3);
    OutputLayer output("output", 3, 1);

    Matrix<float, CPU> hidden_w(2, 3);
    hidden_w.setValue(0, 0, 1.20f); hidden_w.setValue(0, 1, -0.70f); hidden_w.setValue(0, 2, 0.50f);
    hidden_w.setValue(1, 0, 0.30f); hidden_w.setValue(1, 1, 1.10f); hidden_w.setValue(1, 2, -0.40f);

    Matrix<float, CPU> hidden_b(1, 3);
    hidden_b.setValue(0, 0, 0.10f); hidden_b.setValue(0, 1, -0.20f); hidden_b.setValue(0, 2, 0.05f);

    auto hidden_init = make_initializer<float>();
    hidden_init.setMatrix("hidden-weight", hidden_w);
    hidden_init.setMatrix("hidden-bias", hidden_b);
    std::map<std::string, Matrix<float, CPU>> hidden_params;
    hidden.init(hidden_init, hidden_params);

    Matrix<float, CPU> output_w(3, 1);
    output_w.setValue(0, 0, 1.40f);
    output_w.setValue(1, 0, -1.10f);
    output_w.setValue(2, 0, 0.90f);

    Matrix<float, CPU> output_b(1, 1);
    output_b.setValue(0, 0, -0.15f);

    auto output_init = make_initializer<float>();
    output_init.setMatrix("output-weight", output_w);
    output_init.setMatrix("output-bias", output_b);
    std::map<std::string, Matrix<float, CPU>> output_params;
    output.init(output_init, output_params);

    Matrix<float, CPU> sample(1, 2);
    sample.setValue(0, 0, 0.90f);
    sample.setValue(0, 1, -0.30f);

    auto hidden_out_pack = hidden.feedForward(LayerIO::create().set<LayerIO>(sample));
    auto output_out_pack = output.feedForward(hidden_out_pack);

    auto hidden_out = evaluate(hidden_out_pack.get<LayerIO>());
    auto prediction = evaluate(output_out_pack.get<LayerIO>());

    print_matrix(sample, "Input");
    print_matrix(hidden_out, "Hidden activation (tanh)");
    print_matrix(prediction, "Output activation (sigmoid)");

    return 0;
}
