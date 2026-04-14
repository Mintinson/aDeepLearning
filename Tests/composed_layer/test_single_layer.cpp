//
// Created by asus on 2025/2/18.
//
#include <iostream>
#include <map>
#include <string>

#include <metann/layers/compose/single_layer.hpp>
#include <metann/layers/grad_collector.hpp>
#include <metann/layers/initializer.hpp>
#include <metann/policy/policy.hpp>

using std::cout;
using std::endl;
using namespace metann;

void test_single_layer1() {
    // No update, action is sigmoid, with bias
    cout << "Test single layer case 1 ...\t";
    using RootLayer = InjectPolicy_t<SingleLayer>;
    static_assert(!RootLayer::isUpdate, "Test Error");
    static_assert(!RootLayer::isFeedbackOutput, "Test Error");

    RootLayer layer("root", 2, 3);
    std::map<std::string, Matrix<float, CPU>> params;

    Matrix<float, CPU> w1(2, 3);
    w1.setValue(0, 0, 0.1f);
    w1.setValue(1, 0, 0.2f);
    w1.setValue(0, 1, 0.3f);
    w1.setValue(1, 1, 0.4f);
    w1.setValue(0, 2, 0.5f);
    w1.setValue(1, 2, 0.6f);

    Matrix<float, CPU> b1(1, 3);
    b1.setValue(0, 0, 0.7f);
    b1.setValue(0, 1, 0.8f);
    b1.setValue(0, 2, 0.9f);

    auto initializer = make_initializer<float>();
    initializer.setMatrix("root-weight", w1);
    initializer.setMatrix("root-bias", b1);
    layer_init(layer, initializer, params);

    Matrix<float, CPU> i(1, 2);
    i.setValue(0, 0, 0.1f);
    i.setValue(0, 1, 0.2f);

    auto input = LayerIO::create().set<LayerIO>(i);
    auto out = evaluate(layer.feedForward(input).get<LayerIO>());

    assert(fabs(out(0, 0) - (1 / (1 + exp(-0.75)))) < 0.00001);
    assert(fabs(out(0, 1) - (1 / (1 + exp(-0.91)))) < 0.00001);
    assert(fabs(out(0, 2) - (1 / (1 + exp(-1.07)))) < 0.00001);

    auto fbIn = LayerIO::create();
    auto out_grad = layer.feedBackward(fbIn);
    auto fbOut = out_grad.get<LayerIO>();
    static_assert(std::is_same_v<decltype(fbOut), details::NullParameter>, "Test error");

    params.clear();
    layer.saveWeights(params);
    assert(params.find("root-weight") != params.end());
    assert(params.find("root-bias") != params.end());

    cout << "done" << endl;
}

void test_single_layer2() {
    // No update, action is tanh, with bias
    cout << "Test single layer case 2 ...\t";
    using RootLayer = InjectPolicy_t<SingleLayer, TanhAction>;
    static_assert(!RootLayer::isUpdate, "Test Error");
    static_assert(!RootLayer::isFeedbackOutput, "Test Error");

    RootLayer layer("root", 2, 3);
    std::map<std::string, Matrix<float, CPU>> params;

    Matrix<float, CPU> w1(2, 3);
    w1.setValue(0, 0, 0.1f);
    w1.setValue(1, 0, 0.2f);
    w1.setValue(0, 1, 0.3f);
    w1.setValue(1, 1, 0.4f);
    w1.setValue(0, 2, 0.5f);
    w1.setValue(1, 2, 0.6f);

    Matrix<float, CPU> b1(1, 3);
    b1.setValue(0, 0, 0.7f);
    b1.setValue(0, 1, 0.8f);
    b1.setValue(0, 2, 0.9f);

    auto initializer = make_initializer<float>();
    initializer.setMatrix("root-weight", w1);
    initializer.setMatrix("root-bias", b1);
    layer_init(layer, initializer, params);

    Matrix<float, CPU> i(1, 2);
    i.setValue(0, 0, 0.1f);
    i.setValue(0, 1, 0.2f);

    auto input = LayerIO::create().set<LayerIO>(i);
    auto out = evaluate(layer.feedForward(input).get<LayerIO>());

    assert(fabs(out(0, 0) - tanh(0.75)) < 0.00001);
    assert(fabs(out(0, 1) - tanh(0.91)) < 0.00001);
    assert(fabs(out(0, 2) - tanh(1.07)) < 0.00001);

    auto out_grad = layer.feedBackward(LayerIO::create());
    auto fbOut = out_grad.get<LayerIO>();
    static_assert(std::is_same_v<decltype(fbOut), details::NullParameter>, "Test error");

    params.clear();
    layer.saveWeights(params);
    assert(params.find("root-weight") != params.end());
    assert(params.find("root-bias") != params.end());

    cout << "done" << endl;
}

void test_single_layer3() {
    // No update, action is sigmoid, no bias
    cout << "Test single layer case 3 ...\t";
    using RootLayer = InjectPolicy_t<SingleLayer, NoBiasSingleLayer>;
    static_assert(!RootLayer::isUpdate, "Test Error");
    static_assert(!RootLayer::isFeedbackOutput, "Test Error");

    RootLayer layer("root", 2, 3);
    std::map<std::string, Matrix<float, CPU>> params;

    Matrix<float, CPU> w1(2, 3);
    w1.setValue(0, 0, 0.1f);
    w1.setValue(1, 0, 0.2f);
    w1.setValue(0, 1, 0.3f);
    w1.setValue(1, 1, 0.4f);
    w1.setValue(0, 2, 0.5f);
    w1.setValue(1, 2, 0.6f);

    auto initializer = make_initializer<float>();
    initializer.setMatrix("root-weight", w1);
    layer_init(layer, initializer, params);

    Matrix<float, CPU> i(1, 2);
    i.setValue(0, 0, 0.1f);
    i.setValue(0, 1, 0.2f);

    auto input = LayerIO::create().set<LayerIO>(i);
    auto out = evaluate(layer.feedForward(input).get<LayerIO>());

    assert(fabs(out(0, 0) - (1 / (1 + exp(-0.05)))) < 0.00001);
    assert(fabs(out(0, 1) - (1 / (1 + exp(-0.11)))) < 0.00001);
    assert(fabs(out(0, 2) - (1 / (1 + exp(-0.17)))) < 0.00001);

    auto out_grad = layer.feedBackward(LayerIO::create());
    auto fbOut = out_grad.get<LayerIO>();
    static_assert(std::is_same_v<decltype(fbOut), details::NullParameter>, "Test error");

    params.clear();
    layer.saveWeights(params);
    assert(params.find("root-weight") != params.end());

    cout << "done" << endl;
}

void test_single_layer4() {
    // Update, action is sigmoid, with bias
    cout << "Test single layer case 4 ...\t";
    using RootLayer = InjectPolicy_t<SingleLayer, UpdatePolicy>;
    static_assert(RootLayer::isUpdate, "Test Error");
    static_assert(!RootLayer::isFeedbackOutput, "Test Error");

    RootLayer layer("root", 2, 3);
    std::map<std::string, Matrix<float, CPU>> params;

    Matrix<float, CPU> w1(2, 3);
    w1.setValue(0, 0, 0.1f);
    w1.setValue(1, 0, 0.2f);
    w1.setValue(0, 1, 0.3f);
    w1.setValue(1, 1, 0.4f);
    w1.setValue(0, 2, 0.5f);
    w1.setValue(1, 2, 0.6f);

    Matrix<float, CPU> b1(1, 3);
    b1.setValue(0, 0, 0.7f);
    b1.setValue(0, 1, 0.8f);
    b1.setValue(0, 2, 0.9f);

    auto initializer = make_initializer<float>();
    initializer.setMatrix("root-weight", w1);
    initializer.setMatrix("root-bias", b1);
    layer_init(layer, initializer, params);

    Matrix<float, CPU> i(1, 2);
    i.setValue(0, 0, 0.1f);
    i.setValue(0, 1, 0.2f);

    auto input = LayerIO::create().set<LayerIO>(i);
    auto out = evaluate(layer.feedForward(input).get<LayerIO>());

    assert(fabs(out(0, 0) - (1 / (1 + exp(-0.75)))) < 0.00001);
    assert(fabs(out(0, 1) - (1 / (1 + exp(-0.91)))) < 0.00001);
    assert(fabs(out(0, 2) - (1 / (1 + exp(-1.07)))) < 0.00001);

    Matrix<float, CPU> grad(1, 3);
    grad.setValue(0, 0, 0.19);
    grad.setValue(0, 1, 0.23);
    grad.setValue(0, 2, -0.15);

    auto fbIn = LayerIO::create().set<LayerIO>(grad);
    auto out_grad = layer.feedBackward(fbIn);
    auto fbOut = out_grad.get<LayerIO>();
    static_assert(std::is_same_v<decltype(fbOut), details::NullParameter>, "Test error");

    GradCollector<float, CPU> grad_collector;
    layer.gradCollect(grad_collector);
    assert(grad_collector.size() == 2);

    bool weight_update_valid = false;
    bool bias_update_valid = false;

    for (auto& p : grad_collector) {
        auto w = p.m_weight;
        auto info = evaluate(collapse(p.m_grad));
        if (w.rowNum() == w1.rowNum()) {
            weight_update_valid = true;
            assert(fabs(info(0, 0) - 0.0414 * 0.1) < 0.00001);
            assert(fabs(info(1, 0) - 0.0414 * 0.2) < 0.00001);
            assert(fabs(info(0, 1) - 0.04706511 * 0.1) < 0.00001);
            assert(fabs(info(1, 1) - 0.04706511 * 0.2) < 0.00001);
            assert(fabs(info(0, 2) + 0.02852585 * 0.1) < 0.00001);
            assert(fabs(info(1, 2) + 0.02852585 * 0.2) < 0.00001);
        } else if (w.rowNum() == b1.rowNum()) {
            bias_update_valid = true;
            assert(fabs(info(0, 0) - 0.0414) < 0.00001);
            assert(fabs(info(0, 1) - 0.04706511) < 0.00001);
            assert(fabs(info(0, 2) + 0.02852585) < 0.00001);
        } else {
            assert(false);
        }
    }
    assert(bias_update_valid);
    assert(weight_update_valid);

    params.clear();
    layer.saveWeights(params);
    assert(params.find("root-weight") != params.end());
    assert(params.find("root-bias") != params.end());

    cout << "done" << endl;
}

int main() {
    std::cout << "Testing Single Layer..." << std::endl;
    test_single_layer1();
    test_single_layer2();
    test_single_layer3();
    test_single_layer4();
    std::cout << "All tests passed!" << std::endl;
}
