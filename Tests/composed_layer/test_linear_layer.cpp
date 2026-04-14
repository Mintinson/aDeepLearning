
#include <iostream>

#include <metann/layers/compose/compose_core.hpp>
#include <metann/layers/compose/linear_layer.hpp>
#include <metann/layers/compose/structure.hpp>
#include <metann/layers/elementary/add_layer.hpp>
#include <metann/layers/elementary/bias_layer.hpp>
#include <metann/layers/elementary/mul_layer.hpp>
#include <metann/layers/elementary/sigmoid_layer.hpp>
#include <metann/layers/elementary/tanh_layer.hpp>
#include <metann/layers/elementary/weight_layer.hpp>
#include <metann/layers/grad_collector.hpp>
#include <metann/layers/initializer.hpp>
#include <metann/layers/layer_io.hpp>
#include <metann/layers/policies/single_layer_policy.hpp>

using namespace metann;
using std::cout;
using std::endl;

void test_linear_layer1() {
    cout << "Test linear layer case 1 ...\t";
    using RootLayer = InjectPolicy_t<LinearLayer>;
    static_assert(!RootLayer::isUpdate, "Test Error");
    static_assert(!RootLayer::isFeedbackOutput, "Test Error");

    RootLayer layer("root", 2, 3);

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
    std::map<std::string, Matrix<float, CPU>> params;
    layer.init(initializer, params);
    //
    Matrix<float, CPU> i(1, 2);
    i.setValue(0, 0, 0.1f);
    i.setValue(0, 1, 0.2f);
    //
    auto input = LayerIO::create().set<LayerIO>(i);
    auto out = evaluate(layer.feedForward(input).get<LayerIO>());
    //
    assert(fabs(out(0, 0) - 0.75f) < 0.00001);
    assert(fabs(out(0, 1) - 0.91f) < 0.00001);
    assert(fabs(out(0, 2) - 1.07f) < 0.00001);
    //
    auto out_grad = layer.feedBackward(LayerIO::create());
    auto fbOut = out_grad.get<LayerIO>();
    static_assert(std::is_same_v<decltype(fbOut), details::NullParameter>, "Test error");
    //
    params.clear();
    layer.saveWeights(params);
    assert(params.find("root-weight") != params.end());
    assert(params.find("root-bias") != params.end());

    cout << "done" << endl;
}

void test_linear_layer2() {
    cout << "Test linear layer case 2 ...\t";
    using RootLayer = InjectPolicy_t<LinearLayer, UpdatePolicy>;
    static_assert(RootLayer::isUpdate, "Test Error");
    static_assert(!RootLayer::isFeedbackOutput, "Test Error");

    RootLayer layer("root", 2, 3);

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
    std::map<std::string, Matrix<float, CPU>> params;
    layer.init(initializer, params);

    Matrix<float, CPU> i(1, 2);
    i.setValue(0, 0, 0.1f);
    i.setValue(0, 1, 0.2f);

    auto input = LayerIO::create().set<LayerIO>(i);
    auto out = evaluate(layer.feedForward(input).get<LayerIO>());
    assert(fabs(out(0, 0) - 0.75f) < 0.00001);
    assert(fabs(out(0, 1) - 0.91f) < 0.00001);
    assert(fabs(out(0, 2) - 1.07f) < 0.00001);

    Matrix<float, CPU> g(1, 3);
    g.setValue(0, 0, 0.1f);
    g.setValue(0, 1, 0.2f);
    g.setValue(0, 2, 0.3f);
    auto fbIn = LayerIO::create().set<LayerIO>(g);
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

            auto tmp = evaluate(dot(transpose(i), g));
            assert(tmp.rowNum() == info.rowNum());
            assert(tmp.colNum() == info.colNum());

            for (size_t i = 0; i < tmp.rowNum(); ++i) {
                for (size_t j = 0; j < tmp.colNum(); ++j) {
                    assert(fabs(info(i, j) - tmp(i, j)) < 0.0001f);
                }
            }
        } else if (w.rowNum() == b1.rowNum()) {
            bias_update_valid = true;
            for (size_t i = 0; i < info.rowNum(); ++i) {
                for (size_t j = 0; j < info.colNum(); ++j) {
                    assert(fabs(info(i, j) - g(i, j)) < 0.0001f);
                }
            }
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

void test_linear_layer3() {
    cout << "Test linear layer case 3 ...\t";
    using RootLayer =
        InjectPolicy_t<LinearLayer, UpdatePolicy, SubPolicyContainer<SubLayerOf<LinearLayer>::Weight, NoUpdatePolicy>>;
    // using Cont = PolicyContainer<UpdatePolicy, SubPolicyContainer<SubLayerOf<LinearLayer>::Weight, NoUpdatePolicy>>;
    // using PlanP = PlainPolicy_t<Cont, PolicyContainer<>>;
    // // using KeyType = RootLayer::PlainPolicies;
    // static_assert(!details::PolicySelect_t<FeedbackPolicy, PlanP>::isFeedbackOutput);
    // static_assert(details::PolicySelect_t<FeedbackPolicy, PlanP>::isUpdate);
    static_assert(!RootLayer::isFeedbackOutput, "Test Error");
    static_assert(RootLayer::isUpdate, "Test Error");
    //
    RootLayer layer("root", 2, 3);
    //
    Matrix<float, CPU> w1(2, 3);
    w1.setValue(0, 0, 0.1f);
    w1.setValue(1, 0, 0.2f);
    w1.setValue(0, 1, 0.3f);
    w1.setValue(1, 1, 0.4f);
    w1.setValue(0, 2, 0.5f);
    w1.setValue(1, 2, 0.6f);
    //
    Matrix<float, CPU> b1(1, 3);
    b1.setValue(0, 0, 0.7f);
    b1.setValue(0, 1, 0.8f);
    b1.setValue(0, 2, 0.9f);
    //
    auto initializer = make_initializer<float>();
    initializer.setMatrix("root-weight", w1);
    initializer.setMatrix("root-bias", b1);
    std::map<std::string, Matrix<float, CPU>> params;
    layer.init(initializer, params);
    //
    Matrix<float, CPU> i(1, 2);
    i.setValue(0, 0, 0.1f);
    i.setValue(0, 1, 0.2f);
    //
    auto input = LayerIO::create().set<LayerIO>(i);
    auto out = evaluate(layer.feedForward(input).get<LayerIO>());
    //
    assert(fabs(out(0, 0) - 0.75f) < 0.00001);
    assert(fabs(out(0, 1) - 0.91f) < 0.00001);
    assert(fabs(out(0, 2) - 1.07f) < 0.00001);
    //
    Matrix<float, CPU> g(1, 3);
    g.setValue(0, 0, 0.1f);
    g.setValue(0, 1, 0.2f);
    g.setValue(0, 2, 0.3f);
    auto fbIn = LayerIO::create().set<LayerIO>(g);
    auto out_grad = layer.feedBackward(fbIn);
    auto fbOut = out_grad.get<LayerIO>();
    static_assert(std::is_same_v<decltype(fbOut), details::NullParameter>, "Test error");
    //
    GradCollector<float, CPU> grad_collector;
    layer.gradCollect(grad_collector);
    assert(grad_collector.size() == 1);
    //
    auto w = (*grad_collector.begin()).m_weight;
    auto info = evaluate(collapse((*grad_collector.begin()).m_grad));
    assert(w == params.begin()->second);
    for (size_t i = 0; i < info.rowNum(); ++i) {
        for (size_t j = 0; j < info.colNum(); ++j) {
            assert(fabs(info(i, j) - g(i, j)) < 0.0001f);
        }
    }

    params.clear();
    layer.saveWeights(params);
    assert(params.find("root-weight") != params.end());
    assert(params.find("root-bias") != params.end());
    cout << "done" << endl;
}

void test_linear_layer4() {
    cout << "Test linear layer case 4 ...\t";
    using RootLayer =
        InjectPolicy_t<LinearLayer, UpdatePolicy, SubPolicyContainer<SubLayerOf<LinearLayer>::Bias, NoUpdatePolicy>>;
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
    layer.init(initializer, params);

    Matrix<float, CPU> i(1, 2);
    i.setValue(0, 0, 0.1f);
    i.setValue(0, 1, 0.2f);

    auto input = LayerIO::create().set<LayerIO>(i);
    auto out = evaluate(layer.feedForward(input).get<LayerIO>());

    assert(fabs(out(0, 0) - 0.75f) < 0.00001);
    assert(fabs(out(0, 1) - 0.91f) < 0.00001);
    assert(fabs(out(0, 2) - 1.07f) < 0.00001);

    Matrix<float, CPU> g(1, 3);
    g.setValue(0, 0, 0.1f);
    g.setValue(0, 1, 0.2f);
    g.setValue(0, 2, 0.3f);
    auto fbIn = LayerIO::create().set<LayerIO>(g);
    auto out_grad = layer.feedBackward(fbIn);
    auto fbOut = out_grad.get<LayerIO>();
    static_assert(std::is_same_v<decltype(fbOut), details::NullParameter>, "Test error");

    GradCollector<float, CPU> grad_collector;
    layer.gradCollect(grad_collector);
    assert(grad_collector.size() == 1);

    auto w = (*grad_collector.begin()).m_weight;
    assert(w == params.rbegin()->second);

    auto check = collapse(grad_collector.begin()->m_grad);
    auto check2 = dot(transpose(i), g);

    auto handle1 = check.evalRegister();
    auto handle2 = check2.evalRegister();
    EvalPlan<CPU>::eval();

    auto info = handle1.data();
    auto tmp = handle2.data();
    assert(tmp.rowNum() == info.rowNum());
    assert(tmp.colNum() == info.colNum());

    for (size_t i = 0; i < tmp.rowNum(); ++i) {
        for (size_t j = 0; j < tmp.colNum(); ++j) {
            assert(fabs(info(i, j) - tmp(i, j)) < 0.0001f);
        }
    }

    params.clear();
    layer.saveWeights(params);
    assert(params.find("root-weight") != params.end());
    assert(params.find("root-bias") != params.end());
    cout << "done" << endl;
}

void test_linear_layer5() {
    cout << "Test linear layer case 5 ...\t";
    using RootLayer = InjectPolicy_t<LinearLayer, SubPolicyContainer<SubLayerOf<LinearLayer>::Bias, UpdatePolicy>>;
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
    layer.init(initializer, params);

    Matrix<float, CPU> i(1, 2);
    i.setValue(0, 0, 0.1f);
    i.setValue(0, 1, 0.2f);

    auto input = LayerIO::create().set<LayerIO>(i);
    auto out = evaluate(layer.feedForward(input).get<LayerIO>());

    assert(fabs(out(0, 0) - 0.75f) < 0.00001);
    assert(fabs(out(0, 1) - 0.91f) < 0.00001);
    assert(fabs(out(0, 2) - 1.07f) < 0.00001);

    Matrix<float, CPU> g(1, 3);
    g.setValue(0, 0, 0.1f);
    g.setValue(0, 1, 0.2f);
    g.setValue(0, 2, 0.3f);
    auto fbIn = LayerIO::create().set<LayerIO>(g);
    auto out_grad = layer.feedBackward(fbIn);
    auto fbOut = out_grad.get<LayerIO>();
    static_assert(std::is_same_v<decltype(fbOut), details::NullParameter>, "Test error");

    GradCollector<float, CPU> grad_collector;
    layer.gradCollect(grad_collector);
    assert(grad_collector.size() == 1);

    auto w = (*grad_collector.begin()).m_weight;
    auto info = evaluate(collapse(grad_collector.begin()->m_grad));
    assert(w == params.begin()->second);
    for (size_t i = 0; i < info.rowNum(); ++i) {
        for (size_t j = 0; j < info.colNum(); ++j) {
            assert(fabs(info(i, j) - g(i, j)) < 0.0001f);
        }
    }

    params.clear();
    layer.saveWeights(params);
    assert(params.find("root-weight") != params.end());
    assert(params.find("root-bias") != params.end());
    cout << "done" << endl;
}

void test_linear_layer6() {
    cout << "Test linear layer case 6 ...\t";
    using RootLayer = InjectPolicy_t<LinearLayer, SubPolicyContainer<SubLayerOf<LinearLayer>::Weight, UpdatePolicy>>;
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
    layer.init(initializer, params);

    Matrix<float, CPU> i(1, 2);
    i.setValue(0, 0, 0.1f);
    i.setValue(0, 1, 0.2f);

    auto input = LayerIO::create().set<LayerIO>(i);
    auto out = evaluate(layer.feedForward(input).get<LayerIO>());

    assert(fabs(out(0, 0) - 0.75f) < 0.00001);
    assert(fabs(out(0, 1) - 0.91f) < 0.00001);
    assert(fabs(out(0, 2) - 1.07f) < 0.00001);

    Matrix<float, CPU> g(1, 3);
    g.setValue(0, 0, 0.1f);
    g.setValue(0, 1, 0.2f);
    g.setValue(0, 2, 0.3f);
    auto fbIn = LayerIO::create().set<LayerIO>(g);
    auto out_grad = layer.feedBackward(fbIn);
    auto fbOut = out_grad.get<LayerIO>();
    static_assert(std::is_same_v<decltype(fbOut), details::NullParameter>, "Test error");

    GradCollector<float, CPU> grad_collector;
    layer.gradCollect(grad_collector);
    assert(grad_collector.size() == 1);

    auto w = (*grad_collector.begin()).m_weight;
    assert(w == params.rbegin()->second);
    auto check1 = collapse((*grad_collector.begin()).m_grad);
    auto check2 = dot(transpose(i), g);

    auto handle1 = check1.evalRegister();
    auto handle2 = check2.evalRegister();
    EvalPlan<CPU>::eval();

    auto info = handle1.data();
    auto tmp = handle2.data();

    assert(tmp.rowNum() == info.rowNum());
    assert(tmp.colNum() == info.colNum());

    for (size_t i = 0; i < tmp.rowNum(); ++i) {
        for (size_t j = 0; j < tmp.colNum(); ++j) {
            assert(fabs(info(i, j) - tmp(i, j)) < 0.0001f);
        }
    }

    params.clear();
    layer.saveWeights(params);
    assert(params.find("root-weight") != params.end());
    assert(params.find("root-bias") != params.end());
    cout << "done" << endl;
}

int main() {
    std::cout << "Test linear layer ..." << std::endl;
    test_linear_layer1();
    test_linear_layer2();
    test_linear_layer3();
    test_linear_layer4();
    test_linear_layer5();
    test_linear_layer6();
    std::cout << "All tests passed!" << std::endl;
}