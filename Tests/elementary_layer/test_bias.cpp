#include <iostream>
#include <map>
#include <string>

#include <metann/data/data_device.hpp>
#include <metann/data/matrix.hpp>
#include <metann/layers/elementary/abs_layer.hpp>
#include <metann/layers/elementary/add_layer.hpp>
#include <metann/layers/elementary/bias_layer.hpp>
#include <metann/layers/elementary/mul_layer.hpp>
#include <metann/layers/fillers/constant_filler.hpp>
#include <metann/layers/grad_collector.hpp>
#include <metann/layers/initializer.hpp>
#include <metann/layers/interface_fun.hpp>
using namespace metann;
using std::cout;
using std::endl;

template <typename Elem>
inline auto gen_matrix(std::size_t r, std::size_t c, Elem start = 0, Elem scale = 1) {
    using namespace metann;
    Matrix<Elem, CPU> res(r, c);
    for (std::size_t i = 0; i < r; ++i) {
        for (std::size_t j = 0; j < c; ++j) {
            res.setValue(i, j, (Elem)(start * scale));
            start += 1.0f;
        }
    }
    return res;
}

template <typename TElem>
inline auto gen_batch_matrix(size_t r, size_t c, size_t d, float start = 0, float scale = 1) {
    using namespace metann;
    Batch<TElem, metann::CPU, CategoryTags::Matrix> res(d, r, c);
    for (size_t i = 0; i < r; ++i) {
        for (size_t j = 0; j < c; ++j) {
            for (size_t k = 0; k < d; ++k) {
                res.setValue(k, i, j, static_cast<TElem>(start * scale));
                start += 1.0f;
            }
        }
    }
    return res;
}

void test_bias_layer1() {
    cout << "Test bias layer case 1 ...\t";
    using RootLayer = InjectPolicy_t<BiasLayer>;
    static_assert(!RootLayer::isUpdate, "Test Error");
    static_assert(!RootLayer::isFeedbackOutput, "Test Error");

    RootLayer layer("root", 2, 1);

    auto initializer = make_initializer<float>();
    auto weight = gen_matrix<float>(2, 1, 1, 0.1f);
    initializer.setMatrix("root", weight);
    std::map<std::string, Matrix<float, CPU>> params;
    layer.init(initializer, params);

    auto input = gen_matrix<float>(2, 1, 0.5f, -0.1f);
    auto bi = LayerIO::create().set<LayerIO>(input);

    layer_neutral_invariant(layer);
    auto out = layer.feedForward(bi);
    auto res = evaluate(out.get<LayerIO>());
    assert(fabs(res(0, 0) - input(0, 0) - weight(0, 0)) < 0.001);
    assert(fabs(res(1, 0) - input(1, 0) - weight(1, 0)) < 0.001);

    auto fbIn = LayerIO::create();
    auto out_grad = layer.feedBackward(fbIn);
    auto fbOut = out_grad.get<LayerIO>();
    static_assert(std::is_same_v<decltype(fbOut), details::NullParameter>, "Test error");

    params.clear();
    layer.saveWeights(params);
    assert(params.find("root") != params.end());

    layer_neutral_invariant(layer);
    cout << "done" << endl;
}

void test_bias_layer2() {
    cout << "Test bias layer case 2 ...\t";
    using RootLayer = InjectPolicy_t<BiasLayer>;
    static_assert(!RootLayer::isUpdate, "Test Error");
    static_assert(!RootLayer::isFeedbackOutput, "Test Error");

    RootLayer layer("root", 1, 2);

    auto initializer = make_initializer<float>();
    auto weight = gen_matrix<float>(1, 2, 1, 0.1f);
    initializer.setMatrix("root", weight);
    std::map<std::string, Matrix<float, CPU>> params;
    layer.init(initializer, params);

    auto input = gen_matrix<float>(1, 2, 0.5f, -0.1f);
    auto bi = LayerIO::create().set<LayerIO>(input);

    layer_neutral_invariant(layer);
    auto out = layer.feedForward(bi);
    auto res = evaluate(out.get<LayerIO>());
    assert(fabs(res(0, 0) - input(0, 0) - weight(0, 0)) < 0.001);
    assert(fabs(res(0, 1) - input(0, 1) - weight(0, 1)) < 0.001);

    auto fbIn = LayerIO::create();
    auto out_grad = layer.feedBackward(fbIn);
    auto fbOut = out_grad.get<LayerIO>();
    static_assert(std::is_same_v<decltype(fbOut), details::NullParameter>, "Test error");

    layer_neutral_invariant(layer);

    params.clear();
    layer.saveWeights(params);
    assert(params.find("root") != params.end());

    cout << "done" << endl;
}

void test_bias_layer3() {
    cout << "Test bias layer case 3 ...\t";
    using RootLayer = InjectPolicy_t<BiasLayer, UpdatePolicy>;
    static_assert(RootLayer::isUpdate, "Test Error");
    static_assert(!RootLayer::isFeedbackOutput, "Test Error");

    RootLayer layer("root", 2, 1);

    Matrix<float, CPU> w(2, 1);
    w.setValue(0, 0, -0.48f);
    w.setValue(1, 0, -0.13f);

    auto initializer = make_initializer<float>();
    initializer.setMatrix("root", w);
    std::map<std::string, Matrix<float, CPU>> params;
    layer.init(initializer, params);

    Matrix<float, CPU> input(2, 1);
    input.setValue(0, 0, -0.27f);
    input.setValue(1, 0, -0.41f);

    auto bi = LayerIO::create().set<LayerIO>(input);

    layer_neutral_invariant(layer);
    auto out = layer.feedForward(bi);
    auto res = evaluate(out.get<LayerIO>());
    assert(fabs(res(0, 0) + 0.27f + 0.48f) < 0.001);
    assert(fabs(res(1, 0) + 0.41f + 0.13f) < 0.001);

    Matrix<float, CPU> g(2, 1);
    g.setValue(0, 0, -0.0495f);
    g.setValue(1, 0, -0.0997f);

    auto fbIn = LayerIO::create().set<LayerIO>(g);
    auto out_grad = layer.feedBackward(fbIn);

    GradCollector<float, CPU> grad_collector;
    layer.gradCollect(grad_collector);
    assert(grad_collector.size() == 1);

    auto gcit = grad_collector.begin();
    auto claps = collapse(gcit->m_grad);

    auto handle1 = gcit->m_weight.evalRegister();
    auto handle2 = claps.evalRegister();
    EvalPlan<CPU>::eval();

    auto w1 = handle1.data();
    auto g1 = handle2.data();

    assert(fabs(w1(0, 0) + 0.48f) < 0.001);
    assert(fabs(w1(1, 0) + 0.13f) < 0.001);
    assert(fabs(g1(0, 0) + 0.0495f) < 0.001);
    assert(fabs(g1(1, 0) + 0.0997f) < 0.001);
    layer_neutral_invariant(layer);

    params.clear();
    layer.saveWeights(params);
    assert(params.find("root") != params.end());

    cout << "done" << endl;
}

void test_bias_layer4() {
    cout << "Test bias layer case 4 ...\t";
    using RootLayer = InjectPolicy_t<BiasLayer, UpdatePolicy, FeedbackOutputPolicy>;
    static_assert(RootLayer::isUpdate, "Test Error");
    static_assert(RootLayer::isFeedbackOutput, "Test Error");

    RootLayer layer("root", 2, 1);

    Matrix<float, CPU> w(2, 1);
    w.setValue(0, 0, -0.48f);
    w.setValue(1, 0, -0.13f);

    auto initializer = make_initializer<float>();
    initializer.setMatrix("root", w);
    std::map<std::string, Matrix<float, CPU>> params;
    layer.init(initializer, params);

    Matrix<float, CPU> input(2, 1);
    input.setValue(0, 0, -0.27f);
    input.setValue(1, 0, -0.41f);

    auto bi = LayerIO::create().set<LayerIO>(input);

    layer_neutral_invariant(layer);
    auto out = layer.feedForward(bi);
    auto res = evaluate(out.get<LayerIO>());
    assert(fabs(res(0, 0) + 0.27f + 0.48f) < 0.001);
    assert(fabs(res(1, 0) + 0.41f + 0.13f) < 0.001);

    Matrix<float, CPU> g(2, 1);
    g.setValue(0, 0, -0.0495f);
    g.setValue(1, 0, -0.0997f);

    auto fbIn = LayerIO::create().set<LayerIO>(g);
    auto out_grad = layer.feedBackward(fbIn);
    auto fbOut = evaluate(out_grad.get<LayerIO>());

    assert(fabs(fbOut(0, 0) + 0.0495f) < 0.001);
    assert(fabs(fbOut(1, 0) + 0.0997f) < 0.001);

    GradCollector<float, CPU> grad_collector;
    layer.gradCollect(grad_collector);
    assert(grad_collector.size() == 1);

    auto gcit = grad_collector.begin();
    auto claps = collapse(gcit->m_grad);

    auto handle1 = gcit->m_weight.evalRegister();
    auto handle2 = claps.evalRegister();
    EvalPlan<CPU>::eval();

    auto w1 = handle1.data();
    auto g1 = handle2.data();

    assert(fabs(w1(0, 0) + 0.48f) < 0.001);
    assert(fabs(w1(1, 0) + 0.13f) < 0.001);
    assert(fabs(g1(0, 0) + 0.0495f) < 0.001);
    assert(fabs(g1(1, 0) + 0.0997f) < 0.001);
    layer_neutral_invariant(layer);

    params.clear();
    layer.saveWeights(params);
    assert(params.find("root") != params.end());

    cout << "done" << endl;
}

void test_bias_layer5() {
    cout << "Test bias layer case 5 ...\t";
    using RootLayer = InjectPolicy_t<BiasLayer, UpdatePolicy, FeedbackOutputPolicy>;
    static_assert(RootLayer::isUpdate, "Test Error");
    static_assert(RootLayer::isFeedbackOutput, "Test Error");

    RootLayer layer("root", 2, 1);

    Matrix<float, CPU> w(2, 1);
    w.setValue(0, 0, -0.48f);
    w.setValue(1, 0, -0.13f);

    auto initializer = make_initializer<float>();
    initializer.setMatrix("root", w);
    std::map<std::string, Matrix<float, CPU>> params;
    layer.init(initializer, params);

    Matrix<float, CPU> input(2, 1);
    input.setValue(0, 0, -0.27f);
    input.setValue(1, 0, -0.41f);

    auto bi = LayerIO::create().set<LayerIO>(input);

    layer_neutral_invariant(layer);
    auto out = layer.feedForward(bi);
    auto res = evaluate(out.get<LayerIO>());
    assert(fabs(res(0, 0) + 0.27f + 0.48f) < 0.001);
    assert(fabs(res(1, 0) + 0.41f + 0.13f) < 0.001);

    input = Matrix<float, CPU>(2, 1);
    input.setValue(0, 0, 1.27f);
    input.setValue(1, 0, 2.41f);

    bi = LayerIO::create().set<LayerIO>(input);

    out = layer.feedForward(bi);
    res = evaluate(out.get<LayerIO>());
    assert(fabs(res(0, 0) - 1.27f + 0.48f) < 0.001);
    assert(fabs(res(1, 0) - 2.41f + 0.13f) < 0.001);

    Matrix<float, CPU> g(2, 1);
    g.setValue(0, 0, -0.0495f);
    g.setValue(1, 0, -0.0997f);

    auto fbIn = LayerIO::create().set<LayerIO>(g);
    auto out_grad = layer.feedBackward(fbIn);
    auto fbOut = evaluate(out_grad.get<LayerIO>());

    assert(fabs(fbOut(0, 0) + 0.0495f) < 0.001);
    assert(fabs(fbOut(1, 0) + 0.0997f) < 0.001);

    g = Matrix<float, CPU>(2, 1);
    g.setValue(0, 0, 1.0495f);
    g.setValue(1, 0, 2.3997f);

    fbIn = LayerIO::create().set<LayerIO>(g);
    out_grad = layer.feedBackward(fbIn);
    fbOut = evaluate(out_grad.get<LayerIO>());

    assert(fabs(fbOut(0, 0) - 1.0495f) < 0.001);
    assert(fabs(fbOut(1, 0) - 2.3997f) < 0.001);

    GradCollector<float, CPU> grad_collector;
    layer.gradCollect(grad_collector);
    assert(grad_collector.size() == 1);

    auto gcit = grad_collector.begin();
    auto claps = collapse(gcit->m_grad);

    auto handle1 = gcit->m_weight.evalRegister();
    auto handle2 = claps.evalRegister();
    EvalPlan<CPU>::eval();

    auto w1 = handle1.data();
    auto g1 = handle2.data();

    assert(fabs(w1(0, 0) + 0.48f) < 0.001);
    assert(fabs(w1(1, 0) + 0.13f) < 0.001);
    assert(fabs(g1(0, 0) + 0.0495f - 1.0495f) < 0.001);
    assert(fabs(g1(1, 0) + 0.0997f - 2.3997f) < 0.001);
    layer_neutral_invariant(layer);

    params.clear();
    layer.saveWeights(params);
    assert(params.find("root") != params.end());

    cout << "done" << endl;
}

void test_bias_layer6() {
    cout << "Test bias layer case 6 ...\t";
    using RootLayer = InjectPolicy_t<BiasLayer, UpdatePolicy, FeedbackOutputPolicy>;
    static_assert(RootLayer::isUpdate, "Test Error");
    static_assert(RootLayer::isFeedbackOutput, "Test Error");

    RootLayer layer("root", 400);

    auto initializer =
        make_initializer<float, InitializerIs<struct ConstantTag>>().setFiller<ConstantTag>(ConstantFiller{0});
    std::map<std::string, Matrix<float, CPU>> loader;
    layer.init(initializer, loader);

    assert(loader.size() == 1);

    auto& val = loader.begin()->second;

    for (size_t i = 0; i < val.rowNum(); ++i) {
        for (size_t j = 0; j < val.colNum(); ++j) {
            assert(fabs(val(i, j)) < 0.0001);
        }
    }
    cout << "done" << endl;
}

void test_bias_layer7() {
    cout << "Test bias layer case 7 ...\t";
    using RootLayer = InjectPolicy_t<BiasLayer, UpdatePolicy, FeedbackOutputPolicy>;
    static_assert(RootLayer::isUpdate, "Test Error");
    static_assert(RootLayer::isFeedbackOutput, "Test Error");

    RootLayer layer("root", 400);

    auto initializer =
        make_initializer<float, InitializerIs<struct ConstantTag>>().setFiller<ConstantTag>(ConstantFiller{1.5});
    std::map<std::string, Matrix<float, CPU>> loader;
    layer.init(initializer, loader);

    assert(loader.size() == 1);

    auto& val = loader.begin()->second;

    for (size_t i = 0; i < val.rowNum(); ++i) {
        for (size_t j = 0; j < val.colNum(); ++j) {
            assert(fabs(val(i, j) - 1.5) < 0.0001);
        }
    }
    cout << "done" << endl;
}

int main() {
    std::cout << "Test bias layer ..." << std::endl;
    test_bias_layer1();
    test_bias_layer2();
    test_bias_layer3();
    test_bias_layer4();
    test_bias_layer5();
    test_bias_layer6();
    test_bias_layer7();
    std::cout << "All tests passed!" << std::endl;
}