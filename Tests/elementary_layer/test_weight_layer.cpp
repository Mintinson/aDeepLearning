//
// Created by asus on 2025/2/13.
//
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <map>
#include <string>
#include <type_traits>
#include <vector>

#include <metann/data/batch.hpp>
#include <metann/data/data_category.hpp>
#include <metann/data/data_device.hpp>
#include <metann/data/matrix.hpp>
#include <metann/eval/facilities.hpp>
#include <metann/layers/elementary/weight_layer.hpp>
#include <metann/layers/fillers/uniform_filler.hpp>
#include <metann/layers/grad_collector.hpp>
#include <metann/layers/initializer.hpp>
#include <metann/layers/interface_fun.hpp>
#include <metann/layers/layer_io.hpp>
#include <metann/layers/policies/init_policy.hpp>
#include <metann/layers/policies/update_policy.hpp>
#include <metann/operators/binary_operators.hpp>
#include <metann/operators/unary_operators.hpp>
#include <metann/policy/policy.hpp>
#include <metann/utils/vartype_dict.hpp>

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

void test_weight_layer1() {
    cout << "Test weight layer case 1 ...\t";
    using RootLayer = InjectPolicy_t<WeightLayer>;
    static_assert(!RootLayer::isFeedbackOutput, "Test Error");
    static_assert(!RootLayer::isUpdate, "Test Error");

    RootLayer layer("root", 1, 2);

    Matrix<float, CPU> w(1, 2);
    w.setValue(0, 0, -0.27f);
    w.setValue(0, 1, -0.41f);

    auto initializer = make_initializer<float>();
    initializer.setMatrix("root", w);
    std::map<std::string, Matrix<float, CPU>> params;
    layer.init(initializer, params);

    Matrix<float, CPU> input(1, 1);
    input.setValue(0, 0, 1);

    layer_neutral_invariant(layer);
    auto wi = LayerIO::create().set<LayerIO>(input);

    auto out = layer.feedForward(wi);
    auto res = evaluate(out.get<LayerIO>());
    assert(fabs(res(0, 0) + 0.27f) < 0.001);
    assert(fabs(res(0, 1) + 0.41f) < 0.001);

    auto out_grad = layer.feedBackward(LayerIO::create());
    auto fbOut = out_grad.get<LayerIO>();
    static_assert(std::is_same_v<decltype(fbOut), details::NullParameter>, "Test error");

    params.clear();
    layer.saveWeights(params);
    assert(params.find("root") != params.end());

    layer_neutral_invariant(layer);
    cout << "done" << endl;
}

void test_weight_layer2() {
    cout << "Test weight layer case 2 ...\t";
    using RootLayer = InjectPolicy_t<WeightLayer, UpdatePolicy>;
    static_assert(!RootLayer::isFeedbackOutput, "Test Error");
    static_assert(RootLayer::isUpdate, "Test Error");

    RootLayer layer("root", 1, 2);

    Matrix<float, CPU> w(1, 2);
    w.setValue(0, 0, -0.27f);
    w.setValue(0, 1, -0.41f);

    auto initializer = make_initializer<float>();
    initializer.setMatrix("root", w);
    std::map<std::string, Matrix<float, CPU>> params;
    layer.init(initializer, params);

    Matrix<float, CPU> input(1, 1);
    input.setValue(0, 0, 0.1f);

    auto wi = LayerIO::create().set<LayerIO>(input);

    layer_neutral_invariant(layer);
    auto out = layer.feedForward(wi);
    auto res = evaluate(out.get<LayerIO>());
    assert(fabs(res(0, 0) + 0.027f) < 0.001);
    assert(fabs(res(0, 1) + 0.041f) < 0.001);

    Matrix<float, CPU> g(1, 2);
    g.setValue(0, 0, -0.0495f);
    g.setValue(0, 1, -0.0997f);
    auto out_grad = layer.feedBackward(LayerIO::create().set<LayerIO>(g));
    auto fbOut = out_grad.get<LayerIO>();
    static_assert(std::is_same_v<decltype(fbOut), details::NullParameter>, "Test error");

    GradCollector<float, CPU> grad_collector;
    layer.gradCollect(grad_collector);
    assert(grad_collector.size() == 1);

    auto w1 = (*grad_collector.begin()).m_weight;
    auto info_g = evaluate(collapse((*grad_collector.begin()).m_grad));

    assert(fabs(w1(0, 0) + 0.27f) < 0.001);
    assert(fabs(w1(0, 1) + 0.41f) < 0.001);

    assert(fabs(info_g(0, 0) + 0.00495f) < 0.001);
    assert(fabs(info_g(0, 1) + 0.00997f) < 0.001);

    params.clear();
    layer.saveWeights(params);
    assert(params.find("root") != params.end());
    layer_neutral_invariant(layer);

    cout << "done" << endl;
}

void test_weight_layer3() {
    cout << "Test weight layer case 3 ...\t";
    using RootLayer = InjectPolicy_t<WeightLayer, UpdatePolicy, FeedbackOutputPolicy>;
    static_assert(RootLayer::isFeedbackOutput, "Test Error");
    static_assert(RootLayer::isUpdate, "Test Error");

    RootLayer layer("root", 2, 2);

    Matrix<float, CPU> w(2, 2);
    w.setValue(0, 0, 1.1f);
    w.setValue(0, 1, 3.1f);
    w.setValue(1, 0, 0.1f);
    w.setValue(1, 1, 1.17f);

    auto initializer = make_initializer<float>();
    initializer.setMatrix("root", w);
    std::map<std::string, Matrix<float, CPU>> params;
    layer.init(initializer, params);

    Matrix<float, CPU> input(1, 2);
    input.setValue(0, 0, 0.999f);
    input.setValue(0, 1, 0.0067f);

    auto wi = LayerIO::create().set<LayerIO>(input);

    layer_neutral_invariant(layer);
    auto out = layer.feedForward(wi);
    auto res = evaluate(out.get<LayerIO>());
    assert(fabs(res(0, 0) - 1.0996f) < 0.001);
    assert(fabs(res(0, 1) - 3.1047f) < 0.001);

    Matrix<float, CPU> g(1, 2);
    g.setValue(0, 0, 0.0469f);
    g.setValue(0, 1, -0.0394f);
    auto out_grad = layer.feedBackward(LayerIO::create().set<LayerIO>(g));
    auto fbOut = evaluate(out_grad.get<LayerIO>());
    assert(fabs(fbOut(0, 0) + 0.07055) < 0.001);
    assert(fabs(fbOut(0, 1) + 0.041408f) < 0.001);

    GradCollector<float, CPU> grad_collector;
    layer.gradCollect(grad_collector);
    assert(grad_collector.size() == 1);

    auto w1 = grad_collector.begin()->m_weight;
    auto info_g = evaluate(collapse(grad_collector.begin()->m_grad));

    assert(fabs(w1(0, 0) - 1.1) < 0.001);
    assert(fabs(w1(1, 0) - 0.1) < 0.001);
    assert(fabs(w1(0, 1) - 3.1) < 0.001);
    assert(fabs(w1(1, 1) - 1.17) < 0.001);

    assert(fabs(info_g(0, 0) - 0.0468531) < 0.001);
    assert(fabs(info_g(1, 0) - 0.00031423) < 0.001);
    assert(fabs(info_g(0, 1) + 0.0393606) < 0.001);
    assert(fabs(info_g(1, 1) + 0.00026398) < 0.001);

    params.clear();
    layer.saveWeights(params);
    assert(params.find("root") != params.end());

    layer_neutral_invariant(layer);
    cout << "done" << endl;
}

void test_weight_layer4() {
    cout << "Test weight layer case 4 ...\t";
    using RootLayer = InjectPolicy_t<WeightLayer, UpdatePolicy, FeedbackOutputPolicy>;
    static_assert(RootLayer::isFeedbackOutput, "Test Error");
    static_assert(RootLayer::isUpdate, "Test Error");

    RootLayer layer("root", 8, 4);

    auto w = gen_matrix<float>(8, 4, 0.1f, 0.5f);

    auto initializer = make_initializer<float>();
    initializer.setMatrix("root", w);
    std::map<std::string, Matrix<float, CPU>> params;
    layer.init(initializer, params);

    std::vector<Matrix<float, CPU>> op_in;
    std::vector<Matrix<float, CPU>> op_grad;

    for (int loop_count = 0; loop_count < 10; ++loop_count) {
        auto input = gen_matrix<float>(1, 8, loop_count * 0.1f, -0.3f);
        op_in.push_back(input);

        auto out = layer.feedForward(LayerIO::create().set<LayerIO>(input));
        auto check = dot(input, w);

        auto handle1 = out.get<LayerIO>().evalRegister();
        auto handle2 = check.evalRegister();
        EvalPlan<CPU>::eval();

        auto res = handle1.data();
        assert(res.rowNum() == 1);
        assert(res.colNum() == 4);
        auto c = handle2.data();

        for (size_t i = 0; i < 4; ++i) {
            assert(fabs(res(0, i) - c(0, i)) <= 0.0001f);
        }
    }

    for (int loop_count = 9; loop_count >= 0; --loop_count) {
        auto grad = gen_matrix<float>(1, 4, loop_count * 0.2f, -0.1f);
        op_grad.push_back(grad);
        auto out_grad = layer.feedBackward(LayerIO::create().set<LayerIO>(grad));
        auto check = dot(grad, transpose(w));

        auto handle1 = out_grad.get<LayerIO>().evalRegister();
        auto handle2 = check.evalRegister();
        EvalPlan<CPU>::eval();

        auto fbOut = handle1.data();
        auto aimFbout = handle2.data();
        assert(fbOut.rowNum() == 1);
        assert(fbOut.colNum() == 8);

        for (size_t i = 0; i < 8; ++i) {
            assert(fabs(fbOut(0, i) - aimFbout(0, i)) < 0.0001);
        }
    }
    std::reverse(op_grad.begin(), op_grad.end());

    GradCollector<float, CPU> grad_collector;
    layer.gradCollect(grad_collector);
    assert(grad_collector.size() == 1);

    auto w1 = grad_collector.begin()->m_weight;
    auto aim = evaluate(dot(transpose(op_in[0]), op_grad[0]));
    for (int loop_count = 1; loop_count < 10; ++loop_count) {
        aim = evaluate(aim + dot(transpose(op_in[loop_count]), op_grad[loop_count]));
    }

    auto info_g = evaluate(collapse(grad_collector.begin()->m_grad));

    for (size_t i = 0; i < 8; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            assert(fabs(aim(i, j) - info_g(i, j)) < 0.0001f);
        }
    }

    params.clear();
    layer.saveWeights(params);
    assert(params.find("root") != params.end());

    layer_neutral_invariant(layer);
    cout << "done" << endl;
}

void test_weight_layer5() {
    cout << "Test weight layer case 5 ...\t";
    using RootLayer = InjectPolicy_t<WeightLayer, UpdatePolicy, FeedbackOutputPolicy>;
    RootLayer layer("root", 80, 40);

    auto initializer =
        make_initializer<float, InitializerIs<struct UniformTag>>().setFiller<UniformTag>(UniformFiller{-1, 1});
    std::map<std::string, Matrix<float, CPU>> loader;
    layer_init(layer, initializer, loader);

    assert(loader.size() == 1);

    const auto& val = loader.begin()->second;

    float mean = 0;
    for (size_t i = 0; i < val.rowNum(); ++i) {
        for (size_t j = 0; j < val.colNum(); ++j) {
            mean += val(i, j);
        }
    }
    mean /= val.rowNum() * val.colNum();

    float var = 0;
    for (size_t i = 0; i < val.rowNum(); ++i) {
        for (size_t j = 0; j < val.colNum(); ++j) {
            var += (val(i, j) - mean) * (val(i, j) - mean);
        }
    }
    var /= val.rowNum() * val.colNum();

    // should be about 0, 0.333
    cout << "mean-delta = " << fabs(mean) << " Variance-delta = " << fabs(var - 0.333) << ' ';

    cout << "done" << endl;
}

void test_weight_layer6() {
    cout << "Test weight layer case 6 ...\t";
    using RootLayer = InjectPolicy_t<WeightLayer, UpdatePolicy, FeedbackOutputPolicy>;
    RootLayer layer("root", 400, 200);

    auto initializer = make_initializer<float, WeightInitializerIs<struct UniformTag>>().setFiller<UniformTag>(
        UniformFiller{-1.5, 1.5});
    std::map<std::string, Matrix<float, CPU>> loader;
    layer.init(initializer, loader);

    assert(loader.size() == 1);

    auto& val = loader.begin()->second;

    float mean = 0;
    for (size_t i = 0; i < val.rowNum(); ++i) {
        for (size_t j = 0; j < val.colNum(); ++j) {
            mean += val(i, j);
        }
    }
    mean /= val.rowNum() * val.colNum();

    float var = 0;
    for (size_t i = 0; i < val.rowNum(); ++i) {
        for (size_t j = 0; j < val.colNum(); ++j) {
            var += (val(i, j) - mean) * (val(i, j) - mean);
        }
    }
    var /= val.rowNum() * val.colNum();
    // should be about 0, 0.75
    cout << "mean-delta = " << fabs(mean) << " Variance-delta = " << fabs(var - 0.75) << ' ';

    cout << "done" << endl;
}

int main() {
    std::cout << "Test weight layer ...\n";
    test_weight_layer1();
    test_weight_layer2();
    test_weight_layer3();
    test_weight_layer4();
    test_weight_layer5();
    test_weight_layer6();
    std::cout << "All tests passed.\n";
}
