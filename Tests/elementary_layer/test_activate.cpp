//
// Created by asus on 2025/2/13.
//
#include <iostream>
#include <metann/layers/elementary/sigmoid_layer.hpp>
#include <metann/layers/grad_collector.hpp>
#include <metann/layers/initializer.hpp>
#include <metann/layers/interface_fun.hpp>
#include <metann/layers/elementary/softmax_layer.hpp>
#include <metann/layers/elementary/tanh_layer.hpp>

using namespace metann;
using std::cout;
using std::endl;

template <typename Elem>
inline auto gen_matrix(std::size_t r, std::size_t c, Elem start = 0, Elem scale = 1)
{
    using namespace metann;
    Matrix<Elem, CPU> res(r, c);
    for (std::size_t i = 0; i < r; ++i)
    {
        for (std::size_t j = 0; j < c; ++j)
        {
            res.setValue(i, j, (Elem)(start * scale));
            start += 1.0f;
        }
    }
    return res;
}

template <typename TElem>
inline auto gen_batch_matrix(size_t r, size_t c, size_t d, float start = 0, float scale = 1)
{
    using namespace metann;
    Batch<TElem, metann::CPU, CategoryTags::Matrix> res(d, r, c);
    for (size_t i = 0; i < r; ++i)
    {
        for (size_t j = 0; j < c; ++j)
        {
            for (size_t k = 0; k < d; ++k)
            {
                res.setValue(k, i, j, static_cast<TElem>(start * scale));
                start += 1.0f;
            }
        }
    }
    return res;
}

void test_sigmoid_layer1()
{
    cout << "Test sigmoid layer case 1 ...\t";
    using RootLayer = InjectPolicy_t<SigmoidLayer>;
    static_assert(!RootLayer::isFeedbackOutput, "Test Error");
    static_assert(!RootLayer::isUpdate, "Test Error");

    RootLayer layer;

    Matrix<float, CPU> in(2, 1);
    in.setValue(0, 0, -0.27f);
    in.setValue(1, 0, -0.41f);

    auto input = LayerIO::create().set<LayerIO>(in);

    layer_neutral_invariant(layer);
    auto out = layer.feedForward(input);
    auto res = evaluate(out.get<LayerIO>());
    assert(fabs(res(0, 0) - (1 / (1 + exp(0.27f)))) < 0.001);
    assert(fabs(res(1, 0) - (1 / (1 + exp(0.41f)))) < 0.001);

    details::NullParameter fbIn;
    auto out_grad = layer.feedBackward(fbIn);
    auto fb1 = out_grad.get<LayerIO>();
    static_assert(std::is_same_v<decltype(fb1), details::NullParameter>, "Test error");

    layer_neutral_invariant(layer);
    cout << "done" << endl;
}

void test_sigmoid_layer2()
{
    cout << "Test sigmoid layer case 2 ...\t";
    using RootLayer = InjectPolicy_t<SigmoidLayer, FeedbackOutputPolicy>;
    static_assert(RootLayer::isFeedbackOutput, "Test Error");
    static_assert(!RootLayer::isUpdate, "Test Error");

    RootLayer layer;

    Matrix<float, CPU> in(2, 1);
    in.setValue(0, 0, -0.27f);
    in.setValue(1, 0, -0.41f);

    auto input = LayerIO::create().set<LayerIO>(in);

    layer_neutral_invariant(layer);
    auto out = layer.feedForward(input);
    auto res = evaluate(out.get<LayerIO>());
    assert(fabs(res(0, 0) - (1 / (1 + exp(0.27f)))) < 0.001);
    assert(fabs(res(1, 0) - (1 / (1 + exp(0.41f)))) < 0.001);

    Matrix<float, CPU> grad(2, 1);
    grad.setValue(0, 0, 0.1f);
    grad.setValue(1, 0, 0.3f);

    auto out_grad = layer.feedBackward(LayerIO::create().set<LayerIO>(grad));
    auto fb = evaluate(out_grad.get<LayerIO>());
    assert(fabs(fb(0, 0) - 0.1f * exp(0.27f) / (1 + exp(0.27f)) / (1 + exp(0.27f))) < 0.001);
    assert(fabs(fb(1, 0) - 0.3f * exp(0.41f) / (1 + exp(0.41f)) / (1 + exp(0.41f))) < 0.001);

    layer_neutral_invariant(layer);
    cout << "done" << endl;
}

void test_sigmoid_layer3()
{
    cout << "Test sigmoid layer case 3 ...\t";
    using RootLayer = InjectPolicy_t<SigmoidLayer, FeedbackOutputPolicy>;
    static_assert(RootLayer::isFeedbackOutput, "Test Error");
    static_assert(!RootLayer::isUpdate, "Test Error");

    RootLayer layer;

    std::vector<Matrix<float, CPU>> op;

    layer_neutral_invariant(layer);
    for (size_t loop_count = 1; loop_count < 10; ++loop_count)
    {
        auto in = gen_matrix<float>(loop_count, 3, 0.1f, 0.13f);

        op.push_back(in);

        auto input = LayerIO::create().set<LayerIO>(in);

        auto out = layer.feedForward(input);
        auto res = evaluate(out.get<LayerIO>());
        assert(res.rowNum() == loop_count);
        assert(res.colNum() == 3);
        for (size_t i = 0; i < loop_count; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                float aim = 1 / (1 + exp(-in(i, j)));
                assert(fabs(res(i, j) - aim) < 0.0001);
            }
        }
    }

    for (size_t loop_count = 9; loop_count >= 1; --loop_count)
    {
        auto grad = gen_matrix<float>(loop_count, 3, 2, 1.1f);
        auto out_grad = layer.feedBackward(LayerIO::create().set<LayerIO>(grad));

        auto fb = evaluate(out_grad.get<LayerIO>());

        auto in = op.back();
        op.pop_back();
        for (size_t i = 0; i < loop_count; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                float aim = exp(-in(i, j)) / (1 + exp(-in(i, j))) / (1 + exp(-in(i, j)));
                assert(fabs(fb(i, j) - grad(i, j) * aim) < 0.00001f);
            }
        }
    }

    layer_neutral_invariant(layer);

    cout << "done" << endl;
}

void test_softmax_layer1()
{
    cout << "Test softmax layer case 1 ...\t";
    using RootLayer = InjectPolicy_t<SoftmaxLayer, FeedbackOutputPolicy>;
    static_assert(RootLayer::isFeedbackOutput, "Test Error");
    static_assert(!RootLayer::isUpdate, "Test Error");

    RootLayer layer;

    Matrix<float, CPU> in(1, 2);
    in.setValue(0, 0, -0.27f);
    in.setValue(0, 1, -0.41f);

    auto input = LayerIO::create().set<LayerIO>(in);

    layer_neutral_invariant(layer);

    auto out = layer.feedForward(input);
    auto check = vec_softmax(in);

    auto handle1 = out.get<LayerIO>().evalRegister();
    auto handle2 = check.evalRegister();
    EvalPlan<CPU>::eval();

    auto res = handle1.data();
    auto c = handle2.data();

    assert(fabs(res(0, 0) - c(0, 0)) < 0.001);
    assert(fabs(res(0, 1) - c(0, 1)) < 0.001);

    Matrix<float, CPU> grad(1, 2);
    grad.setValue(0, 0, 0.1f);
    grad.setValue(0, 1, 0.3f);

    auto out_grad = layer.feedBackward(LayerIO::create().set<LayerIO>(grad));
    auto fb = evaluate(out_grad.get<LayerIO>());

    c = evaluate(softmax_derivative(grad, c));
    assert(fabs(fb(0, 0) - c(0, 0)) < 0.001);
    assert(fabs(fb(0, 1) - c(0, 1)) < 0.001);

    layer_neutral_invariant(layer);

    cout << "done" << endl;
}

void test_softmax_layer2()
{
    cout << "Test softmax layer case 2 ...\t";
    using RootLayer = InjectPolicy_t<SoftmaxLayer, FeedbackOutputPolicy>;
    static_assert(RootLayer::isFeedbackOutput, "Test Error");
    static_assert(!RootLayer::isUpdate, "Test Error");

    RootLayer layer;

    std::vector<Matrix<float, CPU>> op;

    layer_neutral_invariant(layer);
    for (size_t loop_count = 1; loop_count < 10; ++loop_count)
    {
        auto in = gen_matrix<float>(1, loop_count, 0.1f, 0.13f);

        auto input = LayerIO::create().set<LayerIO>(in);

        auto out = layer.feedForward(input);
        auto check = vec_softmax(in);

        auto handle1 = out.get<LayerIO>().evalRegister();
        auto handle2 = check.evalRegister();
        EvalPlan<CPU>::eval();

        auto res = handle1.data();
        auto c = handle2.data();

        op.push_back(c);
        for (size_t i = 0; i < loop_count; ++i)
        {
            assert(fabs(res(0, i) - c(0, i)) < 0.0001);
        }
    }

    for (size_t loop_count = 9; loop_count >= 1; --loop_count)
    {
        auto grad = gen_matrix<float>(1, loop_count, 1.3f, 1.1f);
        auto out_grad = layer.feedBackward(LayerIO::create().set<LayerIO>(grad));
        auto check = softmax_derivative(grad, op.back());

        auto handle1 = out_grad.get<LayerIO>().evalRegister();
        auto handle2 = check.evalRegister();
        EvalPlan<CPU>::eval();

        auto fb = handle1.data();
        auto c = handle2.data();
        op.pop_back();

        for (size_t i = 0; i < loop_count; ++i)
        {
            assert(fabs(fb(0, i) - c(0, i)) < 0.0001);
        }
    }

    layer_neutral_invariant(layer);

    cout << "done" << endl;
}

void test_tanh_layer1()
{
    cout << "Test tanh layer case 1 ...\t";
    using RootLayer = InjectPolicy_t<TanhLayer>;
    static_assert(!RootLayer::isFeedbackOutput, "Test Error");
    static_assert(!RootLayer::isUpdate, "Test Error");

    RootLayer layer;
    Matrix<float, CPU> in(2, 1);
    in.setValue(0, 0, -0.27f);
    in.setValue(1, 0, -0.41f);

    auto input = LayerIO::create().set<LayerIO>(in);

    layer_neutral_invariant(layer);

    auto out = layer.feedForward(input);
    auto res = evaluate(out.get<LayerIO>());
    assert(fabs(res(0, 0) - tanh(-0.27f)) < 0.001);
    assert(fabs(res(1, 0) - tanh(-0.41f)) < 0.001);

    details::NullParameter fbIn;
    auto out_grad = layer.feedBackward(fbIn);
    auto fb1 = out_grad.get<LayerIO>();
    static_assert(std::is_same_v<decltype(fb1), details::NullParameter>, "Test error");

    layer_neutral_invariant(layer);

    cout << "done" << endl;
}
void test_tanh_layer2()
{
    cout << "Test tanh layer case 2 ...\t";
    using RootLayer = InjectPolicy_t<TanhLayer, FeedbackOutputPolicy>;
    static_assert(RootLayer::isFeedbackOutput, "Test Error");
    static_assert(!RootLayer::isUpdate, "Test Error");

    RootLayer layer;

    Matrix<float, CPU> in(2, 1);
    in.setValue(0, 0, -0.27f);
    in.setValue(1, 0, -0.41f);

    auto input = LayerIO::create().set<LayerIO>(in);

    layer_neutral_invariant(layer);

    auto out = layer.feedForward(input);
    auto res = evaluate(out.get<LayerIO>());
    assert(fabs(res(0, 0) - tanh(-0.27f)) < 0.001);
    assert(fabs(res(1, 0) - tanh(-0.41f)) < 0.001);

    Matrix<float, CPU> grad(2, 1);
    grad.setValue(0, 0, 0.1f);
    grad.setValue(1, 0, 0.3f);
    auto out_grad = layer.feedBackward(LayerIO::create().set<LayerIO>(grad));
    auto fb = evaluate(out_grad.get<LayerIO>());
    assert(fabs(fb(0, 0) - 0.1f * (1 - tanh(-0.27f) * tanh(-0.27f))) < 0.001);
    assert(fabs(fb(1, 0) - 0.3f * (1 - tanh(-0.41f) * tanh(-0.41f))) < 0.001);

    layer_neutral_invariant(layer);
    cout << "done" << endl;
}
void test_tanh_layer3()
{
    cout << "Test tanh layer case 3 ...\t";
    using RootLayer = InjectPolicy_t<TanhLayer, FeedbackOutputPolicy>;
    static_assert(RootLayer::isFeedbackOutput, "Test Error");
    static_assert(!RootLayer::isUpdate, "Test Error");

    RootLayer layer;

    std::vector<Matrix<float, CPU>> op;

    layer_neutral_invariant(layer);
    for (size_t loop_count = 1; loop_count < 10; ++loop_count)
    {
        auto in = gen_matrix<float>(loop_count, 3, 0.1f, 0.13f);

        op.push_back(in);

        auto input = LayerIO::create().set<LayerIO>(in);

        auto out = layer.feedForward(input);
        auto res = evaluate(out.get<LayerIO>());
        assert(res.rowNum() == loop_count);
        assert(res.colNum() == 3);
        for (size_t i = 0; i < loop_count; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                assert(fabs(res(i, j) - tanh(in(i, j))) < 0.0001);
            }
        }
    }

    for (size_t loop_count = 9; loop_count >= 1; --loop_count)
    {
        auto grad = gen_matrix<float>(loop_count, 3, 2, 1.1f);
        auto out_grad = layer.feedBackward(LayerIO::create().set<LayerIO>(grad));

        auto fb = evaluate(out_grad.get<LayerIO>());

        auto in = op.back(); op.pop_back();
        for (size_t i = 0; i < loop_count; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                auto aim = grad(i, j) * (1 - tanh(in(i, j)) * tanh(in(i, j)));
                assert(fabs(fb(i, j) - aim) < 0.00001f);
            }
        }
    }

    layer_neutral_invariant(layer);

    cout << "done" << endl;
}


int main()
{
    test_sigmoid_layer1();
    test_sigmoid_layer2();
    test_sigmoid_layer3();
    test_softmax_layer1();
    test_softmax_layer2();
    test_tanh_layer1();
    test_tanh_layer2();
    test_tanh_layer3();

}
