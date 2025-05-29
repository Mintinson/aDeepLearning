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
#include <iostream>
#include <map>
#include <string>
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

void test_abs_layer1()
{
    cout << "Test abs layer case 1 ...\t";
    using RootLayer = InjectPolicy_t<AbsLayer>;
    static_assert(!RootLayer::isUpdate, "Test Error");
    static_assert(!RootLayer::isFeedbackOutput, "Test Error");

    RootLayer layer;

    auto in = gen_matrix<float>(4, 5, -3.3f, 0.1f);
    auto input = LayerIO::create().set<LayerIO>(in);

    layer_neutral_invariant(layer);
    auto out = layer.feedForward(input);
    auto res = evaluate(out.get<LayerIO>());

    assert(res.rowNum() == 4);
    assert(res.colNum() == 5);

    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j < 5; ++j)
        {
            auto check = fabs(in(i, j));
            assert(fabs(res(i, j) - check) < 0.0001);
        }
    }

    details::NullParameter fbIn;
    auto out_grad = layer.feedBackward(fbIn);
    auto fb1 = out_grad.get<LayerIO>();
    static_assert(std::is_same_v<decltype(fb1), details::NullParameter>, "Test error");

    layer_neutral_invariant(layer);
    cout << "done" << endl;
}

void test_abs_layer2()
{
    cout << "Test abs layer case 2 ...\t";
    using RootLayer = InjectPolicy_t<AbsLayer, FeedbackOutputPolicy>;
    static_assert(RootLayer::isFeedbackOutput, "Test Error");
    static_assert(!RootLayer::isUpdate, "Test Error");

    RootLayer layer;

    auto in = gen_matrix<float>(4, 5, -3.3f, 0.1f);
    auto input = LayerIO::create().set<LayerIO>(in);

    layer_neutral_invariant(layer);
    auto out = layer.feedForward(input);
    auto res = evaluate(out.get<LayerIO>());
    assert(res.rowNum() == 4);
    assert(res.colNum() == 5);

    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j < 5; ++j)
        {
            auto check = fabs(in(i, j));
            assert(fabs(res(i, j) - check) < 0.0001);
        }
    }

    auto grad = gen_matrix<float>(4, 5, 1.8f, -0.2f);
    auto out_grad = layer.feedBackward(LayerIO::create().set<LayerIO>(grad));
    auto fb = evaluate(out_grad.get<LayerIO>());

    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j < 5; ++j)
        {
            auto check = in(i, j) / fabs(in(i, j)) * grad(i, j);
            assert(fabs(fb(i, j) - check) < 0.0001);
        }
    }

    layer_neutral_invariant(layer);
    cout << "done" << endl;
}

void test_abs_layer3()
{
    cout << "Test abs layer case 3 ...\t";
    using RootLayer = InjectPolicy_t<AbsLayer, FeedbackOutputPolicy>;
    static_assert(RootLayer::isFeedbackOutput, "Test Error");
    static_assert(!RootLayer::isUpdate, "Test Error");

    RootLayer layer;

    std::vector<Matrix<float, CPU>> op;

    layer_neutral_invariant(layer);
    for (size_t loop_count = 1; loop_count < 10; ++loop_count)
    {
        auto in = gen_matrix<float>(loop_count, 3, -0.1f, 0.02f);

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
                auto check = fabs(in(i, j));
                assert(fabs(res(i, j) - check) < 0.0001);
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
                auto aim = in(i, j) / fabs(in(i, j)) * grad(i, j);
                assert(fabs(fb(i, j) - aim) < 0.00001f);
            }
        }
    }

    layer_neutral_invariant(layer);

    cout << "done" << endl;
}

void test_abs_layer4()
{
    cout << "Test abs layer case 4 ...\t";
    using RootLayer = InjectPolicy_t<AbsLayer, FeedbackOutputPolicy>;
    static_assert(RootLayer::isFeedbackOutput, "Test Error");
    static_assert(!RootLayer::isUpdate, "Test Error");

    RootLayer layer;

    Matrix<float, CPU> x(1, 4);
    x.setValue(0, 0, 0);
    x.setValue(0, 1, -2);
    x.setValue(0, 2, 3);
    x.setValue(0, 3, -4);
    auto x_out = layer.feedForward(LayerIO::create().set<LayerIO>(x));
    auto x_out_eval = evaluate(x_out.get<LayerIO>());
    assert(fabs(x_out_eval(0, 0) - 0) <= 0.00001);
    assert(fabs(x_out_eval(0, 1) - 2) <= 0.00001);
    assert(fabs(x_out_eval(0, 2) - 3) <= 0.00001);
    assert(fabs(x_out_eval(0, 3) - 4) <= 0.00001);

    Matrix<float, CPU> y(1, 4);
    y.setValue(0, 0, 1);
    y.setValue(0, 1, 5);
    y.setValue(0, 2, 7);
    y.setValue(0, 3, 3);
    auto y_out = layer.feedBackward(LayerIO::create().set<LayerIO>(y)).get<LayerIO>();
    auto y_out_eval = evaluate(y_out);
    // std::cout << "\n";
    // assert(fabs(y_out_eval(0, 0) - 0) <= 0.00001);
    assert(fabs(y_out_eval(0, 1) + 5) <= 0.00001);
    assert(fabs(y_out_eval(0, 2) - 7) <= 0.00001);
    assert(fabs(y_out_eval(0, 3) + 3) <= 0.00001);

    layer_neutral_invariant(layer);

    cout << "done" << endl;
}

void test_add_layer1()
{
    cout << "Test add layer case 1 ...\t";
    using RootLayer = InjectPolicy_t<AddLayer>;
    static_assert(!RootLayer::isFeedbackOutput, "Test Error");
    static_assert(!RootLayer::isUpdate, "Test Error");

    RootLayer layer;

    auto i1 = gen_matrix<float>(2, 3, 1, 0.1f);
    auto i2 = gen_matrix<float>(2, 3, 1.5f, -0.1f);

    auto input = AddLayerInput::create().set<AddLayerIn1>(i1).set<AddLayerIn2>(i2);

    auto out = layer.feedForward(input);
    auto res = evaluate(out.get<LayerIO>());
    for (size_t i = 0; i < 2; ++i)
    {
        for (size_t j = 0; j < 3; ++j)
        {
            assert(fabs(res(i, j) - i1(i, j) - i2(i, j)) < 0.001);
        }
    }

    details::NullParameter fbIn;
    auto out_grad = layer.feedBackward(fbIn);
    auto fb1 = out_grad.get<AddLayerIn1>();
    auto fb2 = out_grad.get<AddLayerIn2>();
    static_assert(std::is_same<decltype(fb1), details::NullParameter>::value, "Test error");
    static_assert(std::is_same<decltype(fb2), details::NullParameter>::value, "Test error");
    cout << "done" << endl;
}

void test_add_layer2()
{
    cout << "Test add layer case 2 ...\t";

    using RootLayer = InjectPolicy_t<AddLayer, FeedbackOutputPolicy>;
    static_assert(RootLayer::isFeedbackOutput, "Test Error");
    static_assert(!RootLayer::isUpdate, "Test Error");

    RootLayer layer;
    auto i1 = gen_matrix<float>(2, 3, 1, 0.1f);
    auto i2 = gen_matrix<float>(2, 3, 1.5f, -0.1f);

    auto input = AddLayerInput::create().set<AddLayerIn1>(i1).set<AddLayerIn2>(i2);

    auto out = layer.feedForward(input);
    auto res = evaluate(out.get<LayerIO>());
    for (size_t i = 0; i < 2; ++i)
    {
        for (size_t j = 0; j < 3; ++j)
        {
            assert(fabs(res(i, j) - i1(i, j) - i2(i, j)) < 0.001);
        }
    }

    auto grad = gen_matrix<float>(2, 3, 0.7f, -0.2f);

    auto out_grad = layer.feedBackward(RootLayer::OutputType::create().set<LayerIO>(grad));

    auto handle1 = out_grad.get<AddLayerIn1>().evalRegister();
    auto handle2 = out_grad.get<AddLayerIn2>().evalRegister();
    EvalPlan<CPU>::eval();

    auto fb1 = handle1.data();
    auto fb2 = handle2.data();
    assert(fb1.rowNum() == fb2.rowNum());
    assert(fb1.colNum() == fb2.colNum());
    assert(fb1.rowNum() == 2);
    assert(fb1.colNum() == 3);

    for (size_t i = 0; i < 2; ++i)
    {
        for (size_t j = 0; j < 3; ++j)
        {
            assert(fb1(i, j) == grad(i, j));
            assert(fb2(i, j) == grad(i, j));
        }
    }

    cout << "done" << endl;
}

void test_add_layer3()
{
    cout << "Test add layer case 3 ...\t";
    using RootLayer = InjectPolicy_t<AddLayer, FeedbackOutputPolicy>;
    static_assert(RootLayer::isFeedbackOutput, "Test Error");
    static_assert(!RootLayer::isUpdate, "Test Error");

    RootLayer layer;

    auto i1 = gen_matrix<float>(2, 3, 1, 0.1f);
    auto i2 = gen_matrix<float>(2, 3, 1.5f, -0.1f);

    auto input = AddLayerInput::create().set<AddLayerIn1>(i1).set<AddLayerIn2>(i2);

    auto out = layer.feedForward(input);
    auto res = evaluate(out.get<LayerIO>());
    for (size_t i = 0; i < 2; ++i)
    {
        for (size_t j = 0; j < 3; ++j)
        {
            assert(fabs(res(i, j) - i1(i, j) - i2(i, j)) < 0.001);
        }
    }

    auto i3 = gen_matrix<float>(2, 3, 1.3, -0.1f);
    auto i4 = gen_matrix<float>(2, 3, 2.5f, -0.7f);

    input = AddLayerInput::create().set<AddLayerIn1>(i3).set<AddLayerIn2>(i4);

    out = layer.feedForward(input);
    res = evaluate(out.get<LayerIO>());
    for (size_t i = 0; i < 2; ++i)
    {
        for (size_t j = 0; j < 3; ++j)
        {
            assert(fabs(res(i, j) - i3(i, j) - i4(i, j)) < 0.001);
        }
    }

    auto grad = gen_matrix<float>(2, 3, 0.7f, -0.2f);

    auto out_grad = layer.feedBackward(RootLayer::OutputType::create().set<LayerIO>(grad));

    auto handle1 = out_grad.get<AddLayerIn1>().evalRegister();
    auto handle2 = out_grad.get<AddLayerIn2>().evalRegister();
    EvalPlan<CPU>::eval();

    auto fb1 = handle1.data();
    auto fb2 = handle2.data();
    assert(fb1.rowNum() == fb2.rowNum());
    assert(fb1.colNum() == fb2.colNum());
    assert(fb1.rowNum() == 2);
    assert(fb1.colNum() == 3);

    for (size_t i = 0; i < 2; ++i)
    {
        for (size_t j = 0; j < 3; ++j)
        {
            assert(fb1(i, j) == grad(i, j));
            assert(fb2(i, j) == grad(i, j));
        }
    }

    grad = gen_matrix<float>(2, 3, -0.7f, 0.2f);

    out_grad = layer.feedBackward(RootLayer::OutputType::create().set<LayerIO>(grad));

    handle1 = out_grad.get<AddLayerIn1>().evalRegister();
    handle2 = out_grad.get<AddLayerIn2>().evalRegister();
    EvalPlan<CPU>::eval();

    fb1 = handle1.data();
    fb2 = handle2.data();

    assert(fb1.rowNum() == fb2.rowNum());
    assert(fb1.colNum() == fb2.colNum());
    assert(fb1.rowNum() == 2);
    assert(fb1.colNum() == 3);

    for (size_t i = 0; i < 2; ++i)
    {
        for (size_t j = 0; j < 3; ++j)
        {
            assert(fb1(i, j) == grad(i, j));
            assert(fb2(i, j) == grad(i, j));
        }
    }
    cout << "done" << endl;
}

void test_element_mul_layer1()
{
    cout << "Test element mul layer case 1 ...\t";
    using RootLayer = InjectPolicy_t<MulLayer>;
    static_assert(!RootLayer::isFeedbackOutput, "Test Error");
    static_assert(!RootLayer::isUpdate, "Test Error");

    RootLayer layer;

    Matrix<float, CPU> i1(2, 3);
    i1.setValue(0, 0, 0.1f);
    i1.setValue(0, 1, 0.2f);
    i1.setValue(0, 2, 0.3f);
    i1.setValue(1, 0, 0.4f);
    i1.setValue(1, 1, 0.5f);
    i1.setValue(1, 2, 0.6f);

    Matrix<float, CPU> i2(2, 3);
    i2.setValue(0, 0, 0.2f);
    i2.setValue(0, 1, 0.3f);
    i2.setValue(0, 2, 0.4f);
    i2.setValue(1, 0, 0.5f);
    i2.setValue(1, 1, 0.6f);
    i2.setValue(1, 2, 0.7f);

    auto input = MulLayerInput::create().set<MulLayerIn1>(i1).set<MulLayerIn2>(i2);

    layer_neutral_invariant(layer);

    auto out = layer.feedForward(input);
    auto res = evaluate(out.get<LayerIO>());
    assert(fabs(res(0, 0) - 0.02f) < 0.001);
    assert(fabs(res(0, 1) - 0.06f) < 0.001);
    assert(fabs(res(0, 2) - 0.12f) < 0.001);
    assert(fabs(res(1, 0) - 0.20f) < 0.001);
    assert(fabs(res(1, 1) - 0.30f) < 0.001);
    assert(fabs(res(1, 2) - 0.42f) < 0.001);

    auto out_grad = layer.feedBackward(LayerIO::create());
    auto fb1 = out_grad.get<MulLayerIn1>();
    auto fb2 = out_grad.get<MulLayerIn2>();
    static_assert(std::is_same<decltype(fb1), details::NullParameter>::value, "Test error");
    static_assert(std::is_same<decltype(fb2), details::NullParameter>::value, "Test error");

    layer_neutral_invariant(layer);
    cout << "done" << endl;
}

void test_element_mul_layer2()
{
    cout << "Test element mul layer case 2 ...\t";
    using RootLayer = InjectPolicy_t<MulLayer, FeedbackOutputPolicy>;
    static_assert(RootLayer::isFeedbackOutput, "Test Error");
    static_assert(!RootLayer::isUpdate, "Test Error");

    RootLayer layer;

    Matrix<float, CPU> i1(2, 3);
    i1.setValue(0, 0, 0.1f);
    i1.setValue(0, 1, 0.2f);
    i1.setValue(0, 2, 0.3f);
    i1.setValue(1, 0, 0.4f);
    i1.setValue(1, 1, 0.5f);
    i1.setValue(1, 2, 0.6f);

    Matrix<float, CPU> i2(2, 3);
    i2.setValue(0, 0, 0.2f);
    i2.setValue(0, 1, 0.3f);
    i2.setValue(0, 2, 0.4f);
    i2.setValue(1, 0, 0.5f);
    i2.setValue(1, 1, 0.6f);
    i2.setValue(1, 2, 0.7f);

    auto input = MulLayerInput::create().set<MulLayerIn1>(i1).set<MulLayerIn2>(i2);

    layer_neutral_invariant(layer);

    auto out = layer.feedForward(input);

    Matrix<float, CPU> grad(2, 3);
    grad.setValue(0, 0, 0.3f);
    grad.setValue(0, 1, 0.6f);
    grad.setValue(0, 2, 0.9f);
    grad.setValue(1, 0, 0.4f);
    grad.setValue(1, 1, 0.1f);
    grad.setValue(1, 2, 0.7f);
    auto out_grad = layer.feedBackward(LayerIO::create().set<LayerIO>(grad));

    auto handle1 = out.get<LayerIO>().evalRegister();
    auto handle2 = out_grad.get<MulLayerIn1>().evalRegister();
    auto handle3 = out_grad.get<MulLayerIn2>().evalRegister();
    EvalPlan<CPU>::eval();

    auto res = handle1.data();
    assert(fabs(res(0, 0) - 0.02f) < 0.001);
    assert(fabs(res(0, 1) - 0.06f) < 0.001);
    assert(fabs(res(0, 2) - 0.12f) < 0.001);
    assert(fabs(res(1, 0) - 0.20f) < 0.001);
    assert(fabs(res(1, 1) - 0.30f) < 0.001);
    assert(fabs(res(1, 2) - 0.42f) < 0.001);

    auto g1 = handle2.data();
    auto g2 = handle3.data();
    assert(fabs(g1(0, 0) - 0.06f) < 0.001);
    assert(fabs(g1(0, 1) - 0.18f) < 0.001);
    assert(fabs(g1(0, 2) - 0.36f) < 0.001);
    assert(fabs(g1(1, 0) - 0.20f) < 0.001);
    assert(fabs(g1(1, 1) - 0.06f) < 0.001);
    assert(fabs(g1(1, 2) - 0.49f) < 0.001);

    assert(fabs(g2(0, 0) - 0.03f) < 0.001);
    assert(fabs(g2(0, 1) - 0.12f) < 0.001);
    assert(fabs(g2(0, 2) - 0.27f) < 0.001);
    assert(fabs(g2(1, 0) - 0.16f) < 0.001);
    assert(fabs(g2(1, 1) - 0.05f) < 0.001);
    assert(fabs(g2(1, 2) - 0.42f) < 0.001);

    layer_neutral_invariant(layer);

    cout << "done" << endl;
}

void test_element_mul_layer3()
{
    cout << "Test element mul layer case 3 ...\t";
    using RootLayer = InjectPolicy_t<MulLayer, FeedbackOutputPolicy>;
    static_assert(RootLayer::isFeedbackOutput, "Test Error");
    static_assert(!RootLayer::isUpdate, "Test Error");

    RootLayer layer;

    std::vector<Matrix<float, CPU>> op1;
    std::vector<Matrix<float, CPU>> op2;
    layer_neutral_invariant(layer);
    for (size_t loop_count = 1; loop_count < 10; ++loop_count)
    {
        auto i1 = gen_matrix<float>(loop_count, 3, 0, 0.3f);
        auto i2 = gen_matrix<float>(loop_count, 3, -1, 1.3f);
        op1.push_back(i1);
        op2.push_back(i2);

        auto input = MulLayerInput::create().set<MulLayerIn1>(i1).set<MulLayerIn2>(i2);

        auto out = layer.feedForward(input);
        auto res = evaluate(out.get<LayerIO>());
        assert(res.rowNum() == loop_count);
        assert(res.colNum() == 3);
        for (size_t i = 0; i < loop_count; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                assert(fabs(res(i, j) - i1(i, j) * i2(i, j)) < 0.0001);
            }
        }
    }

    for (size_t loop_count = 9; loop_count >= 1; --loop_count)
    {
        auto grad = gen_matrix<float>(loop_count, 3, 2, 1.1f);
        auto out_grad = layer.feedBackward(LayerIO::create().set<LayerIO>(grad));

        auto handle1 = out_grad.get<MulLayerIn1>().evalRegister();
        auto handle2 = out_grad.get<MulLayerIn2>().evalRegister();
        EvalPlan<CPU>::eval();

        auto g1 = handle1.data();
        auto g2 = handle2.data();

        auto i1 = op1.back();
        op1.pop_back();
        auto i2 = op2.back();
        op2.pop_back();
        for (size_t i = 0; i < loop_count; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                assert(fabs(g1(i, j) - grad(i, j) * i2(i, j)) < 0.001);
                assert(fabs(g2(i, j) - grad(i, j) * i1(i, j)) < 0.001);
            }
        }
    }

    layer_neutral_invariant(layer);

    cout << "done" << endl;
}

int main()
{
    test_abs_layer2();
    test_abs_layer3();
    test_abs_layer4();
    test_add_layer1();
    test_add_layer2();
    test_add_layer3();

    test_element_mul_layer1();
    test_element_mul_layer2();
    test_element_mul_layer3();
}