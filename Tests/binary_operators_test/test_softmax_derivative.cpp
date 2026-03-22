#include <cassert>
#include <iostream>
#include <metann/layers/elementary/softmax_layer.hpp>
#include <metann/operators/binary_operators.hpp>

using namespace metann;
using namespace std;

using CheckDevice = metann::CPU;

template <typename Elem>
inline auto gen_matrix(std::size_t r, std::size_t c, Elem start = 0, Elem scale = 1)
{
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
inline auto gen_batch_matrix(size_t r, size_t c, size_t d, float start = 0, float scale = 1)
{
    using namespace metann;
    Batch<TElem, metann::CPU, CategoryTags::Matrix> res(d, r, c);
    for (size_t i = 0; i < r; ++i) {
        for (size_t j = 0; j < c; ++j) {
            for (size_t k = 0; k < d; ++k) {
                res.setValue(k, i, j, (TElem)(start * scale));
                start += 1.0f;
            }
        }
    }
    return res;
}

namespace {
void test_softmax_derivative1()
{
    cout << "Test softmax derivative case 1 ...\t";
    auto mSout = gen_matrix<float>(1, 4, 1.0f, 0.0001f);
    auto mGrad = gen_matrix<float>(1, 4, .3f, 0.005f);
    auto t = softmax_derivative(mGrad, mSout);
    auto t_r = evaluate(t);

    Matrix<float, CheckDevice> helper(4, 4);
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            if (i == j) {
                helper.setValue(i, j, mSout(0, i) * (1 - mSout(0, i)));
            } else {
                helper.setValue(i, j, -mSout(0, i) * mSout(0, j));
            }
        }
    }
    helper = evaluate(dot(mGrad, helper));
    for (size_t i = 0; i < 4; ++i) {
        assert(fabs(t_r(0, i) - helper(0, i)) < 0.0001);
    }

    mSout = gen_matrix<float>(111, 113, 1.1f, 0.0001f);
    mGrad = Matrix<float, CheckDevice>(111, 113);
    mGrad = mGrad.subMatrix(27, 28, 41, 45);
    mSout = mSout.subMatrix(17, 18, 31, 35);
    t = softmax_derivative(mGrad, mSout);
    t_r = evaluate(t);

    helper = Matrix<float, CheckDevice>(4, 4);
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            if (i == j) {
                helper.setValue(i, j, mSout(0, i) * (1 - mSout(0, i)));
            } else {
                helper.setValue(i, j, -mSout(0, i) * mSout(0, j));
            }
        }
    }
    helper = evaluate(dot(mGrad, helper));
    for (size_t i = 0; i < 4; ++i) {
        assert(fabs(t_r(0, i) - helper(0, i)) < 0.01);
    }
    cout << "done" << endl;
}

void test_softmax_derivative2()
{
    cout << "Test softmax derivative case 2 ...\t";
    {
        auto bSout = gen_matrix<float>(1, 4, 1.0f, 0.0001f);
        auto bGrad = gen_matrix<float>(1, 4, 0.3f, 0.007f);
        auto res = softmax_derivative(bGrad, bSout);
        auto res2 = softmax_derivative(bGrad, bSout);

        assert(res == res2);

        auto cm1 = evaluate(res);
        auto cm2 = evaluate(res);
        assert(cm1 == cm2);
    }
    {
        auto bSout = gen_matrix<float>(1, 4, 1.0f, 0.0001f);
        auto bGrad = gen_matrix<float>(1, 4, 0.3f, 0.007f);
        auto res = softmax_derivative(bGrad, bSout);
        auto res2 = res;

        assert(res == res2);

        auto handle1 = res.evalRegister();
        auto handle2 = res2.evalRegister();
        EvalPlan<CheckDevice>::eval();

        auto cm1 = handle1.data();
        auto cm2 = handle2.data();
        assert(cm1 == cm2);
    }
    cout << "done" << endl;
}

// void test_softmax_derivative3()
// {
//     cout << "Test softmax derivative case 3 ...\t";
//     {
//         InjectPolicy_t<SoftmaxLayer, FeedbackOutputPolicy> layer1;
//         InjectPolicy_t<NegativeLogLikelihoodLayer, FeedbackOutputPolicy> layer2;
//         auto layer1Input = LayerIO::create().set<LayerIO>(gen_matrix<float>(1, 5, 1, 1));
//         auto layer1Output = layer1.feedForward(layer1Input).get<LayerIO>();

//         auto target = gen_matrix<float>(1, 5, 0.1f, -0.3f);
//         auto layer2Input = CostLayerIn::Create()
//                                .set<CostLayerIn>(layer1Output)
//                                .set<CostLayerLabel>(target);

//         auto layer2Output = layer2.feedForward(layer2Input).get<LayerIO>();

//         auto layer2GradOutput = layer2.feedBackward(LayerIO::create().set<LayerIO>(Scalar<float>(0.7))).get<CostLayerIn>();
//         auto layer1GradOutput = layer1.feedBackward(LayerIO::create().set<LayerIO>(layer2GradOutput)).get<LayerIO>();

//         auto check = evaluate(layer1GradOutput);

//         auto softRes = evaluate(layer1Output);

//         float sum = 0;
//         for (size_t i = 0; i < 5; ++i) {
//             sum += target(0, i);
//         }

//         for (size_t i = 0; i < 5; ++i) {
//             float compare = softRes(0, i) * sum - target(0, i);
//             assert(fabs(check(0, i) - compare * 0.7f) <= 0.0001);
//         }
//     }
//     {
//         InjectPolicy_t<SoftmaxLayer, FeedbackOutputPolicy, BatchMode> layer1;
//         InjectPolicy_t<NegativeLogLikelihoodLayer, FeedbackOutputPolicy, BatchMode> layer2;
//         auto layer1Input = LayerIO::Create().Set<LayerIO>(gen_batch_matrix<float>(1, 5, 7, 1, 1));
//         auto layer1Output = layer1.FeedForward(layer1Input).Get<LayerIO>();

//         auto target = gen_batch_matrix<float>(1, 5, 7, 0.1f, -0.3f);
//         auto layer2Input = CostLayerIn::Create()
//                                .Set<CostLayerIn>(layer1Output)
//                                .Set<CostLayerLabel>(target);

//         auto layer2Output = layer2.FeedForward(layer2Input).Get<LayerIO>();

//         auto scale = MakeDuplicate(7, Scalar<float>(0.7f));
//         auto layer2GradOutput = layer2.FeedBackward(LayerIO::Create().Set<LayerIO>(scale)).Get<CostLayerIn>();
//         auto layer1GradOutput = layer1.FeedBackward(LayerIO::Create().Set<LayerIO>(layer2GradOutput)).Get<LayerIO>();

//         auto check = evaluate(layer1GradOutput);

//         auto softRes = evaluate(layer1Output);

//         for (size_t b = 0; b < 7; ++b) {
//             float sum = 0;
//             for (size_t i = 0; i < 5; ++i) {
//                 sum += target[b](0, i);
//             }

//             for (size_t i = 0; i < 5; ++i) {
//                 float compare = softRes[b](0, i) * sum - target[b](0, i);
//                 assert(fabs(check[b](0, i) - compare * 0.7f) <= 0.0001);
//             }
//         }
//     }
//     cout << "done" << endl;
// }

// void test_softmax_derivative4()
// {
//     cout << "Test softmax derivative case 4 ...\t";
//     {
//         InjectPolicy<SoftmaxLayer, PFeedbackOutput> layer1;
//         InjectPolicy<NegativeLogLikelihoodLayer, PFeedbackOutput> layer2;

//         auto layer1Input = LayerIO::Create().Set<LayerIO>(gen_matrix<float>(1, 5, 1, 1));
//         auto layer1Output = layer1.FeedForward(layer1Input).Get<LayerIO>();

//         auto target = OneHotVector<float, CheckDevice>(5, 3);
//         auto layer2Input = CostLayerIn::Create()
//                                .Set<CostLayerIn>(layer1Output)
//                                .Set<CostLayerLabel>(target);

//         auto layer2Output = layer2.FeedForward(layer2Input).Get<LayerIO>();

//         auto layer2GradOutput = layer2.FeedBackward(LayerIO::Create().Set<LayerIO>(Scalar<float>(0.7))).Get<CostLayerIn>();
//         auto layer1GradOutput = layer1.FeedBackward(LayerIO::Create().Set<LayerIO>(layer2GradOutput)).Get<LayerIO>();

//         auto check = evaluate(layer1GradOutput);

//         auto softRes = evaluate(layer1Output);

//         for (size_t i = 0; i < 5; ++i) {
//             float compare = softRes(0, i);
//             if (i == 3) {
//                 compare -= 1;
//             }
//             assert(fabs(check(0, i) - compare * 0.7f) <= 0.0001);
//         }
//     }
//     cout << "done" << endl;
// }

void test_softmax_derivative5()
{
    cout << "Test softmax derivative case 5 ...\t";
    auto mSout = gen_batch_matrix<float>(1, 4, 7, 1.0f, 0.0001f);
    auto mGrad = gen_batch_matrix<float>(1, 4, 7, .3f, 0.005f);
    auto t = softmax_derivative(mGrad, mSout);
    auto t_r = evaluate(t);

    for (size_t b = 0; b < 7; ++b) {
        Matrix<float, CheckDevice> helper(4, 4);
        for (size_t i = 0; i < 4; ++i) {
            for (size_t j = 0; j < 4; ++j) {
                if (i == j) {
                    helper.setValue(i, j, mSout[b](0, i) * (1 - mSout[b](0, i)));
                } else {
                    helper.setValue(i, j, -mSout[b](0, i) * mSout[b](0, j));
                }
            }
        }
        helper = evaluate(dot(mGrad[b], helper));
        for (size_t i = 0; i < 4; ++i) {
            assert(fabs(t_r[b](0, i) - helper(0, i)) < 0.0001);
        }
    }
    cout << "done" << endl;
}
}

void test_softmax_derivative()
{
    test_softmax_derivative1();
    test_softmax_derivative2();
    // test_softmax_derivative3();
    // test_softmax_derivative4();
    test_softmax_derivative5();
}

int main()
{
    std::cout << "Testing softmax derivative..." << std::endl;
    test_softmax_derivative();
    std::cout << "All tests passed!" << std::endl;
}
