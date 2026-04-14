#include <cassert>
#include <iostream>
#include <metann/layers/interface_fun.hpp>
#include <metann/layers/cost/negative_log_likelihood_layer.hpp>

using namespace metann;
using namespace std;

template<typename Elem>
auto gen_matrix(std::size_t r, std::size_t c, Elem start = 0, Elem scale = 1) {
    using namespace metann;
    Matrix<Elem, CPU> res(r, c);
    for (std::size_t i = 0; i < r; ++i) {
        for (std::size_t j = 0; j < c; ++j) {
            res.setValue(i, j, (Elem) (start * scale));
            start += 1.0f;
        }
    }
    return res;
}

template<typename TElem>
auto gen_batch_matrix(size_t r, size_t c, size_t d, float start = 0, float scale = 1) {
    using namespace metann;
    Batch<TElem, metann::CPU, CategoryTags::Matrix> res(d, r, c);
    for (size_t i = 0; i < r; ++i) {
        for (size_t j = 0; j < c; ++j) {
            for (size_t k = 0; k < d; ++k) {
                res.setValue(k, i, j, (TElem) (start * scale));
                start += 1.0f;
            }
        }
    }
    return res;
}

namespace {
    void test_negative_log_likelihood_layer1() {
        cout << "Test negative log likelyhood layer case 1 ...\t";
        using RootLayer = InjectPolicy_t<NegativeLogLikelihoodLayer>;
        static_assert(!RootLayer::isFeedbackOutput, "Test Error");
        static_assert(!RootLayer::isUpdate, "Test Error");

        RootLayer layer;
        auto in = gen_matrix<float>(3, 4, 0.1f, 0.05f);
        auto label = gen_matrix<float>(3, 4, 0.3f, 0.1f);

        auto input = CostLayerIO::create().set<CostLayerIO>(in).set<CostLayerLabel>(label);

        layer_neutral_invariant(layer);

        auto out = layer.feedforward(input);
        auto res = evaluate(out.get<LayerIO>());
        float check = 0;
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 4; ++j) {
                check -= log(in(i, j)) * label(i, j);
            }
        }
        assert(fabs(res.value() - check) < 0.0001);

        layer_neutral_invariant(layer);

        details::NullParameter fbIn;
        auto out_grad = layer.feedBackward(fbIn);
        auto fb1 = out_grad.get<CostLayerIO>();
        static_assert(std::is_same<decltype(fb1), details::NullParameter>::value, "Test error");

        cout << "done" << endl;
    }

    void test_negative_log_likelihood_layer2() {
        cout << "Test negative log likelyhood layer case 2 ...\t";
        using RootLayer = InjectPolicy_t<NegativeLogLikelihoodLayer, FeedbackOutputPolicy>;
        static_assert(RootLayer::isFeedbackOutput, "Test Error");
        static_assert(!RootLayer::isUpdate, "Test Error");

        RootLayer layer;
        auto in = gen_matrix<float>(3, 4, 0.1f, 0.05f);
        auto label = gen_matrix<float>(3, 4, 0.3f, 0.1f);

        auto input = CostLayerIO::create()
                .set<CostLayerIO>(in)
                .set<CostLayerLabel>(label);

        layer_neutral_invariant(layer);

        auto out = layer.feedforward(input);
        auto res = evaluate(out.get<LayerIO>());
        float check = 0;
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 4; ++j) {
                check -= log(in(i, j)) * label(i, j);
            }
        }
        assert(fabs(res.value() - check) < 0.0001);

        auto fb = LayerIO::create().set<LayerIO>(Scalar<float>(0.5));
        auto out_grad = layer.feedBackward(fb);
        layer_neutral_invariant(layer);

        auto g = evaluate(out_grad.get<CostLayerIO>());
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 4; ++j) {
                assert(fabs(g(i, j) + 0.5 * label(i, j) / in(i, j)) < 0.0001);
            }
        }
        cout << "done" << endl;
    }

    void test_negative_log_likelihood_layer3() {
        cout << "Test negative log likelyhood layer case 3 ...\t";
        using RootLayer = InjectPolicy_t<NegativeLogLikelihoodLayer, FeedbackOutputPolicy>;
        static_assert(RootLayer::isFeedbackOutput, "Test Error");
        static_assert(!RootLayer::isUpdate, "Test Error");

        RootLayer layer;

        vector<Matrix<float, CPU> > op_in;
        vector<Matrix<float, CPU> > op_label;

        layer_neutral_invariant(layer);
        for (size_t loop_count = 1; loop_count < 10; ++loop_count) {
            auto in = gen_matrix<float>(loop_count * 2, 4, 0.1f, 0.05f);
            auto label = gen_matrix<float>(loop_count * 2, 4, 0.3f, 0.1f);

            op_in.push_back(in);
            op_label.push_back(label);

            auto input = CostLayerIO::create().set<CostLayerIO>(in).set<CostLayerLabel>(label);

            auto out = layer.feedforward(input);
            auto res = evaluate(out.get<LayerIO>());

            float check = 0;
            for (size_t i = 0; i < loop_count * 2; ++i) {
                for (size_t j = 0; j < 4; ++j) {
                    check -= log(in(i, j)) * label(i, j);
                }
            }
            assert(fabs(res.Value() - check) < 0.0001);
        }

        for (size_t loop_count = 9; loop_count >= 1; --loop_count) {
            auto out_grad = layer.feedBackward(LayerIO::create().set<LayerIO>(Scalar<float>(0.5 * loop_count)));
            auto fb = evaluate(out_grad.get<CostLayerIO>());

            auto in = op_in.back();
            op_in.pop_back();
            auto label = op_label.back();
            op_label.pop_back();
            for (size_t i = 0; i < loop_count * 2; ++i) {
                for (size_t j = 0; j < 3; ++j) {
                    assert(fabs(fb(i, j) + 0.5 * loop_count * label(i, j) / in(i, j)) < 0.0001);
                }
            }
        }

        layer_neutral_invariant(layer);
        cout << "done" << endl;
    }
}

// void test_negative_log_likelihood_layer()
int main() {
    test_negative_log_likelihood_layer1();
    test_negative_log_likelihood_layer2();
    test_negative_log_likelihood_layer3();
}
