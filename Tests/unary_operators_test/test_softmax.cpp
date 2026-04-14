//
// Created by asus on 2025/1/10.
//

#include <format>
#include <iostream>

#include <metann/data/data_device.hpp>
#include <metann/data/matrix.hpp>
#include <metann/operators/unary_operators.hpp>

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
                res.setValue(k, i, j, (TElem)(start * scale));
                start += 1.0f;
            }
        }
    }
    return res;
}

void test_softmax1() {
    cout << "Test softmax case 1 ...\t";
    auto rm1 = gen_matrix<float>(1, 20, 0, 0.001f);
    auto t = vec_softmax(rm1);
    auto t_r = evaluate(t);

    float sum = 0;
    for (size_t i = 0; i < 20; ++i) {
        sum += exp(rm1(0, i));
    }

    for (size_t i = 0; i < 20; ++i) {
        assert(fabs(t_r(0, i) - exp(rm1(0, i)) / sum) < 0.0001);
    }

    rm1 = gen_matrix<float>(111, 113, 2, 0.001f);
    rm1 = rm1.subMatrix(17, 18, 31, 51);
    t = vec_softmax(rm1);
    t_r = evaluate(t);

    sum = 0;
    for (size_t i = 0; i < 20; ++i) {
        sum += exp(rm1(0, i));
    }

    for (size_t i = 0; i < 20; ++i) {
        assert(fabs(t_r(0, i) - exp(rm1(0, i)) / sum) < 0.0001);
    }
    cout << "done" << endl;
}

void test_softmax2() {
    cout << "Test softmax case 2 ...\t";
    {
        auto rm1 = gen_batch_matrix<float>(1, 20, 0, 0.001f);
        auto res = vec_softmax(rm1);
        auto res2 = vec_softmax(rm1);

        assert(res == res2);

        auto cm1 = evaluate(res);
        auto cm2 = evaluate(res);
        assert(cm1 == cm2);
    }
    {
        auto rm1 = gen_matrix<float>(1, 20, 0, 0.001f);
        auto res = vec_softmax(rm1);
        auto res2 = res;

        assert(res == res2);

        const auto& evalHandle1 = res.evalRegister();
        const auto& evalHandle2 = res2.evalRegister();
        EvalPlan<CPU>::eval();

        auto cm1 = evalHandle1.data();
        auto cm2 = evalHandle2.data();
        assert(cm1 == cm2);
    }
    cout << "done" << endl;
}

void test_softmax3() {
    cout << "Test softmax case 3 ...\t";
    auto rm1 = gen_batch_matrix<float>(1, 20, 7, 0, 0.001f);
    auto t = vec_softmax(rm1);
    auto t_r = evaluate(t);

    for (size_t b = 0; b < 7; ++b) {
        float sum = 0;
        for (size_t i = 0; i < 20; ++i) {
            sum += exp(rm1[b](0, i));
        }

        for (size_t i = 0; i < 20; ++i) {
            assert(fabs(t_r[b](0, i) - exp(rm1[b](0, i)) / sum) < 0.0001);
        }
    }
    cout << "done" << endl;
}

int main() {
    std::cout << std::format("Softmax Tests Start") << std::endl;
    test_softmax1();
    test_softmax2();
    test_softmax3();
    std::cout << std::format("Softmax Tests End") << std::endl;
}
