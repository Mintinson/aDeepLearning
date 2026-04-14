#include <cassert>
#include <iostream>

#include <metann/operators/binary_operators.hpp>

using namespace metann;
using namespace std;

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

namespace {
void test_tanh_derivative1() {
    cout << "Test tanh derivative case 1 ...\t";
    auto rm1 = gen_matrix<float>(4, 5, 0, 0.0001f);
    auto rm2 = gen_matrix<float>(4, 5, 1, 0.0003f);
    auto t = tanh_derivative(rm1, rm2);
    auto t_r = evaluate(t);
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 5; ++j) {
            float aim = rm1(i, j) * (1 - rm2(i, j) * rm2(i, j));
            assert(fabs(t_r(i, j) - aim) < 0.0001);
        }
    }

    rm1 = gen_matrix<float>(111, 113, 1, 0.001f);
    rm2 = gen_matrix<float>(111, 113, 0, 0.003f);
    rm1 = rm1.subMatrix(31, 35, 17, 22);
    rm2 = rm2.subMatrix(30, 34, 18, 23);
    t = tanh_derivative(rm1, rm2);
    t_r = evaluate(t);
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 5; ++j) {
            float aim = rm1(i, j) * (1 - rm2(i, j) * rm2(i, j));
            assert(fabs(t_r(i, j) - aim) < 0.0001);
        }
    }
    cout << "done" << endl;
}

void test_tanh_derivative2() {
    cout << "Test tanh derivative case 2 ...\t";
    {
        auto rm1 = gen_matrix<float>(4, 5, 0, 0.0001f);
        auto rm2 = gen_matrix<float>(4, 5, 1, 0.0003f);
        auto res = tanh_derivative(rm1, rm2);
        auto res2 = tanh_derivative(rm1, rm2);

        assert(res == res2);

        auto cm1 = evaluate(res);
        auto cm2 = evaluate(res);
        assert(cm1 == cm2);
    }
    {
        auto rm1 = gen_matrix<float>(4, 5, 0, 0.0001f);
        auto rm2 = gen_matrix<float>(4, 5, 1, 0.0003f);
        auto res = tanh_derivative(rm1, rm2);
        auto res2 = res;

        assert(res == res2);

        auto handle1 = res.evalRegister();
        auto handle2 = res2.evalRegister();
        EvalPlan<CPU>::eval();

        auto cm1 = handle1.data();
        auto cm2 = handle2.data();
        assert(cm1 == cm2);
    }
    cout << "done" << endl;
}

void test_tanh_derivative3() {
    cout << "Test tanh derivative case 3 ...\t";
    auto rm1 = gen_batch_matrix<float>(4, 5, 7, 0, 0.0001f);
    auto rm2 = gen_batch_matrix<float>(4, 5, 7, 1, 0.0003f);
    auto t = tanh_derivative(rm1, rm2);
    auto t_r = evaluate(t);
    for (size_t b = 0; b < 7; ++b) {
        for (size_t i = 0; i < 4; ++i) {
            for (size_t j = 0; j < 5; ++j) {
                float aim = rm1[b](i, j) * (1 - rm2[b](i, j) * rm2[b](i, j));
                assert(fabs(t_r[b](i, j) - aim) < 0.0001);
            }
        }
    }
    cout << "done" << endl;
}
}  // namespace

void test_tanh_derivative() {
    std::cout << "Tanh Derivative Test Start ..." << std::endl;
    test_tanh_derivative1();
    test_tanh_derivative2();
    test_tanh_derivative3();
    std::cout << "Tanh Derivative Test End ..." << std::endl;
}

int main() {
    test_tanh_derivative();
}