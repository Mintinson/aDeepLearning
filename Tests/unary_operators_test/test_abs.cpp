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

void test_abs1() {
    std::cout << "Test abs case 1 ...\t";
    auto rm1 = gen_matrix<float>(4, 5, -3.3f, 0.1f);
    auto t = metann::abs(rm1);

    auto handle = t.evalRegister();
    EvalPlan<CPU>::eval();
    auto t_r = handle.data();

    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 5; ++j) {
            float aim = fabs(rm1(i, j));
            assert(fabs(t_r(i, j) - aim) < 0.0001);
        }
    }

    rm1 = gen_matrix<float>(111, 113, -22, 0.0001f);
    rm1 = rm1.subMatrix(31, 35, 17, 22);
    t = metann::abs(rm1);
    t_r = metann::evaluate(t);

    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 5; ++j) {
            float aim = fabs(rm1(i, j));
            assert(fabs(t_r(i, j) - aim) < 0.0001);
        }
    }
    std::cout << "done\n";
}

void test_abs2() {
    std::cout << "Test abs case 2 ...\t";
    {
        auto rm1 = gen_matrix<float>(4, 5, 0, 0.0001f);
        auto res = abs(rm1);
        auto res2 = abs(rm1);

        assert(res == res2);

        auto cm1 = evaluate(res);
        auto cm2 = evaluate(res);
        assert(cm1 == cm2);
    }
    {
        auto rm1 = gen_matrix<float>(4, 5, 0, 0.0001f);
        auto res = abs(rm1);
        auto res2 = res;

        assert(res == res2);

        const auto& evalHandle1 = res.evalRegister();
        const auto& evalHandle2 = res2.evalRegister();
        EvalPlan<CPU>::eval();

        auto cm1 = evalHandle1.data();
        auto cm2 = evalHandle2.data();
        assert(cm1 == cm2);
    }
    std::cout << "done" << std::endl;
}

void test_abs3() {
    std::cout << "Test abs case 3 ...\t";
    auto rm1 = gen_batch_matrix<float>(4, 5, 7, -3.3f, 0.1f);
    auto t = metann::abs(rm1);

    auto handle = t.evalRegister();
    EvalPlan<CPU>::eval();
    auto t_r = handle.data();

    for (size_t b = 0; b < 7; ++b) {
        for (size_t i = 0; i < 4; ++i) {
            for (size_t j = 0; j < 5; ++j) {
                float aim = fabs(rm1[b](i, j));
                assert(fabs(t_r[b](i, j) - aim) < 0.0001);
            }
        }
    }
    std::cout << "done" << std::endl;
}

int main() {
    std::cout << std::format("{} {}", "hello", "world") << std::endl;
    test_abs1();
    test_abs2();
    test_abs3();
}
