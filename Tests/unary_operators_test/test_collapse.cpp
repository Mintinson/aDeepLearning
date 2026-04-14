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

void test_collapse1() {
    cout << "Test collapse case 1 ...\t";
    auto rm1 = gen_batch_matrix<float>(4, 5, 7, 1.0f, 0.0001f);
    auto t = collapse(rm1);
    assert(t.rowNum() == 4);
    assert(t.colNum() == 5);

    auto handle = t.evalRegister();
    EvalPlan<CPU>::eval();
    auto t_r = handle.data();

    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 5; ++j) {
            float aim = 0;
            for (size_t k = 0; k < 7; ++k) {
                aim += rm1[k](i, j);
            }
            assert(fabs(t_r(i, j) - aim) < 0.0001);
        }
    }
    cout << "done" << endl;
}

void test_collapse2() {
    cout << "Test collapse case 2 ...\t";
    {
        auto rm1 = gen_batch_matrix<float>(4, 5, 7, 1.0f, 0.0001f);
        auto collapse1 = collapse(rm1);
        auto collapse2 = collapse(rm1);

        assert(collapse1 == collapse2);

        auto handle1 = collapse1.evalRegister();
        auto handle2 = collapse1.evalRegister();
        EvalPlan<CPU>::eval();
        auto cm1 = handle1.data();
        auto cm2 = handle2.data();
        assert(cm1 == cm2);
    }
    {
        auto rm1 = gen_batch_matrix<float>(4, 5, 7, 1.0f, 0.0001f);
        auto collapse1 = collapse(rm1);
        auto collapse2 = collapse1;

        assert(collapse1 == collapse2);

        const auto& evalHandle1 = collapse1.evalRegister();
        const auto& evalHandle2 = collapse2.evalRegister();
        EvalPlan<CPU>::eval();

        auto cm1 = evalHandle1.data();
        auto cm2 = evalHandle2.data();
        assert(cm1 == cm2);
    }
    cout << "done" << endl;
}

int main() {
    std::cout << std::format("Collapse Tests Start") << std::endl;
    test_collapse1();
    test_collapse2();
    std::cout << std::format("Collapse Tests End") << std::endl;
}
