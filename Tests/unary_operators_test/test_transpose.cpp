//
// Created by asus on 2025/1/10.
//

#include <metann/data/data_device.hpp>
#include <metann/data/matrix.hpp>
#include <metann/operators/unary_operators.hpp>
#include <format>
#include <iostream>

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
void test_transpose1()
{
    std::cout << "Test transpose case 1 ...\t";
    auto rm1 = gen_matrix<int>(4, 5, 0, 1);
    auto tr = transpose(rm1);
    auto tr_r = evaluate(tr);
    for (size_t i = 0; i < 5; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            assert(tr_r(i, j) == rm1(j, i));
        }
    }

    rm1 = gen_matrix<int>(111, 113, 2, 3);
    rm1 = rm1.subMatrix(31, 35, 17, 22);
    tr = transpose(rm1);
    tr_r = evaluate(tr);
    for (size_t i = 0; i < 5; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            assert(tr_r(i, j) == rm1(j, i));
        }
    }
    std::cout << "done" << std::endl;
}
void test_transpose2()
{
    std::cout << "Test transpose case 2 ...\t";
    {
        auto rm1 = gen_matrix<int>(4, 5, 0, 1);
        auto res = transpose(rm1);
        auto res2 = transpose(rm1);

        assert(res == res2);

        auto cm1 = evaluate(res);
        auto cm2 = evaluate(res);
        assert(cm1 == cm2);
    }
    {
        auto rm1 = gen_matrix<int>(4, 5, 0, 1);
        auto res = transpose(rm1);
        auto res2 = res;

        assert(res == res2);

        const auto& evalHandle1 = res.evalRegister();
        const auto& evalHandle2 = res2.evalRegister();
        EvalPlan<CPU>::eval();

        auto cm1 = evalHandle1.data();
        auto cm2 = evalHandle2.data();
    }
    std::cout << "done" << std::endl;
}
void test_transpose3()
{
    std::cout << "Test transpose case 3 ...\t";
    auto rm1 = gen_batch_matrix<int>(4, 5, 7, 0, 1);
    auto tr = transpose(rm1);
    auto tr_r = evaluate(tr);
    for (size_t b = 0; b < 7; ++b) {
        for (size_t i = 0; i < 5; ++i) {
            for (size_t j = 0; j < 4; ++j) {
                assert(tr_r[b](i, j) == rm1[b](j, i));
            }
        }
    }
    std::cout << "done" << std::endl;
}
int main()
{
    std::cout << std::format("Transpose Tests Start") << std::endl;
    test_transpose1();
    test_transpose2();
    test_transpose3();
    std::cout << std::format("Transpose Tests End") << std::endl;

}
