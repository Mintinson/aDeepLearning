
#include <metann/operators/binary_operators.hpp>
#include <iostream>
#include <cassert>

using namespace std;
using namespace metann;

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


void test_mul1()
{
    cout << "Test element mul case 1 ...\t";
    auto rm1 = gen_matrix<int>(4, 5, 0, 1);
    auto rm2 = gen_matrix<int>(4, 5, 3, 2);
    auto mul = rm1 * rm2;
    auto mul_r = evaluate(mul);
    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j < 5; ++j)
        {
            assert(mul_r(i, j) == rm1(i, j) * rm2(i, j));
        }
    }

    rm1 = gen_matrix<int>(111, 113, 4, 2);
    rm2 = gen_matrix<int>(111, 113, 1, 1);
    rm1 = rm1.subMatrix(31, 35, 17, 22);
    rm2 = rm2.subMatrix(41, 45, 27, 32);
    mul = rm1 * rm2;
    mul_r = evaluate(mul);
    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j < 5; ++j)
        {
            assert(mul_r(i, j) == rm1(i, j) * rm2(i, j));
        }
    }
    cout << "done" << endl;
}

void test_mul2()
{
    cout << "Test element mul case 2 ...\t";
    auto rm1 = gen_matrix<int>(4, 5, 0, 1);
    auto mul = rm1 * Scalar<int>(2);
    auto mul_r = evaluate(mul);
    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j < 5; ++j)
        {
            assert(mul_r(i, j) == rm1(i, j) * 2);
        }
    }

    rm1 = gen_matrix<int>(111, 113, 2, 3);
    rm1 = rm1.subMatrix(31, 35, 17, 22);
    mul = Scalar<int>(3) * rm1;

    mul_r = evaluate(mul);
    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j < 5; ++j)
        {
            assert(mul_r(i, j) == rm1(i, j) * 3);
        }
    }
    cout << "done" << endl;
}

void test_mul3()
{
    cout << "Test element mul case 3 ...\t";
    {
        auto rm1 = gen_matrix<int>(4, 5, 0, 1);
        auto rm2 = gen_matrix<int>(4, 5, 1, 3);
        auto mul = rm1 * rm2;
        auto mul2 = rm1 * rm2;

        assert(mul == mul2);

        auto handle1 = mul.evalRegister();
        auto handle2 = mul.evalRegister();
        EvalPlan<CPU>::eval();

        auto cm1 = handle1.data();
        auto cm2 = handle2.data();
        assert(cm1 == cm2);
    }
    {
        auto rm1 = gen_matrix<int>(4, 5, 0, 1);
        auto rm2 = gen_matrix<int>(4, 5, 1, 3);
        auto mul = rm1 * rm2;
        auto mul2 = mul;

        assert(mul == mul2);

        auto handle1 = mul.evalRegister();
        auto handle2 = mul2.evalRegister();
        EvalPlan<CPU>::eval();

        auto& cm1 = handle1.data();
        auto& cm2 = handle2.data();
        assert(cm1 == cm2);
    }
    cout << "done" << endl;
}

void test_mul4()
{
    cout << "Test element mul case 4 ...\t";
    {
        auto rm1 = gen_matrix<int>(4, 5, 0, 1);
        auto rm2 = gen_batch_matrix<int>(4, 5, 7, 3, 2);
        auto mul = rm1 * rm2;
        auto mul_r = evaluate(mul);
        for (size_t b = 0; b < 7; ++b)
        {
            for (size_t i = 0; i < 4; ++i)
            {
                for (size_t j = 0; j < 5; ++j)
                {
                    assert(mul_r[b](i, j) == rm1(i, j) * rm2[b](i, j));
                }
            }
        }
    }

    {
        auto rm1 = gen_matrix<int>(4, 5, 0, 1);
        auto rm2 = gen_batch_matrix<int>(4, 5, 7, 3, 2);
        auto mul = rm2 * rm1;
        auto mul_r = evaluate(mul);
        for (size_t b = 0; b < 7; ++b)
        {
            for (size_t i = 0; i < 4; ++i)
            {
                for (size_t j = 0; j < 5; ++j)
                {
                    assert(mul_r[b](i, j) == rm1(i, j) * rm2[b](i, j));
                }
            }
        }
    }
    cout << "done" << endl;
}

void test_mul5()
{
    cout << "Test element mul case 5 ...\t";
    {
        auto rm1 = gen_batch_matrix<int>(4, 5, 7, 0, 1);
        auto rm2 = Scalar<int>(13);
        auto mul = rm1 * rm2;
        auto mul_r = evaluate(mul);
        for (size_t b = 0; b < 7; ++b)
        {
            for (size_t i = 0; i < 4; ++i)
            {
                for (size_t j = 0; j < 5; ++j)
                {
                    assert(mul_r[b](i, j) == rm1[b](i, j) * 13);
                }
            }
        }
    }

    {
        auto rm1 = gen_batch_matrix<int>(4, 5, 7, 0, 1);
        auto rm2 = Scalar<int>(13);
        auto mul = rm2 * rm1;
        auto mul_r = evaluate(mul);
        for (size_t b = 0; b < 7; ++b)
        {
            for (size_t i = 0; i < 4; ++i)
            {
                for (size_t j = 0; j < 5; ++j)
                {
                    assert(mul_r[b](i, j) == rm1[b](i, j) * 13);
                }
            }
        }
    }
    cout << "done" << endl;
}

void test_element_mul()
{
    std::cout << "Mul Tests Begin\n";
    test_mul1();
    test_mul2();
    test_mul3();
    test_mul4();
    test_mul5();
    std::cout << "Mul Tests End" << std::endl;
}
int main()
{
    test_element_mul();
    return 0;
}