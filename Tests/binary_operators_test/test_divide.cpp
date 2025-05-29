#include <metann/operators/binary_operators.hpp>
#include <cmath>
#include <cassert>
#include <iostream>

using namespace metann;
using namespace std;


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

void test_div1()
{
    cout << "Test divide case 1 ...\t";
    auto rm1 = gen_matrix<float>(4, 5, 1, 1);
    auto rm2 = gen_matrix<float>(4, 5, 2, 2);
    auto div = rm1 / rm2;
    auto div_r = evaluate(div);
    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j < 5; ++j)
        {
            assert(fabs(div_r(i, j) - rm1(i, j) / rm2(i, j)) < 0.001);
        }
    }

    rm1 = gen_matrix<float>(111, 113, 1, 1);
    rm2 = gen_matrix<float>(111, 113, 2, 3);
    rm1 = rm1.subMatrix(31, 35, 17, 22);
    rm2 = rm2.subMatrix(41, 45, 27, 32);
    div = rm1 / rm2;

    div_r = evaluate(div);
    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j < 5; ++j)
        {
            assert(fabs(div_r(i, j) - rm1(i, j) / rm2(i, j)) < 0.001);
        }
    }
    cout << "done" << endl;
}

void test_div2()
{
    cout << "Test divide case 2 ...\t";
    auto rm1 = gen_matrix<float>(4, 5, 1, 1);
    auto div = rm1 / Scalar<float>(2);
    auto div_r = evaluate(div);
    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j < 5; ++j)
        {
            assert(fabs(div_r(i, j) - rm1(i, j) / 2) < 0.001);
        }
    }

    rm1 = gen_matrix<float>(111, 113, 2, 3);
    rm1 = rm1.subMatrix(31, 35, 17, 22);
    auto div1 = Scalar<float>(3) / rm1;

    div_r = evaluate(div1);
    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j < 5; ++j)
        {
            assert(fabs(div_r(i, j) - 3 / rm1(i, j)) < 0.001);
        }
    }
    cout << "done" << endl;
}

void test_div3()
{
    cout << "Test divide case 3 ...\t";
    auto rm1 = gen_batch_matrix<float>(4, 5, 7, 1, 1);
    auto rm2 = gen_batch_matrix<float>(4, 5, 7, 2, 2);
    auto div = rm1 / rm2;
    auto div_r = evaluate(div);
    for (size_t b = 0; b < 7; ++b)
    {
        for (size_t i = 0; i < 4; ++i)
        {
            for (size_t j = 0; j < 5; ++j)
            {
                assert(fabs(div_r[b](i, j) - rm1[b](i, j) / rm2[b](i, j)) < 0.001);
            }
        }
    }

    cout << "done" << endl;
}
//
void test_div4()
{
    cout << "Test divide case 4 ...\t";
    {
        auto rm1 = gen_batch_matrix<float>(4, 5, 7, 1, 1);
        auto rm2 = gen_matrix<float>(4, 5, 2, 2);
        auto div = rm1 / rm2;
        auto div_r = evaluate(div);
        for (size_t b = 0; b < 7; ++b)
        {
            for (size_t i = 0; i < 4; ++i)
            {
                for (size_t j = 0; j < 5; ++j)
                {
                    assert(fabs(div_r[b](i, j) - rm1[b](i, j) / rm2(i, j)) < 0.001);
                }
            }
        }
    }

    {
        auto rm1 = gen_batch_matrix<float>(4, 5, 7, 1, 1);
        auto rm2 = gen_matrix<float>(4, 5, 2, 2);
        auto div = rm2 / rm1;
        auto div_r = evaluate(div);
        for (size_t b = 0; b < 7; ++b)
        {
            for (size_t i = 0; i < 4; ++i)
            {
                for (size_t j = 0; j < 5; ++j)
                {
                    assert(fabs(div_r[b](i, j) - rm2(i, j) / rm1[b](i, j)) < 0.001);
                }
            }
        }
    }

    cout << "done" << endl;
}
//
void test_div5()
{
    cout << "Test divide case 5 ...\t";
    {
        auto rm1 = gen_batch_matrix<float>(4, 5, 7, 1, 1);
        auto rm2 = Scalar<int>(3);
        auto div = rm1 / rm2;
        //class metann::BinaryOperator<struct metann::BinaryOperTags::Div,
        //    class metann::Batch<float, struct metann::CPU, struct metann::CategoryTags::Matrix>,
        //    class metann::Duplicate<class metann::TrivialMatrix<int, struct metann::CPU,
        //    struct metann::Scalar<int, struct metann::CPU> > > >;
        //std::cout << typeid(div).name() << std::endl;
        auto div_r = evaluate(div);
        for (size_t b = 0; b < 7; ++b)
        {
            for (size_t i = 0; i < 4; ++i)
            {
                for (size_t j = 0; j < 5; ++j)
                {
                    assert(fabs(div_r[b](i, j) - rm1[b](i, j) / 3) < 0.001);
                }
            }
        }
    }

    {
        auto rm1 = gen_batch_matrix<float>(4, 5, 7, 1, 1);
        auto rm2 = Scalar<int>(3);
        auto div = rm2 / rm1;
        auto div_r = evaluate(div);
        for (size_t b = 0; b < 7; ++b)
        {
            for (size_t i = 0; i < 4; ++i)
            {
                for (size_t j = 0; j < 5; ++j)
                {
                    assert(fabs(div_r[b](i, j) - 3 / rm1[b](i, j)) < 0.001);
                }
            }
        }
    }

    cout << "done" << endl;
}

void test_divide()
{
    std::cout << "Divide Tests Start" << std::endl;

    test_div1();
    test_div2();
    test_div3();
    test_div4();
    test_div5();
    std::cout << "Divide Tests Start" << std::endl;

}

int main()
{
    test_divide();
}
