#include <metann/data/data_device.hpp>
#include <metann/data/data_category.hpp>
#include <metann/data/matrix.hpp>
#include <metann/data/duplicate.hpp>

#include <format>
#include <iostream>


class A
{
};

namespace metann
{
    template <>
    constexpr bool IsBatchMatrixHelper_v<A> = true;
}

using namespace metann;
using CheckElement = float;
using CheckDevice = CPU;


void test_one_hot_vector1()
{
    std::cout << "Test one-hot vector case 1...\t";
    static_assert(IsMatrix_v<OneHotVector<int, CheckDevice>>, "Test Error");
    static_assert(IsMatrix_v<OneHotVector<int, CheckDevice>&>, "Test Error");
    static_assert(IsMatrix_v<OneHotVector<int, CheckDevice>&&>, "Test Error");
    static_assert(IsMatrix_v<const OneHotVector<int, CheckDevice>&>, "Test Error");
    static_assert(IsMatrix_v<const OneHotVector<int, CheckDevice>&&>, "Test Error");

    auto rm = OneHotVector<int, CheckDevice>(100, 37);
    assert(rm.rowNum() == 1);
    assert(rm.colNum() == 100);
    assert(rm.hotPos() == 37);

    auto rm1 = evaluate(rm);
    for (size_t i = 0; i < 1; ++i)
    {
        for (size_t j = 0; j < 100; ++j)
        {
            if (j != 37)
            {
                assert(rm1(i, j) == 0);
            }
            else
            {
                assert(rm1(i, j) == 1);
            }
        }
    }
    std::cout << "done" << std::endl;
}

void test_one_hot_vector2()
{
    std::cout << "Test one-hot vector case 2...\t";
    auto rm1 = OneHotVector<int, CheckDevice>(100, 37);
    auto rm2 = OneHotVector<int, CheckDevice>(50, 16);

    auto evalRes1 = rm1.evalRegister();
    auto evalRes2 = rm2.evalRegister();

    EvalPlan<CPU>::eval();
    for (size_t j = 0; j < 100; ++j)
    {
        if (j == 37)
        {
            assert(evalRes1.data()(0, j) == 1);
        }
        else
        {
            assert(evalRes1.data()(0, j) == 0);
        }
    }

    for (size_t j = 0; j < 50; ++j)
    {
        if (j == 16)
        {
            assert(evalRes2.data()(0, j) == 1);
        }
        else
        {
            assert(evalRes2.data()(0, j) == 0);
        }
    }
    std::cout << "done" << std::endl;
}

void test_scalar1()
{
    std::cout << "Test scalar case 1...\t";
    static_assert(IsScalar_v<Scalar<int, CPU>>, "Test Error");
    static_assert(IsScalar_v<Scalar<int, CPU>&>, "Test Error");
    static_assert(IsScalar_v<Scalar<int, CPU>&&>, "Test Error");
    static_assert(IsScalar_v<const Scalar<int, CPU>&>, "Test Error");
    static_assert(IsScalar_v<const Scalar<int, CPU>&&>, "Test Error");

    Scalar<float, CPU> pi(3.1415926f);
    assert(pi == pi);

    auto x = pi.evalRegister();
    assert(x.data() == pi);
    std::cout << "done" << std::endl;
}

int main()
{

    std::cout << std::format("Other Data Type Tests Start!") << std::endl;
    std::cout << metann::IsBatchMatrix_v<A&> << std::endl;
    metann::Matrix<float> mat{ 100, 200 };

    test_scalar1();
    test_one_hot_vector1();
    test_one_hot_vector2();
    std::cout << std::format("Other Data Type Tests End!") << std::endl;


}
