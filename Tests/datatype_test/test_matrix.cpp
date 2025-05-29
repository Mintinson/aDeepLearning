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


void test_matrix1()
{
    std::cout << "Test general matrix case 1...\t";
    static_assert(IsMatrix_v<Matrix<CheckElement, CheckDevice>>, "Test Error");
    static_assert(IsMatrix_v<Matrix<CheckElement, CheckDevice>&>, "Test Error");
    static_assert(IsMatrix_v<Matrix<CheckElement, CheckDevice>&&>, "Test Error");
    static_assert(IsMatrix_v<const Matrix<CheckElement, CheckDevice>&>, "Test Error");
    static_assert(IsMatrix_v<const Matrix<CheckElement, CheckDevice>&&>, "Test Error");

    Matrix<CheckElement, CheckDevice> rm;
    assert(rm.rowNum() == 0);
    assert(rm.colNum() == 0);

    rm = Matrix<CheckElement, CheckDevice>(10, 20);
    assert(rm.rowNum() == 10);
    assert(rm.colNum() == 20);

    int c = 0;
    for (size_t i = 0; i < 10; ++i)
    {
        for (size_t j = 0; j < 20; ++j)
        {
            rm.setValue(i, j, static_cast<float>(c++));
        }
    }

    const Matrix<CheckElement, CheckDevice> rm2 = rm;
    c = 0;
    for (size_t i = 0; i < 10; ++i)
    {
        for (size_t j = 0; j < 20; ++j)
            assert(rm2(i, j) == c++);
    }

    auto rm3 = rm.subMatrix(3, 7, 5, 15);
    for (size_t i = 0; i < rm3.rowNum(); ++i)
    {
        for (size_t j = 0; j < rm3.colNum(); ++j)
        {
            assert(rm3(i, j) == rm(i + 3, j + 5));
        }
    }

    auto evalHandle = rm.evalRegister();
    auto cm = evalHandle.data();

    for (size_t i = 0; i < cm.rowNum(); ++i)
    {
        for (size_t j = 0; j < cm.colNum(); ++j)
        {
            assert(cm(i, j) == rm(i, j));
        }
    }
    std::cout << "done" << std::endl;
}

void test_matrix2()
{
    std::cout << "Test general matrix case 2...\t";
    auto rm1 = Matrix<CheckElement, CheckDevice>(10, 20);
    int c = 0;
    for (size_t i = 0; i < 10; ++i)
    {
        for (size_t j = 0; j < 20; ++j)
        {
            rm1.setValue(i, j, static_cast<float>(c++));
        }
    }

    auto rm2 = Matrix<CheckElement, CheckDevice>(3, 7);
    for (size_t i = 0; i < 3; ++i)
    {
        for (size_t j = 0; j < 7; ++j)
        {
            rm2.setValue(i, j, static_cast<float>(c++));
        }
    }
    std::cout << "done" << std::endl;
}

void test_trival_matrix1()
{
    std::cout << "Test trival matrix case 1...\t";
    static_assert(IsMatrix_v<TrivialMatrix<int, CheckDevice, Scalar<int, CPU>>>, "Test Error");
    static_assert(IsMatrix_v<TrivialMatrix<int, CheckDevice, Scalar<int, CPU>>&>, "Test Error");
    static_assert(IsMatrix_v<TrivialMatrix<int, CheckDevice, Scalar<int, CPU>>&&>, "Test Error");
    static_assert(IsMatrix_v<const TrivialMatrix<int, CheckDevice, Scalar<int, CPU>>&>, "Test Error");
    static_assert(IsMatrix_v<const TrivialMatrix<int, CheckDevice, Scalar<int, CPU>>&&>, "Test Error");

    auto rm = make_trivial_matrix<int, CheckDevice>(10, 20, 100);
    assert(rm.rowNum() == 10);
    assert(rm.colNum() == 20);

    const auto& evalHandle = rm.evalRegister();
    EvalPlan<CPU>::eval();

    auto rm1 = evalHandle.data();
    for (size_t i = 0; i < 10; ++i)
    {
        for (size_t j = 0; j < 20; ++j)
        {
            assert(rm1(i, j) == 100);
        }
    }

    std::cout << "done" << std::endl;
}

void test_trival_matrix2()
{
    std::cout << "Test trival matrix case 2...\t";
    auto rm1 = make_trivial_matrix<int, CheckDevice>(100, 10, 14);
    auto rm2 = make_trivial_matrix<int, CheckDevice>(10, 20, 35);

    const auto& evalRes1 = rm1.evalRegister();
    const auto& evalRes2 = rm2.evalRegister();

    EvalPlan<CPU>::eval();
    for (size_t j = 0; j < 100; ++j)
    {
        for (size_t k = 0; k < 10; ++k)
        {
            assert(evalRes1.data()(j, k) == 14);
        }
    }

    for (size_t j = 0; j < 10; ++j)
    {
        for (size_t k = 0; k < 20; ++k)
        {
            assert(evalRes2.data()(j, k) == 35);
        }
    }

    std::cout << "done" << std::endl;
}

void test_zero_matrix1()
{
    std::cout << "Test zero matrix case 1...\t";
    static_assert(IsMatrix_v<ZeroMatrix<int, CheckDevice>>, "Test Error");
    static_assert(IsMatrix_v<ZeroMatrix<int, CheckDevice>&>, "Test Error");
    static_assert(IsMatrix_v<ZeroMatrix<int, CheckDevice>&&>, "Test Error");
    static_assert(IsMatrix_v<const ZeroMatrix<int, CheckDevice>&>, "Test Error");
    static_assert(IsMatrix_v<const ZeroMatrix<int, CheckDevice>&&>, "Test Error");

    auto rm = ZeroMatrix<int, CheckDevice>(10, 20);
    assert(rm.rowNum() == 10);
    assert(rm.colNum() == 20);

    const auto& evalHandle = rm.evalRegister();
    EvalPlan<CPU>::eval();

    auto rm1 = evalHandle.data();
    for (size_t i = 0; i < 10; ++i)
    {
        for (size_t j = 0; j < 20; ++j)
        {
            assert(rm1(i, j) == 0);
        }
    }

    std::cout << "done" << std::endl;
}

int main()
{

    std::cout << std::format("Matrix Tests Start!") << std::endl;
    metann::Matrix<float> mat{ 100, 200 };

    test_matrix1();
    test_matrix2();
    test_trival_matrix1();
    test_trival_matrix2();
    test_zero_matrix1();

    std::cout << std::format("Matrix Tests End!") << std::endl;
}
