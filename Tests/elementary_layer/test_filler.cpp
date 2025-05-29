//
// Created by asus on 2025/1/13.
//
#include <iostream>
#include <metann/layers/initializer.hpp>
#include <metann/layers/fillers/constant_filler.hpp>
#include <metann/layers/fillers/gaussian_filler.hpp>
#include <metann/layers/fillers/var_scale_filler.hpp>

using namespace metann;
using std::cout;
using std::endl;

void test_constant_filler1()
{
    cout << "test constant filler case 1 ...";

    ConstantFiller filler(0);
    Matrix<float, CPU> mat(11, 13);
    filler.fill(mat, 11, 13);
    for (size_t i = 0; i < 11; ++i)
    {
        for (size_t j = 0; j < 13; ++j)
        {
            assert(fabs(mat(i, j)) < 0.0001);
        }
    }

    ConstantFiller filler2(1.5f);
    Matrix<float, CPU> mat2(21, 33);
    filler2.fill(mat2, 21, 33);
    for (size_t i = 0; i < 21; ++i)
    {
        for (size_t j = 0; j < 33; ++j)
        {
            assert(fabs(mat2(i, j) - 1.5) < 0.0001);
        }
    }

    cout << "done" << endl;
}
void test_gaussian_filler1()
{
    cout << "test gaussian filler case 1 ...";

    GaussianFiller filler(1.5, 3.3);
    Matrix<float, CPU> mat(1000, 3000);
    filler.fill(mat, 1000, 3000);

    float mean = 0;
    for (size_t i = 0; i < mat.rowNum(); ++i)
    {
        for (size_t j = 0; j < mat.colNum(); ++j)
        {
            mean += mat(i, j);
        }
    }
    mean /= mat.rowNum() * mat.colNum();

    float var = 0;
    for (size_t i = 0; i < mat.rowNum(); ++i)
    {
        for (size_t j = 0; j < mat.colNum(); ++j)
        {
            var += (mat(i, j) - mean) * (mat(i, j) - mean);
        }
    }
    var /= mat.rowNum() * mat.colNum();

    // mean = 1.5, std = 3.3
    cout << "mean-delta = " << fabs(mean - 1.5) << " std-delta = " << fabs(sqrt(var) - 3.3) << ' ';
    cout << "done" << endl;
}

void test_xavier_filler1()
{
    cout << "test xavier filler case 1 ...";

    XavierFiller<PolicyContainer<UniformVarScale>> filler;
    Matrix<float, CPU> mat(100, 200);
    filler.fill(mat, 100, 200);

    float mean = 0;
    for (size_t i = 0; i < mat.rowNum(); ++i)
    {
        for (size_t j = 0; j < mat.colNum(); ++j)
        {
            mean += mat(i, j);
        }
    }
    mean /= mat.rowNum() * mat.colNum();

    float var = 0;
    for (size_t i = 0; i < mat.rowNum(); ++i)
    {
        for (size_t j = 0; j < mat.colNum(); ++j)
        {
            var += (mat(i, j) - mean) * (mat(i, j) - mean);
        }
    }
    var /= mat.rowNum() * mat.colNum();

    // std = 0.0816 (sqrt(1/150))
    cout << "mean-delta = " << fabs(mean) << " std-delta = " << fabs(sqrt(var) - 0.0816) << ' ';
    cout << "done" << endl;
}

int main()
{
    std::cout << "Test Filler ..." << std::endl;
    test_constant_filler1();
    test_gaussian_filler1();
    test_xavier_filler1();
    std::cout << "Test Filler done" << std::endl;
}
