#include <metann/data/data_category.hpp>
#include  <metann/data/array.hpp>
#include <metann/data/data_device.hpp>
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

static void test_array_1()
{
    static_assert(std::is_default_constructible_v<Batch<float, CPU, CategoryTags::Matrix>>, "  456");
    static_assert(DataConcept<Matrix<CheckElement, CheckDevice>>, "error");
    static_assert(DeviceConcept<CheckDevice>, "error");
    static_assert(metann::IsMatrix_v<Matrix<CheckElement, CheckDevice>>, "Test Error");
    std::cout << typeid(Matrix<CheckElement, CheckDevice>::ElementType).name() << std::endl;
    std::cout << typeid(Matrix<CheckElement, CheckDevice>).name() << std::endl;
    static_assert(DeviceConcept<CheckDevice>, "456");
    // std::cout << "Test array case 1 (matrix array)...\t";
    static_assert(metann::IsBatchMatrix_v<Array<Matrix<CheckElement, CheckDevice>>>, "Test Error");
    static_assert(metann::IsBatchMatrix_v<Array<Matrix<CheckElement, CheckDevice>>&>, "Test Error");
    static_assert(IsBatchMatrix_v<Array<Matrix<CheckElement, CheckDevice>>&&>, "Test Error");
    static_assert(IsBatchMatrix_v<const Array<Matrix<CheckElement, CheckDevice>>&>, "Test Error");
    static_assert(IsBatchMatrix_v<const Array<Matrix<CheckElement, CheckDevice>>&&>, "Test Error");
    //
    auto rm1 = Array<Matrix<CheckElement, CheckDevice>>(10, 20);
    assert(rm1.batchNum() == 0);
    assert(rm1.empty());
    //
    int c = 0;
    auto me1 = Matrix<CheckElement, CheckDevice>(10, 20);
    auto me2 = Matrix<CheckElement, CheckDevice>(10, 20);
    auto me3 = Matrix<CheckElement, CheckDevice>(10, 20);
    for (size_t i = 0; i < 10; ++i)
    {
        for (size_t j = 0; j < 20; ++j)
        {
            me1.setValue(i, j, static_cast<float>(c++));
            me2.setValue(i, j, static_cast<float>(c++));
            me3.setValue(i, j, static_cast<float>(c++));
        }
    }
    rm1.push_back(me1);
    rm1.push_back(me2);
    rm1.push_back(me3);
    assert(rm1.batchNum() == 3);
    assert(!rm1.empty());
    //
    const auto evalHandle = rm1.evalRegister();
    EvalPlan<CPU>::eval();
    auto rm2 = evalHandle.data();

    for (size_t i = 0; i < 10; ++i)
    {
        for (size_t j = 0; j < 20; ++j)
        {
            assert(rm1[0](i, j) == me1(i, j));
            assert(rm1[1](i, j) == me2(i, j));
            assert(rm1[2](i, j) == me3(i, j));
        }
    }
    std::cout << "done" << std::endl;
}

void TestArray2()
{
    std::cout << "Test array case 2 (scalar array)...\t";
    static_assert(IsBatchScalar_v<Array<Scalar<CheckElement, CheckDevice>>>, "Test Error");
    static_assert(IsBatchScalar_v<Array<Scalar<CheckElement, CheckDevice>>&>, "Test Error");
    static_assert(IsBatchScalar_v<Array<Scalar<CheckElement, CheckDevice>>&&>, "Test Error");
    static_assert(IsBatchScalar_v<const Array<Scalar<CheckElement, CheckDevice>>&>, "Test Error");
    static_assert(IsBatchScalar_v<const Array<Scalar<CheckElement, CheckDevice>>&&>, "Test Error");

    auto rm1 = Array<Scalar<CheckElement, CheckDevice>>();
    assert(rm1.batchNum() == 0);
    assert(rm1.empty());

    rm1.push_back(Scalar<CheckElement>(3));
    rm1.push_back(Scalar<CheckElement>(8));
    rm1.push_back(Scalar<CheckElement>(2));
    assert(rm1.batchNum() == 3);
    assert(!rm1.empty());

    auto evalHandle = rm1.evalRegister();
    EvalPlan<CPU>::eval();
    auto rm2 = evalHandle.data();

    assert(rm2[0] == 3);
    assert(rm2[1] == 8);
    assert(rm2[2] == 2);
    std::cout << "done" << std::endl;
}

void TestArray3()
{
    std::cout << "Test array case 3 ...\t";

    {
        std::vector<Matrix<CheckElement, CheckDevice>> check;
        check.emplace_back(10, 16);
        check.emplace_back(10, 16);
        check.emplace_back(10, 16);

        auto tmp = make_array(check.begin(), check.end());
        assert(tmp.rowNum() == 10);
        assert(tmp.colNum() == 16);
        assert(tmp.batchNum() == 3);
    }
    // {
    //     std::vector<Scalar<int, CheckDevice>> check{
    //         Scalar<int, CheckDevice>{3},
    //         Scalar<int, CheckDevice>{5},
    //         Scalar<int, CheckDevice>{8},
    //         Scalar<int, CheckDevice>{6}
    //     };
    //     auto tmp = make_array(check.begin(), check.end());
    //     assert(tmp.batchNum() == 4);
    //     assert(tmp[0].value() == 3);
    //     assert(tmp[1].value() == 5);
    //     assert(tmp[2].value() == 8);
    //     assert(tmp[3].value() == 6);
    // }
    std::cout << "done" << std::endl;
}

int main()
{
    std::cout << "Hello world!" << std::endl;

    std::cout << std::format("Hello world!") << std::endl;
    std::cout << metann::IsBatchMatrix_v<A&> << std::endl;
    metann::Matrix<float> mat{ 100, 200 };
    test_array_1();
    TestArray2();
    TestArray3();
}
