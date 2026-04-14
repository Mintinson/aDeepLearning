#include <format>
#include <iostream>

#include <metann/data/data_category.hpp>
#include <metann/data/data_device.hpp>
#include <metann/data/duplicate.hpp>
#include <metann/data/matrix.hpp>

class A {};

namespace metann {
template <>
constexpr bool IsBatchMatrixHelper_v<A> = true;
}  // namespace metann

using namespace metann;
using CheckElement = float;
using CheckDevice = CPU;

void test_batch_scalar1() {
    std::cout << "Test batch scalar case 1...\t";
    static_assert(IsBatchScalar_v<Batch<CheckElement, CheckDevice, CategoryTags::Scalar>>, "Test Error");
    static_assert(IsBatchScalar_v<Batch<CheckElement, CheckDevice, CategoryTags::Scalar>&>, "Test Error");
    static_assert(IsBatchScalar_v<Batch<CheckElement, CheckDevice, CategoryTags::Scalar>&&>, "Test Error");
    static_assert(IsBatchScalar_v<const Batch<CheckElement, CheckDevice, CategoryTags::Scalar>&>, "Test Error");
    static_assert(IsBatchScalar_v<const Batch<CheckElement, CheckDevice, CategoryTags::Scalar>&&>, "Test Error");

    Batch<CheckElement, CheckDevice, CategoryTags::Scalar> check;
    assert(check.batchNum() == 0);

    check = Batch<CheckElement, CheckDevice, CategoryTags::Scalar>(13);
    assert(check.batchNum() == 13);

    int c = 0;
    for (size_t i = 0; i < 13; ++i) {
        check.setValue(i, static_cast<float>(c++));
    }

    const Batch<CheckElement, CheckDevice, CategoryTags::Scalar> c2 = check;
    c = 0;
    for (size_t i = 0; i < 13; ++i) {
        assert(c2[i] == static_cast<float>(c++));
    }

    auto evalHandle = check.evalRegister();
    auto cm = evalHandle.data();

    for (size_t i = 0; i < cm.batchNum(); ++i) {
        assert(cm[i] == check[i]);
    }
    std::cout << "done" << std::endl;
}

void test_batch_matrix1() {
    std::cout << "Test batch matrix case 1...\t";
    static_assert(IsBatchMatrix_v<Batch<int, CheckDevice, CategoryTags::Matrix>>, "Test Error");
    static_assert(IsBatchMatrix_v<Batch<int, CheckDevice, CategoryTags::Matrix>&>, "Test Error");
    static_assert(IsBatchMatrix_v<Batch<int, CheckDevice, CategoryTags::Matrix>&&>, "Test Error");
    static_assert(IsBatchMatrix_v<const Batch<int, CheckDevice, CategoryTags::Matrix>&>, "Test Error");
    static_assert(IsBatchMatrix_v<const Batch<int, CheckDevice, CategoryTags::Matrix>&&>, "Test Error");

    Batch<int, CheckDevice, CategoryTags::Matrix> data(10, 13, 35);
    assert(data.availableForWrite());
    assert(data.batchNum() == 10);
    assert(data.rowNum() == 13);
    assert(data.colNum() == 35);
    for (size_t i = 0; i < 10; ++i) {
        for (size_t j = 0; j < 13; ++j) {
            for (size_t k = 0; k < 35; ++k) {
                data.setValue(i, j, k, static_cast<int>(i * 1000 + j * 100 + k));
            }
        }
    }

    for (size_t i = 0; i < 10; ++i) {
        for (size_t j = 0; j < 13; ++j) {
            for (size_t k = 0; k < 35; ++k) {
                assert(data[i](j, k) == static_cast<int>(i * 1000 + j * 100 + k));
            }
        }
    }

    auto data2 = data.subMatrix(3, 7, 11, 22);
    assert(!data.availableForWrite());
    assert(!data2.availableForWrite());
    assert(data2.batchNum() == 10);
    assert(data2.rowNum() == 4);
    assert(data2.colNum() == 11);

    for (size_t i = 0; i < 10; ++i) {
        for (size_t j = 3; j < 7; ++j) {
            for (size_t k = 11; k < 22; ++k) {
                assert(data2[i](j - 3, k - 11) == (int)(i * 1000 + j * 100 + k));
            }
        }
    }
    std::cout << "done" << std::endl;
}

void test_batch_matrix2() {
    std::cout << "Test batch matrix case 2...\t";
    static_assert(IsBatchMatrix_v<Batch<CheckElement, CheckDevice, CategoryTags::Matrix>>, "Test Error");
    static_assert(IsBatchMatrix_v<Batch<CheckElement, CheckDevice, CategoryTags::Matrix>&>, "Test Error");
    static_assert(IsBatchMatrix_v<Batch<CheckElement, CheckDevice, CategoryTags::Matrix>&&>, "Test Error");
    static_assert(IsBatchMatrix_v<const Batch<CheckElement, CheckDevice, CategoryTags::Matrix>&>, "Test Error");
    static_assert(IsBatchMatrix_v<const Batch<CheckElement, CheckDevice, CategoryTags::Matrix>&&>, "Test Error");

    auto rm1 = Batch<CheckElement, CheckDevice, CategoryTags::Matrix>(3, 10, 20);
    assert(rm1.batchNum() == 3);

    int c = 0;
    auto me1 = Matrix<CheckElement, CheckDevice>(10, 20);
    auto me2 = Matrix<CheckElement, CheckDevice>(10, 20);
    auto me3 = Matrix<CheckElement, CheckDevice>(10, 20);
    for (size_t i = 0; i < 10; ++i) {
        for (size_t j = 0; j < 20; ++j) {
            me1.setValue(i, j, (float)(c++));
            me2.setValue(i, j, (float)(c++));
            me3.setValue(i, j, (float)(c++));
            rm1.setValue(0, i, j, me1(i, j));
            rm1.setValue(1, i, j, me2(i, j));
            rm1.setValue(2, i, j, me3(i, j));
        }
    }

    for (size_t i = 0; i < 10; ++i) {
        for (size_t j = 0; j < 20; ++j) {
            assert(rm1[0](i, j) == me1(i, j));
            assert(rm1[1](i, j) == me2(i, j));
            assert(rm1[2](i, j) == me3(i, j));
        }
    }

    rm1 = rm1.subMatrix(3, 7, 11, 16);
    assert(rm1.rowNum() == 4);
    assert(rm1.colNum() == 5);
    assert(rm1.batchNum() == 3);
    me1 = me1.subMatrix(3, 7, 11, 16);
    me2 = me2.subMatrix(3, 7, 11, 16);
    me3 = me3.subMatrix(3, 7, 11, 16);
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 5; ++j) {
            assert(rm1[0](i, j) == me1(i, j));
            assert(rm1[1](i, j) == me2(i, j));
            assert(rm1[2](i, j) == me3(i, j));
        }
    }

    auto evalHandle = rm1.evalRegister();
    EvalPlan<CPU>::eval();
    auto rm2 = evalHandle.data();

    for (size_t k = 0; k < 3; ++k) {
        for (size_t i = 0; i < 4; ++i) {
            for (size_t j = 0; j < 5; ++j) {
                assert(rm1[k](i, j) == rm2[k](i, j));
            }
        }
    }
    std::cout << "done" << std::endl;
}

void test_duplicate1() {
    std::cout << "Test duplicate case 1 (matrix)...\t";
    static_assert(IsBatchMatrix_v<Duplicate<Matrix<CheckElement, CheckDevice>>>, "Test Error");
    static_assert(IsBatchMatrix_v<Duplicate<Matrix<CheckElement, CheckDevice>>&>, "Test Error");
    static_assert(IsBatchMatrix_v<Duplicate<Matrix<CheckElement, CheckDevice>>&&>, "Test Error");
    static_assert(IsBatchMatrix_v<const Duplicate<Matrix<CheckElement, CheckDevice>>&>, "Test Error");
    static_assert(IsBatchMatrix_v<const Duplicate<Matrix<CheckElement, CheckDevice>>&&>, "Test Error");

    auto me1 = Matrix<CheckElement, CheckDevice>(10, 20);
    int c = 0;
    for (size_t i = 0; i < 10; ++i) {
        for (size_t j = 0; j < 20; ++j) {
            me1.setValue(i, j, static_cast<float>(c++));
        }
    }
    auto rm1 = make_duplicate(13, me1);
    assert(rm1.batchNum() == 13);
    assert(rm1.rowNum() == 10);
    assert(rm1.colNum() == 20);

    auto rm2 = evaluate(rm1);
    for (size_t i = 0; i < 13; ++i) {
        for (size_t j = 0; j < 10; ++j) {
            for (size_t k = 0; k < 20; ++k) {
                assert(rm2[i](j, k) == me1(j, k));
            }
        }
    }
    std::cout << "done" << std::endl;
}

void test_duplicate2() {
    std::cout << "Test duplicate case 2 (scalar)...\t";
    static_assert(IsBatchScalar_v<Duplicate<Scalar<CheckElement, CheckDevice>>>, "Test Error");
    static_assert(IsBatchScalar_v<Duplicate<Scalar<CheckElement, CheckDevice>>&>, "Test Error");
    static_assert(IsBatchScalar_v<Duplicate<Scalar<CheckElement, CheckDevice>>&&>, "Test Error");
    static_assert(IsBatchScalar_v<const Duplicate<Scalar<CheckElement, CheckDevice>>&>, "Test Error");
    static_assert(IsBatchScalar_v<const Duplicate<Scalar<CheckElement, CheckDevice>>&&>, "Test Error");

    auto rm1 = Duplicate<Scalar<CheckElement, CheckDevice>>(Scalar<CheckElement, CheckDevice>{3}, 13);
    assert(rm1.size() == 13);

    auto evalHandle = rm1.evalRegister();
    EvalPlan<CPU>::eval();
    auto rm2 = evalHandle.data();

    for (size_t i = 0; i < 13; ++i) {
        assert(rm2[i] == 3);
    }
    std::cout << "done" << std::endl;
}

int main() {
    std::cout << std::format("Batch and Duplicate Tests Start!") << std::endl;

    std::cout << metann::IsBatchMatrix_v<A&> << std::endl;
    metann::Matrix<float> mat{100, 200};

    test_batch_scalar1();
    test_batch_matrix1();
    test_batch_matrix2();
    test_duplicate1();
    test_duplicate2();

    std::cout << std::format("Batch and Duplicate Tests End!") << std::endl;
}
