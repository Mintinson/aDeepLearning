#include <cassert>
#include <iostream>

#include <metann/operators/binary_operators.hpp>
using namespace metann;
using namespace std;

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

namespace {
void test_substract1() {
    cout << "Test substract case 1 ...\t";
    auto rm1 = gen_matrix<int>(4, 5, 0, 1);
    auto rm2 = gen_matrix<int>(4, 5, 3, -1);
    auto sub = rm1 - rm2;
    auto sub_r = evaluate(sub);
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 5; ++j) {
            assert(sub_r(i, j) == rm1(i, j) - rm2(i, j));
        }
    }

    rm1 = gen_matrix<int>(111, 113, 0, 1);
    rm2 = gen_matrix<int>(111, 113, 2, 3);
    rm1 = rm1.subMatrix(31, 35, 17, 22);
    rm2 = rm2.subMatrix(41, 45, 27, 32);
    sub = rm1 - rm2;
    sub_r = evaluate(sub);
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 5; ++j) {
            assert(sub_r(i, j) == rm1(i, j) - rm2(i, j));
        }
    }
    cout << "done" << endl;
}

void test_substract2() {
    cout << "Test substract case 2 ...\t";
    auto rm1 = gen_matrix<int>(4, 5, 3, -1);
    auto sub = rm1 - Scalar<int>(2);
    auto sub_r = evaluate(sub);
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 5; ++j) {
            assert(sub_r(i, j) == rm1(i, j) - 2);
        }
    }

    rm1 = gen_matrix<int>(111, 113, 2, 3);
    rm1 = rm1.subMatrix(31, 35, 17, 22);
    auto sub1 = Scalar<int>(3) - rm1;
    sub_r = evaluate(sub1);
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 5; ++j) {
            assert(sub_r(i, j) == 3 - rm1(i, j));
        }
    }
    cout << "done" << endl;
}

void test_substract3() {
    cout << "Test substract case 3 ...\t";
    {
        auto rm1 = gen_matrix<float>(4, 5, 1, 2);
        auto rm2 = gen_matrix<float>(4, 5, 3, 2);
        auto res = rm1 - rm2;
        auto res2 = rm1 - rm2;

        assert(res == res2);

        auto cm1 = evaluate(res);
        auto cm2 = evaluate(res);
        assert(cm1 == cm2);
    }
    {
        auto rm1 = gen_matrix<float>(4, 5, 1, 2);
        auto rm2 = gen_matrix<float>(4, 5, 3, 2);
        auto res = rm1 - rm2;
        auto res2 = res;

        assert(res == res2);

        auto handle1 = res.evalRegister();
        auto handle2 = res2.evalRegister();
        EvalPlan<CPU>::eval();

        auto cm1 = handle1.data();
        auto cm2 = handle2.data();
        assert(cm1 == cm2);
    }
    cout << "done" << endl;
}

void test_substract4() {
    cout << "Test substract case 4 ...\t";
    {
        auto rm1 = gen_batch_matrix<int>(4, 5, 7, 3, -1);
        auto sub = Scalar<int>(2) - rm1;
        auto sub_r = evaluate(sub);
        for (size_t b = 0; b < 7; ++b) {
            for (size_t i = 0; i < 4; ++i) {
                for (size_t j = 0; j < 5; ++j) {
                    assert(sub_r[b](i, j) == 2 - rm1[b](i, j));
                }
            }
        }
    }
    {
        auto rm1 = gen_batch_matrix<int>(4, 5, 7, 3, -1);
        auto sub = rm1 - Scalar<int>(2);
        auto sub_r = evaluate(sub);
        for (size_t b = 0; b < 7; ++b) {
            for (size_t i = 0; i < 4; ++i) {
                for (size_t j = 0; j < 5; ++j) {
                    assert(sub_r[b](i, j) == rm1[b](i, j) - 2);
                }
            }
        }
    }
    cout << "done" << endl;
}

void test_substract5() {
    cout << "Test substract case 5 ...\t";
    auto rm1 = gen_batch_matrix<int>(4, 5, 7, 3, -1);
    auto rm2 = gen_batch_matrix<int>(4, 5, 7, 13, 3);
    auto sub = rm1 - rm2;
    auto sub_r = evaluate(sub);
    for (size_t b = 0; b < 7; ++b) {
        for (size_t i = 0; i < 4; ++i) {
            for (size_t j = 0; j < 5; ++j) {
                assert(sub_r[b](i, j) == rm1[b](i, j) - rm2[b](i, j));
            }
        }
    }
    cout << "done" << endl;
}

void test_substract6() {
    cout << "Test substract case 6 ...\t";
    {
        auto rm1 = gen_batch_matrix<int>(4, 5, 7, 3, -1);
        auto rm2 = gen_matrix<int>(4, 5, 13, 3);
        auto sub = rm1 - rm2;
        auto sub_r = evaluate(sub);
        for (size_t b = 0; b < 7; ++b) {
            for (size_t i = 0; i < 4; ++i) {
                for (size_t j = 0; j < 5; ++j) {
                    assert(sub_r[b](i, j) == rm1[b](i, j) - rm2(i, j));
                }
            }
        }
    }

    {
        auto rm1 = gen_batch_matrix<int>(4, 5, 7, 3, -1);
        auto rm2 = gen_matrix<int>(4, 5, 13, 3);
        auto sub = rm2 - rm1;
        auto sub_r = evaluate(sub);
        for (size_t b = 0; b < 7; ++b) {
            for (size_t i = 0; i < 4; ++i) {
                for (size_t j = 0; j < 5; ++j) {
                    assert(sub_r[b](i, j) == rm2(i, j) - rm1[b](i, j));
                }
            }
        }
    }
    cout << "done" << endl;
}
}  // namespace

void test_substract() {
    std::cout << "Subtraction Tests starts" << std::endl;
    test_substract1();
    test_substract2();
    test_substract3();
    test_substract4();
    test_substract5();
    test_substract6();
    std::cout << "Subtraction Tests ends" << std::endl;
}

int main() {
    test_substract();
    return 0;
}