

#include <cassert>
#include <cmath>
#include <iostream>

#include <metann/operators/binary_operators.hpp>
using namespace metann;
using namespace std;

using CheckElement = float;
using CheckDevice = CPU;

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

void test_add1() {
    cout << "Test add case 1 ...\t";
    auto rm1 = gen_matrix<int>(4, 5, 0, 1);
    auto rm2 = gen_matrix<int>(4, 5, 2, 3);
    auto add = rm1 + rm2;
    auto add_r = evaluate(add);

    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 5; ++j) {
            assert(add_r(i, j) == rm1(i, j) + rm2(i, j));
        }
    }

    rm1 = gen_matrix<int>(111, 113, 1, 2);
    rm2 = gen_matrix<int>(111, 113, 2, 3);
    rm1 = rm1.subMatrix(31, 35, 17, 22);
    rm2 = rm2.subMatrix(41, 45, 27, 32);
    add = rm1 + rm2;
    add_r = evaluate(add);
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 5; ++j) {
            assert(add_r(i, j) == rm1(i, j) + rm2(i, j));
        }
    }
    cout << "done" << endl;
}

void test_add2() {
    cout << "Test add case 2 ...\t";
    auto rm1 = gen_matrix<int>(4, 5, 0, 1);
    auto add = rm1 + Scalar<int, CPU>(2);
    auto add_r = evaluate(add);
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 5; ++j) {
            assert(add_r(i, j) == rm1(i, j) + 2);
        }
    }

    rm1 = gen_matrix<int>(111, 113, 2, 3);
    rm1 = rm1.subMatrix(31, 35, 17, 22);
    add = Scalar<int>(3) + rm1;
    add_r = evaluate(add);
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 5; ++j) {
            assert(add_r(i, j) == rm1(i, j) + 3);
        }
    }
    cout << "done" << endl;
}

void test_add3() {
    cout << "Test add case 3 ...\t";
    auto rm1 = make_trivial_matrix<int, CheckDevice>(2, 10, 3);
    auto rm2 = make_trivial_matrix<int, CheckDevice>(2, 10, 5);
    auto add = rm1 + rm2;
    auto add_r = evaluate(add);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 10; ++j) {
            assert(add_r(i, j) == 8);
        }
    }
    cout << "done" << endl;
}

void test_add4() {
    cout << "Test add case 4 ...\t";
    auto rm1 = gen_batch_matrix<int>(4, 5, 7, 1, -1);
    auto rm2 = gen_matrix<int>(4, 5, 2, 3);
    auto add = rm1 + rm2;
    auto add_r = evaluate(add);

    assert(add.rowNum() == 4);
    assert(add.colNum() == 5);
    assert(add.batchNum() == 7);

    for (size_t b = 0; b < 7; ++b) {
        for (size_t i = 0; i < 4; ++i) {
            for (size_t j = 0; j < 5; ++j) {
                assert(add_r[b](i, j) == rm1[b](i, j) + rm2(i, j));
            }
        }
    }

    auto add2 = rm2 + rm1;
    add_r = evaluate(add2);
    assert(add2.rowNum() == 4);
    assert(add2.colNum() == 5);
    assert(add2.batchNum() == 7);

    for (size_t b = 0; b < 7; ++b) {
        for (size_t i = 0; i < 4; ++i) {
            for (size_t j = 0; j < 5; ++j) {
                assert(add_r[b](i, j) == rm1[b](i, j) + rm2(i, j));
            }
        }
    }
    cout << "done" << endl;
}

void test_add5() {
    cout << "Test add case 5 ...\t";
    auto rm1 = gen_batch_matrix<int>(4, 5, 7, 1, -1);
    auto rm2 = gen_batch_matrix<int>(4, 5, 7, 2, 3);
    auto add = rm1 + rm2;
    auto add_r = evaluate(add);

    assert(add.rowNum() == 4);
    assert(add.colNum() == 5);
    assert(add.batchNum() == 7);

    for (size_t b = 0; b < 7; ++b) {
        for (size_t i = 0; i < 4; ++i) {
            for (size_t j = 0; j < 5; ++j) {
                assert(add_r[b](i, j) == rm1[b](i, j) + rm2[b](i, j));
            }
        }
    }
    cout << "done" << endl;
}

void test_add6() {
    cout << "Test add case 6 ...\t";
    auto rm1 = gen_batch_matrix<int>(4, 5, 7, 1, -1);
    auto add = rm1 + Scalar<int>(3);
    auto add_r = evaluate(add);

    assert(add.rowNum() == 4);
    assert(add.colNum() == 5);
    assert(add.batchNum() == 7);

    for (size_t b = 0; b < 7; ++b) {
        for (size_t i = 0; i < 4; ++i) {
            for (size_t j = 0; j < 5; ++j) {
                assert(add_r[b](i, j) == rm1[b](i, j) + 3);
            }
        }
    }

    auto add2 = Scalar<int>(3) + rm1;
    add_r = evaluate(add2);

    assert(add2.rowNum() == 4);
    assert(add2.colNum() == 5);
    assert(add2.batchNum() == 7);

    for (size_t b = 0; b < 7; ++b) {
        for (size_t i = 0; i < 4; ++i) {
            for (size_t j = 0; j < 5; ++j) {
                assert(add_r[b](i, j) == rm1[b](i, j) + 3);
            }
        }
    }
    cout << "done" << endl;
}

int main() {
    std::cout << "Add Tests Start" << std::endl;
    test_add1();
    test_add2();
    test_add3();
    test_add4();
    test_add5();
    test_add6();
    std::cout << "Add Tests End" << std::endl;
}
