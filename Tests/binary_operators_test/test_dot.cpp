// #include "test_dot.h"
// #include "../facilities/data_gen.h"
#include <cassert>
#include <iostream>

#include <metann/operators/binary_operators.hpp>
using namespace std;
using namespace metann;

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

void test_dot_1() {
    cout << "Test dot case 1 ...\t";
    auto rm = gen_matrix<int>(4, 5, 0, 1);
    auto cm = gen_matrix<int>(5, 3, 3, 2);
    auto mul = dot(rm, cm);
    auto mul_r = evaluate(mul);
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            int h = 0;
            for (size_t k = 0; k < 5; ++k) {
                h += rm(i, k) * cm(k, j);
            }
            assert(h == mul_r(i, j));
        }
    }

    auto rm2 = gen_matrix<int>(111, 113, 0, 1);
    auto cm2 = gen_matrix<int>(111, 113, 2, 3);
    rm2 = rm2.subMatrix(31, 35, 17, 22);
    cm2 = cm2.subMatrix(31, 36, 41, 44);
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 5; ++j) {
            rm2.setValue(i, j, rm(i, j));
        }
    }
    for (size_t i = 0; i < 5; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            cm2.setValue(i, j, cm(i, j));
        }
    }
    auto mul2 = dot(rm2, cm2);
    auto mul2_r = evaluate(mul2);

    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            assert(mul2_r(i, j) == mul_r(i, j));
        }
    }
    cout << "done" << endl;
}

void test_dot_2() {
    cout << "Test dot case 2 ...\t";
    auto rm = gen_batch_matrix<int>(4, 5, 3, 0, 1);
    auto cm = gen_matrix<int>(5, 3, 3, 2);
    auto mul = dot(rm, cm);
    auto mul_r = evaluate(mul);
    for (size_t b = 0; b < 3; ++b) {
        auto rm1 = rm[b];
        for (size_t i = 0; i < 4; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                int h = 0;
                for (size_t k = 0; k < 5; ++k) {
                    h += rm1(i, k) * cm(k, j);
                }
                assert(h == mul_r[b](i, j));
            }
        }
    }
    cout << "done" << endl;
}

void test_dot_3() {
    cout << "Test dot case 3 ...\t";
    auto rm = gen_matrix<int>(4, 5, 0, 1);
    auto cm = gen_batch_matrix<int>(5, 3, 3, 3, 2);
    auto mul = dot(rm, cm);
    auto mul_r = evaluate(mul);
    for (size_t b = 0; b < 3; ++b) {
        auto cm1 = cm[b];
        for (size_t i = 0; i < 4; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                int h = 0;
                for (size_t k = 0; k < 5; ++k) {
                    h += rm(i, k) * cm1(k, j);
                }
                assert(h == mul_r[b](i, j));
            }
        }
    }
    cout << "done" << endl;
}

int main() {
    std::cout << "Dot Tests Starts" << std::endl;
    test_dot_1();
    test_dot_2();  // BatchMatrix dot matrix
    test_dot_3();
    std::cout << "Dot Tests Ends" << std::endl;
}
