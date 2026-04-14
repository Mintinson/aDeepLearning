//
// Created by asus on 2025/2/4.
//

#ifndef CONSTANT_FILLER_HPP
#define CONSTANT_FILLER_HPP
#include <cstddef>

#include "../../data/matrix.hpp"
#include "../dynamic_data.hpp"

namespace metann {
namespace details {
namespace constant_filler {
template <typename Element>
void fill(Matrix<Element, CPU>& mat, const double value) {
    if (!mat.availableForWrite()) {
        throw std::runtime_error("matrix is sharing weight, cannot fill-in.");
    }
    auto mem = lower_access(mat);
    const std::size_t rowNum = mat.rowNum();
    const std::size_t colNum = mat.colNum();
    const std::size_t tgtPackNum = mem.rowLen();
    auto r = mem.mutableRawMemory();

    for (std::size_t i = 0; i < rowNum; ++i) {
        for (std::size_t j = 0; j < colNum; ++j) {
            r[j] = static_cast<Element>(value);
        }
        r += tgtPackNum;
    }
}
}  // namespace constant_filler
}  // namespace details

class ConstantFiller {
public:
    explicit ConstantFiller(double val = 0.) : m_value{val} {}

    template <DataConcept DataType>
    void fill(DataType& data, std::size_t, std::size_t) {
        details::constant_filler::fill(data, m_value);
    }

private:
    double m_value;
};
}  // namespace metann

#endif  // CONSTANT_FILLER_HPP
