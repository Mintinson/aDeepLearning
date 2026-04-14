//
// Created by asus on 2025/2/4.
//

#ifndef FILL_WITH_DIST_HPP
#define FILL_WITH_DIST_HPP
#include "../../data/matrix.hpp"

namespace metann {
template <typename Element, typename Dist, typename Engine>
void fill_with_dist(Matrix<Element, CPU>& data, Dist& dist, Engine& engin) {
    if (!data.availableForWrite()) {
        throw std::runtime_error("matrix is sharing weight, cannot fill-in");
    }
    auto mem = lower_access(data);
    const std::size_t rowNum = data.rowNum();
    const std::size_t colNum = data.colNum();
    const std::size_t tgtPackNum = mem.rowLen();

    auto r = mem.mutableRawMemory();

    for (std::size_t i = 0; i < rowNum; ++i) {
        for (std::size_t j = 0; j < colNum; ++j) {
            r[j] = static_cast<Element>(dist(engin));
        }
        r += tgtPackNum;
    }
}
}  // namespace metann

#endif  // FILL_WITH_DIST_HPP
