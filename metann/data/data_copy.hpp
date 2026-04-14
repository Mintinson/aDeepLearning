//
// Created by asus on 2025/2/4.
//

#ifndef DATA_COPY_HPP
#define DATA_COPY_HPP
#include "matrix.hpp"

namespace metann {
template <typename Element>
void data_copy(const Matrix<Element, CPU>& src, Matrix<Element, CPU>& dst) {
    const std::size_t rowNum = src.rowNum();
    const std::size_t colNum = src.colNum();

    if ((rowNum != dst.rowNum()) || (colNum != dst.colNum())) {
        throw std::runtime_error("data_copy: matrix sizes don't match");
    }
    const auto memSrc = lower_access(src);
    auto memDst = lower_access(dst);
    const std::size_t srcPackNum = memSrc.rowLen();
    const std::size_t dstPackNum = memDst.rowLen();

    const Element* r1 = memSrc.rawMemory();
    Element* r = memDst.mutableRawMemory();

    if ((srcPackNum == colNum) && (dstPackNum == colNum)) {
        memcpy(r, r1, sizeof(Element) * rowNum * colNum);
    } else {
        for (size_t i = 0; i < rowNum; ++i) {
            memcpy(r, r1, sizeof(Element) * colNum);
            r += dstPackNum;
            r1 += srcPackNum;
        }
    }
}
}  // namespace metann

#endif  // DATA_COPY_HPP
