#ifndef MNIST_HPP
#define MNIST_HPP

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "batch.hpp"

namespace metann {
namespace details {
inline std::uint32_t read_big_endian_u32(std::istream& input) {
    unsigned char bytes[4];
    input.read(reinterpret_cast<char*>(bytes), 4);
    if (!input) {
        throw std::runtime_error("Failed to read MNIST header field.");
    }

    return (static_cast<std::uint32_t>(bytes[0]) << 24U) | (static_cast<std::uint32_t>(bytes[1]) << 16U) |
           (static_cast<std::uint32_t>(bytes[2]) << 8U) | static_cast<std::uint32_t>(bytes[3]);
}

inline std::size_t checked_image_buffer_size(std::size_t imageCount, std::size_t imageSize) {
    if (imageSize != 0 && imageCount > (std::numeric_limits<std::size_t>::max() / imageSize)) {
        throw std::runtime_error("MNIST image buffer size overflow.");
    }
    return imageCount * imageSize;
}
}  // namespace details

template <typename Element = float>
struct MnistDataset {
    std::size_t imageRows = 0;
    std::size_t imageCols = 0;
    std::vector<Element> images;
    std::vector<std::uint8_t> labels;

    [[nodiscard]] std::size_t sampleCount() const { return labels.size(); }

    [[nodiscard]] std::size_t imageSize() const { return imageRows * imageCols; }

    [[nodiscard]] const Element* imageData(const std::size_t sampleIdx) const {
        return images.data() + sampleIdx * imageSize();
    }
};

template <typename Element = float>
MnistDataset<Element> load_mnist_dataset(const std::string& imageFile,
                                         const std::string& labelFile,
                                         std::size_t maxSamples = 0,
                                         bool normalizeToUnit = true) {
    std::ifstream imageInput(imageFile, std::ios::binary);
    if (!imageInput) {
        throw std::runtime_error("Cannot open MNIST image file: " + imageFile);
    }

    std::ifstream labelInput(labelFile, std::ios::binary);
    if (!labelInput) {
        throw std::runtime_error("Cannot open MNIST label file: " + labelFile);
    }

    const std::uint32_t imageMagic = details::read_big_endian_u32(imageInput);
    const std::uint32_t imageCount = details::read_big_endian_u32(imageInput);
    const std::uint32_t imageRows = details::read_big_endian_u32(imageInput);
    const std::uint32_t imageCols = details::read_big_endian_u32(imageInput);

    const std::uint32_t labelMagic = details::read_big_endian_u32(labelInput);
    const std::uint32_t labelCount = details::read_big_endian_u32(labelInput);

    if (imageMagic != 2051U) {
        throw std::runtime_error("Invalid MNIST image magic number in: " + imageFile);
    }
    if (labelMagic != 2049U) {
        throw std::runtime_error("Invalid MNIST label magic number in: " + labelFile);
    }
    if (imageCount != labelCount) {
        throw std::runtime_error("MNIST image/label count mismatch.");
    }

    const std::size_t imageSize = static_cast<std::size_t>(imageRows) * static_cast<std::size_t>(imageCols);
    const std::size_t totalImageCount = static_cast<std::size_t>(imageCount);

    const std::size_t loadedCount =
        (maxSamples == 0) ? totalImageCount : std::min<std::size_t>(totalImageCount, maxSamples);

    const std::size_t rawImageBytes = details::checked_image_buffer_size(totalImageCount, imageSize);
    std::vector<std::uint8_t> rawImages(rawImageBytes);
    std::vector<std::uint8_t> rawLabels(totalImageCount);

    imageInput.read(reinterpret_cast<char*>(rawImages.data()), static_cast<std::streamsize>(rawImages.size()));
    if (!imageInput) {
        throw std::runtime_error("Failed to read full MNIST image data from: " + imageFile);
    }

    labelInput.read(reinterpret_cast<char*>(rawLabels.data()), static_cast<std::streamsize>(rawLabels.size()));
    if (!labelInput) {
        throw std::runtime_error("Failed to read full MNIST label data from: " + labelFile);
    }

    MnistDataset<Element> dataset;
    dataset.imageRows = static_cast<std::size_t>(imageRows);
    dataset.imageCols = static_cast<std::size_t>(imageCols);
    dataset.labels.assign(rawLabels.begin(), rawLabels.begin() + static_cast<std::ptrdiff_t>(loadedCount));
    dataset.images.resize(loadedCount * imageSize);

    const Element scale = normalizeToUnit ? static_cast<Element>(1.0 / 255.0) : static_cast<Element>(1.0);
    for (std::size_t i = 0; i < loadedCount; ++i) {
        const std::size_t sourceOffset = i * imageSize;
        const std::size_t targetOffset = i * imageSize;
        for (std::size_t j = 0; j < imageSize; ++j) {
            dataset.images[targetOffset + j] = static_cast<Element>(rawImages[sourceOffset + j]) * scale;
        }
    }

    return dataset;
}

inline std::vector<std::size_t> make_index_order(const std::size_t sampleCount) {
    std::vector<std::size_t> order(sampleCount);
    std::iota(order.begin(), order.end(), 0);
    return order;
}

template <typename Element>
Batch<Element, CPU, CategoryTags::Matrix> make_mnist_image_batch(const MnistDataset<Element>& dataset,
                                                                 const std::vector<std::size_t>& order,
                                                                 const std::size_t begin,
                                                                 const std::size_t batchSize) {
    if (begin >= order.size()) {
        throw std::runtime_error("make_mnist_image_batch: begin index out of range.");
    }

    const std::size_t actualBatch = std::min<std::size_t>(batchSize, order.size() - begin);
    const std::size_t imageSize = dataset.imageSize();

    Batch<Element, CPU, CategoryTags::Matrix> batch(actualBatch, 1, imageSize);

    for (std::size_t b = 0; b < actualBatch; ++b) {
        const std::size_t sampleIdx = order[begin + b];
        if (sampleIdx >= dataset.sampleCount()) {
            throw std::runtime_error("make_mnist_image_batch: sample index out of range.");
        }

        const Element* src = dataset.imageData(sampleIdx);
        for (std::size_t f = 0; f < imageSize; ++f) {
            batch.setValue(b, 0, f, src[f]);
        }
    }

    return batch;
}

template <typename Element>
Batch<Element, CPU, CategoryTags::Matrix> make_mnist_one_hot_batch(const MnistDataset<Element>& dataset,
                                                                   const std::vector<std::size_t>& order,
                                                                   const std::size_t begin,
                                                                   const std::size_t batchSize,
                                                                   const std::size_t classNum = 10) {
    if (begin >= order.size()) {
        throw std::runtime_error("make_mnist_one_hot_batch: begin index out of range.");
    }

    const std::size_t actualBatch = std::min<std::size_t>(batchSize, order.size() - begin);
    Batch<Element, CPU, CategoryTags::Matrix> batch(actualBatch, 1, classNum);

    for (std::size_t b = 0; b < actualBatch; ++b) {
        const std::size_t sampleIdx = order[begin + b];
        if (sampleIdx >= dataset.sampleCount()) {
            throw std::runtime_error("make_mnist_one_hot_batch: sample index out of range.");
        }

        for (std::size_t c = 0; c < classNum; ++c) {
            batch.setValue(b, 0, c, static_cast<Element>(0));
        }

        const auto label = static_cast<std::size_t>(dataset.labels[sampleIdx]);
        if (label >= classNum) {
            throw std::runtime_error("make_mnist_one_hot_batch: label is out of class range.");
        }
        batch.setValue(b, 0, label, static_cast<Element>(1));
    }

    return batch;
}
}  // namespace metann

#endif  // MNIST_HPP
