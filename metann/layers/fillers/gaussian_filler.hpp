//
// Created by asus on 2025/2/4.
//

#ifndef GAUSSIAN_FILLER_HPP
#define GAUSSIAN_FILLER_HPP
#include <stdexcept>

#include "../../data/allocator.hpp"
#include "../../policy/policy.hpp"
#include "../policies/init_policy.hpp"
#include "fill_with_dist.hpp"

namespace metann {
template <typename PolicyCont = PolicyContainer<>>
class GaussianFiller {
    using RandomEngine = typename details::PolicySelect_t<InitPolicy, PolicyCont>::RandEngine;

public:
    GaussianFiller(const double mean, const double sigma, unsigned seed = std::random_device{}())
        : m_engine(seed)
        , m_mean(mean)
        , m_sigma(sigma) {
        if (m_sigma <= 0) {
            throw std::runtime_error("Invalid sigma.");
        }
    }

    template <DataConcept DataType>
    void fill(DataType& data, std::size_t, std::size_t) {
        using ElementType = typename DataType::ElementType;
        std::normal_distribution<ElementType> dist{static_cast<ElementType>(m_mean), static_cast<ElementType>(m_sigma)};
        fill_with_dist(data, dist, m_engine);
    }

private:
    RandomEngine m_engine;
    double m_mean;
    double m_sigma;
};
}  // namespace metann

#endif  // GAUSSIAN_FILLER_HPP
