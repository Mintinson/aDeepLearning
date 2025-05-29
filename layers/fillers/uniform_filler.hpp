//
// Created by asus on 2025/2/4.
//

#ifndef UNIFORM_FILLER_HPP
#define UNIFORM_FILLER_HPP
#include <stdexcept>

#include "fill_with_dist.hpp"
#include "../../data/allocator.hpp"
#include "../../policy/policy.hpp"
#include "../policies/init_policy.hpp"

namespace metann
{
    template <typename PolicyCont = PolicyContainer<>>
    class UniformFiller
    {
        using RandomEngine = typename details::PolicySelect_t<InitPolicy, PolicyCont>::RandEngine;

    public:
        UniformFiller(const double min, const double max, unsigned seed = std::random_device{}())
            : m_engine(seed), m_min(min), m_max(max)
        {
            if (m_min >= m_max)
            {
                throw std::runtime_error("UniformFiller: min must be lesser than m_max");
            }
        }

        template <DataConcept DataType>
        void fill(DataType& data, std::size_t, std::size_t)
        {
            using ElementType = typename DataType::ElementType;
            using DistType = std::conditional_t<std::is_integral_v<ElementType>,
                                                std::uniform_int_distribution<ElementType>,
                                                std::uniform_real_distribution<ElementType>>;
            DistType dist{static_cast<ElementType>(m_min), static_cast<ElementType>(m_max) };
            fill_with_dist(data, dist, m_engine);

        }

    private:
        RandomEngine m_engine;
        double m_min;
        double m_max;
    };
}

#endif //UNIFORM_FILLER_HPP
