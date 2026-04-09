//
// Created by asus on 2025/2/10.
//

#ifndef VAR_SCALE_FILLER_HPP
#define VAR_SCALE_FILLER_HPP
#include "../../policy/policy.hpp"
#include "../policies/init_policy.hpp"
#include <random>

#include "fill_with_dist.hpp"

namespace metann
{
    template <typename PolicyCont = PolicyContainer<>>
    class VarScaleFiller
    {
        using RandomEngine = typename details::PolicySelect_t<InitPolicy, PolicyCont>::RandEngine;

    public:
        VarScaleFiller(double factor = 1, unsigned seed = std::random_device{}())
            : m_factor(factor), m_engine(seed)
        {
        }

        template <typename Data>
        void fill(Data& data, std::size_t fanIn, std::size_t fanOut)
        {
            using ScaleMode = typename details::PolicySelect_t<VarScaleFillerPolicy, PolicyCont>::ScaleMode;
            double fanFactor = 0;
            if constexpr (std::is_same_v<ScaleMode, VarScaleFillerPolicy::ScaleModeTypeCate::FanIn>)
            {
                fanFactor = fanIn;
            }
            else if constexpr (std::is_same_v<ScaleMode, VarScaleFillerPolicy::ScaleModeTypeCate::FanOut>)
            {
                fanFactor = fanOut;
            }
            else if constexpr (std::is_same_v<ScaleMode, VarScaleFillerPolicy::ScaleModeTypeCate::FanAvg>)
            {
                fanFactor = (fanIn + fanOut) / 2;
            }

            using DistType = typename details::PolicySelect_t<VarScaleFillerPolicy, PolicyCont>::Distribute;
            using ElementType = typename Data::ElementType;
            if constexpr (std::is_same_v<DistType, VarScaleFillerPolicy::DistributeTypeCate::Uniform>)
            {
                double limit = std::sqrt(3.0 * m_factor / fanFactor);
                std::uniform_real_distribution<ElementType> dist(-limit, limit);
                fill_with_dist(data, dist, m_engine);
            }
            else if constexpr (std::is_same_v<DistType, VarScaleFillerPolicy::DistributeTypeCate::Norm>)
            {
                double stddev = sqrt(m_factor / fanFactor);
                std::normal_distribution<ElementType> dist(0, m_factor);
                fill_with_dist(data, dist, m_engine);
            }
        }

    private:
        double m_factor;
        RandomEngine m_engine;
    };
    // Xavier Filler, use FanAvg Mode
    template <typename PolicyCont = PolicyContainer<>>
    class XavierFiller : public VarScaleFiller<ChangePolicy_t<VarScaleFanAvg, PolicyCont>>
    {
        using BaseType = VarScaleFiller<ChangePolicy_t<VarScaleFanAvg, PolicyCont>>;

    public:
        explicit XavierFiller(unsigned seed = std::random_device{}()): BaseType(1, seed)
        {
        }
    };
    // MSRA Filler, use Norm Dist and FanIn Mode
    template <typename PolicyCont = PolicyContainer<>>
    class MSRAFiller
        : public VarScaleFiller<ChangePolicy_t<VarScaleFanIn, ChangePolicy_t<NormVarScale, PolicyCont>>>
    {
        using BaseType = VarScaleFiller<ChangePolicy_t<VarScaleFanIn, ChangePolicy_t<NormVarScale, PolicyCont>>>;

    public:
        MSRAFiller(unsigned seed = std::random_device{}())
            : BaseType(2, seed)
        {
        }
    };
} // namespace metann

#endif //VAR_SCALE_FILLER_HPP
