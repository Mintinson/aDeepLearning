//
// Created by asus on 2025/2/12.
//

#ifndef GRAD_COLLECTOR_HPP
#define GRAD_COLLECTOR_HPP

#include "../data/array.hpp"
#include "../data/data_device.hpp"
#include "../data/matrix.hpp"
#include "dynamic_data.hpp"
#include <cstddef>
#include <unordered_map>
#include <utility>

namespace metann
{
    template <typename Element, DeviceConcept Device>
    struct MatrixGradInfo
    {
        using GradItemType = DynamicData<Element, Device, CategoryTags::Matrix>;

        MatrixGradInfo(Matrix<Element, Device> weight)
            : m_weight(std::move(weight))
              , m_grad(m_weight.rowNum(), m_weight.colNum())
        {
        }

        Matrix<Element, Device> m_weight;
        Array<GradItemType> m_grad;
    };

    template <typename Element, DeviceConcept Device>
    class GradCollectorIterator
    {
        using IteratorType = typename std::unordered_map<const Element*, MatrixGradInfo<Element, Device>>::iterator;

    public:
        GradCollectorIterator(IteratorType it)
            : m_it(std::move(it))
        {
        }

        const auto& operator*() const
        {
            return m_it->second;
        }

        const auto operator->() const
        {
            return &(m_it->second);
        }

        auto operator++()
        {
            ++m_it;
            return *this;
        }

        auto operator++(int)
        {
            auto tmp = *this;
            ++m_it;
            return tmp;
        }

        bool operator==(const GradCollectorIterator& git) const
        {
            return m_it == git.m_it;
        }

        bool operator!=(const GradCollectorIterator& git) const
        {
            return !(operator==(git));
        }

    private:
        IteratorType m_it;
    };

    template <typename Element, DeviceConcept Device>
    class GradCollector
    {
    public:
        GradCollector() = default;
        GradCollector(const GradCollector&) = delete;
        GradCollector(GradCollector&&) = default;

        GradCollector& operator=(const GradCollector&) = delete;
        GradCollector& operator=(GradCollector&&) = delete;

        template <typename Grad>
        void collect(const Matrix<Element, Device>& weight, const Grad& grad)
        {
            auto mem = lower_access(weight);
            auto buf = mem.rawMemory();

            if (auto it = m_matricesInfo.find(buf); it != m_matricesInfo.end())
            {
                if constexpr (IsMatrix_v<Grad>)
                {
                    it->second.m_grad.push_back(make_dynamic(grad));
                }
                else if (IsBatchMatrix_v<Grad>)
                {
                    it->second.m_grad.push_back(make_dynamic(collapse(grad)));
                }
                else
                {
                    static_assert(false);
                }
            }
            else
            {
                MatrixGradInfo<Element, Device> mgi{weight};

                if constexpr (IsMatrix_v<Grad>)
                {
                    mgi.m_grad.push_back(make_dynamic(grad));
                }
                else if (IsBatchMatrix_v<Grad>)
                {
                    mgi.m_grad.push_back(make_dynamic(collapse(grad)));
                }
                else
                {
                    static_assert(false);
                }
                m_matricesInfo.insert({buf, std::move(mgi)});
            }
        }

        void clear()
        {
            m_matricesInfo.clear();
        }

        [[nodiscard]] std::size_t size() const
        {
            return m_matricesInfo.size();
        }

        [[nodiscard]] auto begin()
        {
            return GradCollectorIterator<Element, Device>{m_matricesInfo.begin()};
        }

        [[nodiscard]] auto begin() const
        {
            return GradCollectorIterator<Element, Device>{m_matricesInfo.begin()};
        }

        auto end()
        {
            return GradCollectorIterator<Element, Device>{m_matricesInfo.end()};
        }

        auto end() const
        {
            return GradCollectorIterator<Element, Device>{m_matricesInfo.end()};
        }

    private:
        std::unordered_map<const Element*, MatrixGradInfo<Element, Device>> m_matricesInfo;
    };
}

#endif // GRAD_COLLECTOR_HPP
