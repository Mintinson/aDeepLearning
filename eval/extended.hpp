//
// Created by asus on 2025/1/24.
//

#ifndef EXTENDED_HPP
#define EXTENDED_HPP
#include <list>

#include "facilities.hpp"

namespace metann
{
    template <EvalUnitConcept EvalUnit>
    class TrivialEvalGroup : public BaseEvalGroup<typename EvalUnit::DeviceType>
    {
    public:
        using DeviceType = typename EvalUnit::DeviceType;

        std::shared_ptr<BaseEvalUnit<DeviceType>> getEvalUnit() override
        {
            std::shared_ptr<BaseEvalUnit<DeviceType>> evalUnit;
            if (m_unitList.empty())return evalUnit;
            evalUnit = std::make_shared<EvalUnit>(std::move(m_unitList.front()));
            m_unitList.pop_front();
            return evalUnit;
        }
        void merge(BaseEvalUnit<DeviceType>& unit) override
        {
            m_unitList.push_back(static_cast<EvalUnit&>(unit));
        }

        void merge(BaseEvalUnit<DeviceType>&& unit) override
        {
            m_unitList.push_back(static_cast<EvalUnit&&>(unit));
        }

    private:
        std::list<EvalUnit> m_unitList;
    };
}


#endif //EXTENDED_HPP
