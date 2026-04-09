//
// Created by asus on 2025/1/24.
//

#ifndef FACILITIES_HPP
#define FACILITIES_HPP

#include "../data/data_category.hpp"
#include "../data/data_device.hpp"
#include <cassert>
#include <concepts>
#include <list>
#include <memory>
#include <stdexcept>
#include <typeindex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace metann {
template <DeviceConcept Device>
class BaseEvalUnit {
public:
    using DeviceType = Device;
    virtual ~BaseEvalUnit() = default;
    virtual void eval() = 0;
};

template <typename T>
concept EvalUnitConcept = requires {
    typename T::DeviceType;
    requires std::derived_from<T, BaseEvalUnit<typename T::DeviceType>>;
};

template <DeviceConcept Device>
class BaseEvalGroup {
public:
    virtual ~BaseEvalGroup() = default;
    virtual std::shared_ptr<BaseEvalUnit<Device>> getEvalUnit() = 0;
    virtual void merge(BaseEvalUnit<Device>&) = 0;
    virtual void merge(BaseEvalUnit<Device>&&) = 0;
};

template <typename T>
concept EvalGroupConcept = requires {
    typename T::DeviceType;
    requires std::derived_from<T, BaseEvalGroup<typename T::DeviceType>>;
};

template <DeviceConcept Device>
using EvalCluster = std::unordered_map<std::type_index, std::shared_ptr<BaseEvalGroup<Device>>>;

template <typename Device>
class BaseEvalPool {
public:
    virtual ~BaseEvalPool() = default;
    virtual void process(std::shared_ptr<BaseEvalUnit<Device>>&) = 0;
    virtual void barrier() = 0;
};

template <DataConcept Data>
class EvalHandle {
    struct DataWithEvalInfo {
        Data m_data;
        bool m_eval = false;
    };

public:
    EvalHandle()
        : m_data(std::make_shared<DataWithEvalInfo>())
    {
    }

    bool isEvaluated() const noexcept
    {
        return m_data->m_eval;
    }

    Data& mutableData()
    {
        if (isEvaluated()) {
            throw std::runtime_error("Data is already evaluated.");
        }
        return m_data->m_data;
    }

    void setEval()
    {
        if (isEvaluated()) {
            throw std::runtime_error("Data is already evaluated.");
        }
        m_data->m_eval = true;
    }

    const Data& data() const
    {
        if (!isEvaluated()) {
            throw std::runtime_error("Data is not evaluated.");
        }
        return m_data->m_data;
    }

    const void* dataPtr() const
    {
        return m_data.get();
    }

    template <typename... Params>
    void allocate(Params&&... params) const
    {
        if (isEvaluated()) {
            throw std::runtime_error("Data is already evaluated.");
        }
        m_data->m_data = Data(std::forward<Params>(params)...);
    }

private:
    std::shared_ptr<DataWithEvalInfo> m_data;
};

template <DataConcept Data>
class ConstEvalHandle {
public:
    explicit ConstEvalHandle(Data data)
        : m_constData(std::move(data))
    {
    }

    const Data& data() const
    {
        return m_constData;
    }

    const void* dataPtr() const
    {
        return &m_constData;
    }

private:
    Data m_constData;
};

template <DataConcept Data>
class ConstEvalHandle<EvalHandle<Data>> {
public:
    explicit ConstEvalHandle(EvalHandle<Data> data)
        : m_constData(std::move(data))
    {
    }

    const Data& data() const
    {
        return m_constData.data();
    }

    const void* dataPtr() const
    {
        return m_constData.dataPtr();
    }

private:
    EvalHandle<Data> m_constData;
};

template <DataConcept Data>
auto make_const_eval_handle(const Data& data)
{
    return ConstEvalHandle<Data>(data);
}

namespace details {
    template <DataConcept Data>
    class DynamicHandleDataBase {
    public:
        virtual ~DynamicHandleDataBase() = default;
        virtual const Data& data() const = 0;
        virtual const void* dataPtr() const = 0;
    };

    template <typename TData>
    class DynamicHandleData;

    template <DataConcept Data>
    class DynamicHandleData<ConstEvalHandle<Data>> final
        : public DynamicHandleDataBase<Data> {
    public:
        explicit DynamicHandleData(ConstEvalHandle<Data> data)
            : DynamicHandleDataBase<Data>()
            , m_data(std::move(data))
        {
        }

        const Data& data() const override
        {
            return m_data.data();
        }

        const void* dataPtr() const override
        {
            return m_data.dataPtr();
        }

    private:
        ConstEvalHandle<Data> m_data;
    };

    template <DataConcept TData>
    class DynamicHandleData<ConstEvalHandle<EvalHandle<TData>>> final
        : public DynamicHandleDataBase<TData> {
    public:
        explicit DynamicHandleData(ConstEvalHandle<EvalHandle<TData>> data)
            : DynamicHandleDataBase<TData>()
            , m_data(std::move(data))
        {
        }

        const TData& data() const override
        {
            return m_data.data();
        }

        const void* dataPtr() const override
        {
            return m_data.dataPtr();
        }

    private:
        ConstEvalHandle<EvalHandle<TData>> m_data;
    };
}

template <DataConcept Data>
class DynamicConstEvalHandle {
    using BaseData = details::DynamicHandleDataBase<Data>;

public:
    template <typename TRealHandle>
    explicit DynamicConstEvalHandle(TRealHandle data)
        : m_data(std::make_shared<details::DynamicHandleData<TRealHandle>>(std::move(data)))
    {
        assert(m_data);
    }

    const Data& data() const
    {
        return m_data->data();
    }

    const void* dataPtr() const
    {
        return m_data->dataPtr();
    }

private:
    std::shared_ptr<BaseData> m_data;
};

template <typename Device>
class TrivialEvalPool;

template <>
class TrivialEvalPool<CPU> : public BaseEvalPool<CPU> {
public:
    static TrivialEvalPool& instance()
    {
        static TrivialEvalPool inst;
        return inst;
    }

    void process(std::shared_ptr<BaseEvalUnit<CPU>>& evalUnit) override
    {
        evalUnit->eval();
    }

    void barrier() override
    {
    }

private:
    TrivialEvalPool() = default;
};

template <DataConcept Data>
class EvalBuffer {
public:
    using DataType = Data;

    auto handle() const
    {
        return m_handle;
    }

    auto constHandle() const
    {
        return ConstEvalHandle<EvalHandle<Data>>(m_handle);
    }

    bool isEvaluated() const
    {
        return m_handle.isEvaluated();
    }

private:
    EvalHandle<Data> m_handle;
};

template <typename Device>
class EvalLayer {
public:
    [[nodiscard]] std::size_t size() const
    {
        return m_evalSeq.size();
    }

    [[nodiscard]] EvalCluster<Device>& operator[](std::size_t index)
    {
        return m_evalSeq[index];
    }

    bool empty() const
    {
        return m_evalSeq.empty();
    }

    void clear()
    {
        m_evalSeq.clear();
        m_operands.clear();
        m_outputs.clear();
    }

    template <typename EvalGroup, typename EvalUnit>
    void evalRegister(EvalUnit&& evalReq, const void* resPtr,
        const std::vector<const void*>& paramPtr)
    {
        if (!resPtr)
            return;
        if (m_outputs.contains(resPtr))
            return;
        std::size_t depth = 0;
        for (auto p : paramPtr) {
            if (auto it = m_outputs.find(p); it != m_outputs.end())
                depth = std::max(depth, it->second);
        }
        depth += 1;
        if (m_evalSeq.size() <= depth) {
            m_evalSeq.resize(depth + 1);
        }
        EvalCluster<Device>& ec = m_evalSeq[depth];

        const auto typeIndex = std::type_index(typeid(EvalGroup));
        auto it = ec.find(typeIndex);
        if (it == ec.end()) {
            it = ec.insert(std::make_pair(typeIndex, std::make_shared<EvalGroup>())).first;
        }
        it->second->merge(std::forward<EvalUnit>(evalReq));

        m_outputs.insert({ resPtr, depth });
        ;
    }

private:
    std::vector<EvalCluster<Device>> m_evalSeq;
    std::unordered_set<const void*> m_operands;
    std::unordered_map<const void*, std::size_t> m_outputs;
};

enum class EvalPoolEnum {
    Trivial
};

template <DeviceConcept Device>
class EvalPlan {
public:
    static void setEvalPool(EvalPoolEnum epType)
    {
        globalEvalPool() = epType;
    }

    template <EvalGroupConcept EvalGroup, EvalUnitConcept EvalUnit>
    static void registerFun(EvalUnit&& evalReq, const void* outputPtr,
        const std::vector<const void*>& paramPtr)
    {
        threadInst().template evalRegister<EvalGroup>(
            std::forward<EvalUnit>(evalReq),
            outputPtr,
            paramPtr);
    }

    static void eval()
    {
        EvalPlan& plan = threadInst();
        if ((threadEvalPool() != globalEvalPool()) || (!plan.m_evalPool)) {
            switch (globalEvalPool()) {
            case EvalPoolEnum::Trivial:
                plan.m_evalPool = &(TrivialEvalPool<Device>::instance());
                break;
            default:
                assert(false);
            }
            threadEvalPool() = globalEvalPool();
        }
        if (!plan.m_evalPool) {
            throw std::runtime_error("No Evaluation Pool is available");
        }
        plan.doLayerEval();
        // if (())
    }

private:
    EvalPlan()
        : m_evalPool(nullptr)
    {
        m_evalLayers.resize(1);
    }

    static EvalPoolEnum& globalEvalPool()
    {
        static EvalPoolEnum inst = EvalPoolEnum::Trivial;
        return inst;
    }

    static EvalPoolEnum& threadEvalPool()
    {
        static thread_local EvalPoolEnum inst = globalEvalPool();
        return inst;
    }

    static EvalPlan& threadInst()
    {
        static thread_local EvalPlan inst;
        return inst;
    }

    template <typename EvalGroup, typename EvalUnit>
    void evalRegister(EvalUnit&& evalReq, const void* outputPtr,
        const std::vector<const void*>& paramPtr)
    {
        auto& curLayer = m_evalLayers.back();
        curLayer.template evalRegister<EvalGroup>(
            std::forward<EvalUnit>(evalReq), outputPtr, paramPtr);
    }

    void doLayerEval()
    {
        EvalLayer<Device>& curLayer = m_evalLayers.back();
        if (curLayer.empty())
            return;
        m_evalLayers.push_back(EvalLayer<Device> {});
        const std::size_t seqLen = curLayer.size();
        for (size_t i = 0; i < seqLen; ++i) {
            EvalCluster<Device>& ec = curLayer[i];
            for (auto& eg : ec) {
                while (auto unit = eg.second->getEvalUnit()) {
                    m_evalPool->process(unit);
                }
            }
            m_evalPool->barrier();
            if (!m_evalLayers.back().empty()) {
                doLayerEval();
            }
        }
        m_evalLayers.pop_back();
        curLayer.clear();
    }

    std::list<EvalLayer<Device>> m_evalLayers;
    BaseEvalPool<Device>* m_evalPool;
};

template <typename TData>
auto evaluate(const TData& data)
{
    using DeviceType = typename TData::DeviceType;
    auto evalHandle = data.evalRegister();
    EvalPlan<DeviceType>::eval();
    return evalHandle.data();
}

namespace eval {
    template <typename EvalType>
    struct UnitWrapper;
}
}
#endif // FACILITIES_HPP
