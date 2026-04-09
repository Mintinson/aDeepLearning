//
// Created by asus on 2025/1/15.
//

#ifndef LAYER_HELPER_HPP
#define LAYER_HELPER_HPP
#include "../data/data_category.hpp"
#include "../data/data_device.hpp"
#include "../operators/binary_operators.hpp"
#include "../operators/unary_operators.hpp"
#include "../utils/vartype_dict.hpp"
#include "dynamic_data.hpp"
#include "layer_io.hpp"
#include <list>
#include <stack>


namespace metann {
namespace details {
    template <typename Element, DeviceConcept DeviceType, CategoryConcept CateType>
    struct LayerInternalBufType {
        using tmp2 = DynamicData<Element, DeviceType, CateType>;
        using type = std::stack<tmp2, std::list<tmp2>>;
    };

    template <typename Element, DeviceConcept DeviceType, CategoryConcept CateType>
    using LayerInternalBufType_t = typename LayerInternalBufType<Element, DeviceType, CateType>::type;

    template <bool activate, bool batchMode,
        typename Element, DeviceConcept DeviceType,
        CategoryConcept CateTypeSingle, CategoryConcept CateTypeBatch>
    struct LayerInternalBuf {
        using type = std::conditional_t<batchMode,
            LayerInternalBufType_t<Element, DeviceType,
                CateTypeBatch>,
            LayerInternalBufType_t<Element, DeviceType,
                CateTypeSingle>>;
    };

    template <bool batchMode,
        typename ElementType, typename DeviceType,
        typename CateTypeSingle, typename CateTypeBatch>
    struct LayerInternalBuf<false, batchMode, ElementType, DeviceType, CateTypeSingle, CateTypeBatch> {
        using type = NullParameter;
    };

    template <bool activate, bool batchMode,
        typename ElementType, typename DeviceType,
        typename CateTypeSingle, typename CateTypeBatch>
    using LayerInternalBuf_t = typename LayerInternalBuf<
        activate, batchMode, ElementType, DeviceType, CateTypeSingle, CateTypeBatch>::type;
} // namespace details

namespace details {
    template <bool isFeedback>
    struct FeedbackOut {
        template <typename ElementType, DeviceConcept DeviceType>
        using InternalType = LayerInternalBufType_t<ElementType, DeviceType,
            CategoryTags::Matrix>;

        template <typename T, typename DataType>
        static auto recordData(const T& p_in, DataType& data)
        {
            auto tmp = make_dynamic(p_in);
            data.push(tmp);
            return tmp;
        }

        template <typename Grad, typename DataType>
        static auto feedback(DataType& data, const Grad& grad)
        {
            if (data.empty()) {
                throw std::runtime_error("Cannot feed back in SigmoidLayer");
            }
            auto tmp = grad.template get<LayerIO>();
            auto& tmp2 = data.top();
            auto res = LayerIO::create().set<LayerIO>(sign(tmp2) * tmp);
            data.pop();
            return res;
        }
    };

    template <>
    struct FeedbackOut<false> {
        template <typename ElementType, typename DeviceType>
        using InternalType = NullParameter;

        template <typename T, typename DataType>
        static auto recordData(T&& val, DataType&)
        {
            return std::forward<T>(val);
        }

        template <typename TGrad, typename DataType>
        static auto feedback(const DataType&, const TGrad&)
        {
            return LayerIO::create();
        }
    };
} // namespace details
template <typename Weight, typename Grad, typename GradCollector>
void matrix_grad_collect(const Weight& weight,
    Grad& grad,
    GradCollector& col)
{
    while (!grad.empty()) {
        auto g = grad.top();
        grad.pop();
        col.collect(weight, g);
    }
}
} // namespace metann

#endif // LAYER_HELPER_HPP
