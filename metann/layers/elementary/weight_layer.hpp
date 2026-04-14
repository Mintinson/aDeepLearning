//
// Created by asus on 2025/1/15.
//

#ifndef WEIGHT_LAYER_HPP
#define WEIGHT_LAYER_HPP

#include "../../data/matrix.hpp"
#include "../../operators/binary_operators.hpp"
#include "../../policy/policy.hpp"
#include "../../utils/vartype_dict.hpp"
#include "../layer_helper.hpp"
#include "../layer_io.hpp"
#include "../policies/init_policy.hpp"
#include "../policies/input_policy.hpp"
#include "../policies/operand_policy.hpp"
#include "../policies/update_policy.hpp"

namespace metann {
namespace details {
template <typename Weight, typename In>
auto dot_eval_helper(const Weight& p_weight, const In& p_in) {
    return dot(p_in, p_weight);
}
}  // namespace details

template <typename PoliciesContainer>
    requires IsPolicyContainer_v<PoliciesContainer>
class WeightLayer {
    using CurLayerPolicies = PlainPolicy_t<PoliciesContainer, PolicyContainer<>>;

public:
    static constexpr bool isFeedbackOutput =
        details::PolicySelect_t<FeedbackPolicy, CurLayerPolicies>::isFeedbackOutput;
    static constexpr bool isUpdate = details::PolicySelect_t<FeedbackPolicy, CurLayerPolicies>::isUpdate;
    using InputType = LayerIO;
    using OutputType = LayerIO;
    using ElementType = typename details::PolicySelect_t<OperandPolicy, CurLayerPolicies>::ElementType;
    using DeviceType = typename details::PolicySelect_t<OperandPolicy, CurLayerPolicies>::Device;

    WeightLayer(std::string name, const std::size_t inLen, const std::size_t outLen)
        : m_name(std::move(name))
        , m_inputLen(inLen)
        , m_outputLen(outLen) {}

    template <typename Initializer, typename Buffer, typename InitPolicies = typename Initializer::PolicyCont>
    void init(Initializer& initializer, Buffer& loadBuffer, std::ostream* log = nullptr) {
        if (auto cit = loadBuffer.find(m_name); cit != loadBuffer.end()) {
            const Matrix<ElementType, DeviceType>& matrix = cit->second;
            if (matrix.rowNum() == m_inputLen && matrix.colNum() == m_outputLen) {
                m_weight = matrix;
                if (log != nullptr) {
                    const std::string logInfo = "load from load buffer: " + m_name + "\n";
                    (*log) << logInfo;
                }
                return;
            }
            throw std::runtime_error("WeightLayer::init(): matrix size mismatch");
        } else if (initializer.isMatrixExist(m_name)) {
            m_weight = Matrix<ElementType, DeviceType>{m_inputLen, m_outputLen};
            initializer.getMatrix(m_name, m_weight);
            loadBuffer[m_name] = m_weight;
            if (log) {
                const std::string logInfo = "Copy from initializer: " + m_name + '\n';
                (*log) << logInfo;
            }
        } else {
            m_weight = Matrix<ElementType, DeviceType>{m_inputLen, m_outputLen};
            using CurInitializer = PickInitializer_t<InitPolicies, InitPolicy::WeightTypeCate>;
            if constexpr (!std::is_same_v<CurInitializer, void>) {
                // std::size_t fanIO = m_inputLen * m_outputLen;
                auto& curInit = initializer.template getFiller<CurInitializer>();
                curInit.fill(m_weight, m_inputLen, m_outputLen);
                loadBuffer[m_name] = m_weight;
                if (log) {
                    const std::string logInfo = "Random init from initializer: " + m_name + '\n';
                    (*log) << logInfo;
                }
            } else {
                throw std::runtime_error("WeightLayer::init(): Cannot get initializer for InitPolicy::WeightTypeCate");
            }
        }
    }

    template <typename Save>
    void saveWeights(Save& saver) const {
        auto cit = saver.find(m_name);
        if (cit != saver.end() && cit->second != m_weight) {
            throw std::runtime_error("BiasLayer::save(): same matrix exists");
        }
        saver[m_name] = m_weight;
    }

    template <typename InType>
    auto feedForward(const InType& input) {
        const auto& val = input.template get<LayerIO>();
        using rawType = std::decay_t<decltype(val)>;
        static_assert(!std::is_same_v<rawType, details::NullParameter>, "parameter is invalid");

        if constexpr (isUpdate) {
            m_updateInfo.push(make_dynamic(val));
        }

        auto res = details::dot_eval_helper(m_weight, val);
        return LayerIO::create().template set<LayerIO>(std::move(res));
    }

    template <typename GradType>
    auto feedBackward(const GradType& grad) {
        if constexpr (isUpdate) {
            auto tmp = grad.template get<LayerIO>();
            if (m_updateInfo.empty()) {
                throw std::runtime_error("Cannot do FeedBackward for Weight Layer");
            }

            auto tw = transpose(m_updateInfo.top());
            m_updateInfo.pop();
            auto res = details::dot_eval_helper(tmp, tw);
            m_gradInfo.push(make_dynamic(res));
        }

        if constexpr (isFeedbackOutput) {
            auto tmp = grad.template get<LayerIO>();
            auto tw = transpose(m_weight);
            auto res = details::dot_eval_helper(tw, tmp);
            return LayerIO::create().template set<LayerIO>(std::move(res));
        } else {
            return LayerIO::create();
        }
    }

    void neutralInvariant() {
        if constexpr (isUpdate) {
            if ((!m_updateInfo.empty()) || (!m_gradInfo.empty())) {
                throw std::runtime_error("neutralInvariant: Neural Invariant Fail!");
            }
        }
    }

    template <typename GradCollector>
    void gradCollect(GradCollector& col) {
        if constexpr (isUpdate) {
            // LayerTraits::MatrixGradCollect(m_bias, m_grad, col);
            matrix_grad_collect(m_weight, m_gradInfo, col);
        }
    }

private:
    const std::string m_name;
    const std::size_t m_inputLen;
    const std::size_t m_outputLen;

    Matrix<ElementType, DeviceType> m_weight;
    using DataType = details::LayerInternalBuf_t<isUpdate,
                                                 details::PolicySelect_t<InputPolicy, CurLayerPolicies>::BatchModel,
                                                 ElementType,
                                                 DeviceType,
                                                 CategoryTags::Matrix,
                                                 CategoryTags::BatchMatrix>;
    DataType m_updateInfo;
    DataType m_gradInfo;
};
}  // namespace metann

#endif  // WEIGHT_LAYER_HPP
