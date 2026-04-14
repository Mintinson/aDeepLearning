//
// Created by asus on 2025/1/15.
//

#ifndef BIAS_LAYER_HPP
#define BIAS_LAYER_HPP

#include "../../data/matrix.hpp"
#include "../../policy/policy.hpp"
#include "../../utils/vartype_dict.hpp"
#include "../layer_helper.hpp"
#include "../layer_io.hpp"
#include "../policies/init_policy.hpp"
#include "../policies/input_policy.hpp"
#include "../policies/operand_policy.hpp"
#include "../policies/update_policy.hpp"

namespace metann {
// using AddLayerInput = VarTypeDict<struct AddLayerIn1, struct AddLayerIn2>;

template <typename PoliciesContainer>
    requires IsPolicyContainer_v<PoliciesContainer>
class BiasLayer {
    using CurLayerPolicies = PlainPolicy_t<PoliciesContainer, PolicyContainer<>>;

public:
    static constexpr bool isFeedbackOutput =
        details::PolicySelect_t<FeedbackPolicy, CurLayerPolicies>::isFeedbackOutput;
    static constexpr bool isUpdate = details::PolicySelect_t<FeedbackPolicy, CurLayerPolicies>::isUpdate;
    using InputType = LayerIO;
    using OutputType = LayerIO;
    using ElementType = typename details::PolicySelect_t<OperandPolicy, CurLayerPolicies>::ElementType;
    using DeviceType = typename details::PolicySelect_t<OperandPolicy, CurLayerPolicies>::Device;

    BiasLayer(std::string name, const std::size_t vecLen) : m_name(std::move(name)), m_rowNum(1), m_colNum(vecLen) {}

    BiasLayer(std::string name, const std::size_t vecLen, const std::size_t colNum)
        : m_name(std::move(name))
        , m_rowNum(vecLen)
        , m_colNum(colNum) {}

    template <typename Initializer, typename Buffer, typename InitPolicies = typename Initializer::PolicyCont>
    void init(Initializer& initializer, Buffer& loadBuffer, std::ostream* log = nullptr) {
        if (auto cit = loadBuffer.find(m_name); cit != loadBuffer.end()) {
            const Matrix<ElementType, DeviceType>& matrix = cit->second;
            if (matrix.rowNum() == m_rowNum && matrix.colNum() == m_colNum) {
                m_bias = matrix;
                if (log != nullptr) {
                    const std::string logInfo = "load from load buffer: " + m_name + "\n";
                    (*log) << logInfo;
                }
                return;
            }
            throw std::runtime_error("BiasLayer::init(): matrix size mismatch");
        } else if (initializer.isMatrixExist(m_name)) {
            m_bias = Matrix<ElementType, DeviceType>{m_rowNum, m_colNum};
            initializer.getMatrix(m_name, m_bias);
            loadBuffer[m_name] = m_bias;
            if (log) {
                const std::string logInfo = "Copy from initializer: " + m_name + '\n';
                (*log) << logInfo;
            }
        } else {
            m_bias = Matrix<ElementType, DeviceType>{m_rowNum, m_colNum};
            using CurInitializer = PickInitializer_t<InitPolicies, InitPolicy::BiasTypeCate>;
            if constexpr (!std::is_same_v<CurInitializer, void>) {
                std::size_t fanIO = m_rowNum * m_colNum;
                auto& curInit = initializer.template getFiller<CurInitializer>();
                curInit.fill(m_bias, fanIO, fanIO);
                loadBuffer[m_name] = m_bias;
                if (log) {
                    const std::string logInfo = "Random init from initializer: " + m_name + '\n';
                    (*log) << logInfo;
                }
            } else {
                throw std::runtime_error("BiasLayer::init(): matrix size mismatch");
            }
        }
    }

    template <typename Save>
    void saveWeights(Save& saver) const {
        auto cit = saver.find(m_name);
        if (cit != saver.end() && cit->second != m_bias) {
            throw std::runtime_error("BiasLayer::save(): same matrix exists");
        }
        saver[m_name] = m_bias;
    }

    template <typename InType>
    auto feedForward(const InType& input) {
        const auto& val = input.template get<LayerIO>();
        return LayerIO::create().template set<LayerIO>(val + m_bias);
    }

    template <typename GradType>
    auto feedBackward(const GradType& grad) {
        if constexpr (isUpdate) {
            const auto& tmp = grad.template get<LayerIO>();
            assert(tmp.rowNum() == m_rowNum && tmp.colNum() == m_colNum);
            m_grad.push(make_dynamic(tmp));
        }
        if constexpr (isFeedbackOutput) {
            return grad;
        } else {
            return LayerIO::create();
        }
    }

    void neutralInvariant() {
        if constexpr (isFeedbackOutput) {
            if (m_grad.empty()) {
                return;
            }
            throw std::runtime_error("neutralInvariant: Neural Invariant Fail!");
        }
    }

    template <typename GradCollector>
    void gradCollect(GradCollector& col) {
        if constexpr (isUpdate) {
            // LayerTraits::MatrixGradCollect(m_bias, m_grad, col);
            matrix_grad_collect(m_bias, m_grad, col);
        }
    }

private:
    const std::string m_name;
    std::size_t m_rowNum;
    std::size_t m_colNum;

    Matrix<ElementType, DeviceType> m_bias;
    using DataType = details::LayerInternalBuf_t<isUpdate,
                                                 details::PolicySelect_t<InputPolicy, CurLayerPolicies>::BatchModel,
                                                 ElementType,
                                                 DeviceType,
                                                 CategoryTags::Matrix,
                                                 CategoryTags::BatchMatrix>;
    DataType m_grad;
};
}  // namespace metann

#endif  // BIAS_LAYER_HPP
