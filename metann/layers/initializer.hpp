//
// Created by asus on 2025/1/13.
//

#ifndef INITIALIZER_HPP
#define INITIALIZER_HPP
#include <map>
#include <string>

#include "../data/data_copy.hpp"
#include "../data/data_device.hpp"
#include "../data/matrix.hpp"
#include "../policy/policy.hpp"
#include "../utils/type_traits.hpp"
#include "../utils/vartype_dict.hpp"
#include "policies/init_policy.hpp"

namespace metann {
template <typename Element, typename PolicyContainer, typename Fillers>
class ParamInitializer {
public:
    using PolicyCont = PolicyContainer;

    explicit ParamInitializer(Fillers&& filler) : m_fillers{std::forward<Fillers>(filler)} {}

    template <typename Tag, typename Val>
    auto setFiller(Val&& val) && {
        auto newFiller = std::move(m_fillers).template set<Tag, Val>(std::forward<Val>(val));
        using NewFillerType = std::remove_cvref_t<decltype(newFiller)>;
        return ParamInitializer<Element, PolicyCont, NewFillerType>{std::move(newFiller)};
    }

    template <typename Tag>
    auto& getFiller() {
        return m_fillers.template get<Tag>();
    }

    template <typename Elem, DeviceConcept Device>
    void setMatrix(const std::string& name, const Matrix<Elem, Device>& param) {
        if (m_params.find(name) != m_params.end()) {
            throw std::runtime_error("Duplicate parameter matrix: " + name);
        }
        if constexpr (std::is_same_v<Elem, Element> && std::is_same_v<Device, CPU>) {
            m_params.insert({name, param});
        } else {
            Matrix<Element, CPU> mat{param.rowNum(), param.colNum()};
            data_copy(param, mat);
            m_params.insert({name, std::move(mat)});
        }
    }

    template <typename Elem, DeviceConcept Device>
    void getMatrix(const std::string& name, Matrix<Elem, Device>& res) const {
        auto it = m_params.find(name);
        if (it == m_params.end()) {
            throw std::runtime_error("Duplicate parameter matrix: " + name);
        }
        const auto& oriMat = it->second;
        if ((oriMat.rowNum() != res.rowNum()) || (oriMat.colNum() != res.colNum())) {
            throw std::runtime_error("Duplicate parameter matrix: " + name);
        }
        data_copy(oriMat, res);
    }

    [[nodiscard]] bool isMatrixExist(const std::string& name) const {
        auto it = m_params.find(name);
        return it != m_params.end();
    }

private:
    Fillers m_fillers;
    std::map<std::string, Matrix<Element, CPU>> m_params;
};

namespace details {
template <typename Result, typename... Policies>
struct FilterTagFromPolicies {
    using type = Result;
};

template <typename... Passed, template <typename...> class Container, typename Cur, typename... Policies>
struct FilterTagFromPolicies<Container<Passed...>, InitializerIs<Cur>, Policies...> {
    using type = typename FilterTagFromPolicies<Container<Passed..., Cur>, Policies...>::type;
};

template <typename... Passed, template <typename...> class Container, typename Cur, typename... Policies>
struct FilterTagFromPolicies<Container<Passed...>, WeightInitializerIs<Cur>, Policies...> {
    using type = typename FilterTagFromPolicies<Container<Passed..., Cur>, Policies...>::type;
};

template <typename... Passed, template <typename...> class Container, typename Cur, typename... Policies>
struct FilterTagFromPolicies<Container<Passed...>, BiasInitializerIs<Cur>, Policies...> {
    using type = typename FilterTagFromPolicies<Container<Passed..., Cur>, Policies...>::type;
};

template <typename Result, typename... Policies>
using FilterTagFromPolicies_t = typename FilterTagFromPolicies<Result, Policies...>::type;
}  // namespace details

template <typename Element, InitPolicyConcept... Policies>
auto make_initializer() {
    // using DictType = TransferTo_t<UniqueFromContainer_t<details::FilterTagFromPolicies_t<
    //                                   std::tuple<>, Policies...>>, VarTypeDict>;
    // using DictType = UniqueFromContainer_t<details::FilterTagFromPolicies_t<VarTypeDict<>, Policies...>>;
    using DictType = Unique_t<VarTypeDict, InnerType_t<Policies>...>;
    // using
    using FillDictType = std::remove_cvref_t<decltype(DictType::create())>;
    return ParamInitializer<Element, PolicyContainer<Policies...>, FillDictType>(DictType::create());
    // using dictType =
}
}  // namespace metann

#endif  // INITIALIZER_HPP
