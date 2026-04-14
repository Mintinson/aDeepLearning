//
// Created by asus on 2025/1/10.
//

#ifndef OPERATOR_HELPER_HPP
#define OPERATOR_HELPER_HPP
#include <tuple>

#include "../data/data_category.hpp"
#include "operator_category.hpp"

// #include
namespace metann {
template <typename OperTag, typename Oper1, typename... Operands>
struct OperElementType {
    using type = typename Oper1::ElementType;
};

// default return the first element type
template <typename OperTag, typename Oper1, typename... Operands>
using OperElementType_t = typename OperElementType<OperTag, Oper1, Operands...>::type;

template <typename OperTag, typename Oper1, typename... Operands>
struct OperDeviceType {
    using type = typename Oper1::DeviceType;
};

// default return the first element device type
template <typename OperTag, typename Oper1, typename... Operands>
using OperDeviceType_t = typename OperDeviceType<OperTag, Oper1, Operands...>::type;

template <typename CateContainer, typename...>
struct Data2Cate {
    using type = CateContainer;
};

template <typename... ProcessedType, typename CurType, typename... RemainType>
struct Data2Cate<std::tuple<ProcessedType...>, CurType, RemainType...> {
private:
    using type1 = DataCategory_t<CurType>;
    using type2 = std::tuple<ProcessedType..., type1>;

public:
    using type = typename Data2Cate<type2, RemainType...>::type;
};

/// return the DataCategory tuple
template <typename HeadType, typename... RemainType>
using Data2Cate_t = typename Data2Cate<std::tuple<>, HeadType, RemainType...>::type;

// template<typename OperTag, typename>
template <typename OperTag, typename HeadCate, typename... RemainCate>
// requires std::conjunction_v<std::is_same<HeadCate, RemainCate>...>
struct OperCategory {
    using type = HeadCate;
};

template <typename OperTag, typename... RemainCate>
using OperaCategory_t = typename OperCategory<OperTag, RemainCate...>::type;

template <typename OperTag, typename CateContainer>
struct CateInduce;
template <typename OperTag, typename CateContainer>
using CateInduce_t = typename CateInduce<OperTag, CateContainer>::type;

template <typename OperTag, typename... Cates>
struct CateInduce<OperTag, std::tuple<Cates...>> {
    using type = OperaCategory_t<OperTag, Cates...>;
};

// given an operator and a series of operands, return the category of the result type
template <typename OperTag, typename Head, typename... Remain>
using OperaCateCal_t = CateInduce_t<OperTag, Data2Cate_t<Head, Remain...>>;

// given an operator and a series of operands, return the shape of the result
template <OperTagConcept OperTag, CategoryConcept Cate>
class OperOrganizer;

template <OperTagConcept OperTag>
class OperOrganizer<OperTag, CategoryTags::Scalar> {
public:
    template <DataConcept Head, DataConcept... Remain>
    OperOrganizer(const Head& head, const Remain&...) {}
};

template <OperTagConcept OperTag>
class OperOrganizer<OperTag, CategoryTags::BatchScalar> {
    template <DataConcept Head, DataConcept... Remain>
    bool is_same_dim(const Head& head, const Remain&...) {
        return true;
    }

    template <DataConcept Head, DataConcept CurType, DataConcept... Remain>
    bool is_same_dim(const Head& head, const CurType& cur, const Remain&... rem) {
        return (head.batchNum() == cur.batchNum()) && (is_same_dim(cur, rem...));
    }

public:
    // template <typename Head, typename ... Remain>
    template <DataConcept Head, DataConcept... Remain>
    OperOrganizer(const Head& head, const Remain&... rem) : m_batchNum(head.batchNum()) {
        assert(is_same_dim(head, rem...));
    }

    std::size_t batchNum() const { return m_batchNum; }

private:
    std::size_t m_batchNum{};
};

template <OperTagConcept OperTag>
class OperOrganizer<OperTag, CategoryTags::Matrix> {
    template <DataConcept Head, DataConcept... Remain>
    bool is_same_dim(const Head& head, const Remain&...) {
        return true;
    }

    template <DataConcept Head, DataConcept CurType, DataConcept... Remain>
    bool is_same_dim(const Head& head, const CurType& cur, const Remain&... rem) {
        return (head.rowNum() == cur.rowNum() && head.colNum() == cur.colNum()) && (is_same_dim(cur, rem...));
    }

public:
    // template <typename Head, typename ... Remain>
    template <DataConcept Head, DataConcept... Remain>
    OperOrganizer(const Head& head, const Remain&... rem) : m_rowNum(head.rowNum())
                                                          , m_colNum(head.colNum()) {
        assert(is_same_dim(head, rem...));
    }

    [[nodiscard]] std::size_t rowNum() const { return m_rowNum; }

    [[nodiscard]] std::size_t colNum() const { return m_colNum; }

private:
    std::size_t m_rowNum{};
    std::size_t m_colNum{};
};

template <OperTagConcept OperTag>
class OperOrganizer<OperTag, CategoryTags::BatchMatrix> {
    template <DataConcept Head, DataConcept... Remain>
    bool is_same_dim(const Head& head, const Remain&...) {
        return true;
    }

    template <DataConcept Head, DataConcept CurType, DataConcept... Remain>
    bool is_same_dim(const Head& head, const CurType& cur, const Remain&... rem) {
        return (head.rowNum() == cur.rowNum() && head.colNum() == cur.colNum() && head.batchNum() == cur.batchNum()) &&
               (is_same_dim(cur, rem...));
    }

public:
    // template <typename Head, typename ... Remain>
    template <DataConcept Head, DataConcept... Remain>
    OperOrganizer(const Head& head, const Remain&... rem)
        : m_rowNum(head.rowNum())
        , m_colNum(head.colNum())
        , m_batchNum(head.batchNum()) {
        assert(is_same_dim(head, rem...));
    }

    [[nodiscard]] std::size_t rowNum() const { return m_rowNum; }

    [[nodiscard]] std::size_t colNum() const { return m_colNum; }

    [[nodiscard]] std::size_t batchNum() const { return m_batchNum; }

private:
    std::size_t m_rowNum{};
    std::size_t m_colNum{};
    std::size_t m_batchNum{};
};

template <OperTagConcept OperTag>
struct OperSeq;
template <typename... Elems>
struct OperSeqContainer;

}  // namespace metann

#endif  // OPERATOR_HELPER_HPP
