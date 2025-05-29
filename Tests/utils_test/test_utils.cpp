
#include "acc_policy.h"
#include <format>
#include <iostream>
#include <vector>
#include <memory>
#include <metann/utils/vartype_dict.hpp>

constexpr char a[] = "Hello";
constexpr char b[] = "Hello";

class A;
class B;
class Weight;
class C;
class D;

using FParams = metann::VarTypeDict<A, B, Weight>;
using TupleParams = metann::VarTypeDictTuple<A, B, Weight>;

using MoreParams = metann::AddItem_t<FParams, C>;

template <typename T>
void foo(const T& in)
{
    auto a = in.template get<A>();
    auto b = in.template get<B>();
    auto weight = in.template get<Weight>();
    // std::cout << std::format("result is {}\n", (a * weight) + (b * (1.0 - weight)));
    std::cout << std::format("result is {}\n", (a * weight) + (b * (1.0 - weight)));
}

class FullClass
{
};

class NonClass;
int func(int);

template <typename T>
struct Oper
{
    using type = typename T::size_type;
};

int main()
{
    std::cout << std::format("{:<10}{:>10}\n", "Hello", "World");
    foo(FParams::create().set<A>(3.5).set<B>(2.4).set<Weight>(0.25));
    foo(TupleParams::create().set<A>(3.5).set<B>(2.4).set<Weight>(0.25));
    std::cout << typeid(MoreParams).name() << '\n';
    std::cout << typeid(metann::AddItem_t<MoreParams, D, NonClass>).name() << '\n';
    std::cout << typeid(metann::DelItem_t<MoreParams, C, A>).name() << '\n';
    std::array arr{ 1, 2, 3, 4, 5 };
    // decltype(func())
    // Accumulator<>::eval(arr);
    Accumulator<MulAccuPolicy, AveAccuPolicy>::eval(arr);
    std::cout << Accumulator<>::eval(arr) << std::endl;
    //
    std::cout << Accumulator<MulAccuPolicy>::eval(arr) << std::endl;
    std::cout << Accumulator<MulAccuPolicy, AveAccuPolicy>::eval(arr) << std::endl;
    std::cout << Accumulator<MulAccuPolicy, ValueAccuPolicy<double>>::eval(arr) << std::endl;
    // std::cout << typeid(metann::Unique_t<int, double, float, int, long, double>).name() << std::endl;
    std::cout << typeid(metann::UniqueFromContainer_t<std::tuple<int, double, float, int, long, double>>).name() <<
        std::endl;
    // static_assert(metann::isInContainer_v<int, std::tuple<float, double, int>>>, "error");
    std::cout << metann::isInContainer_v<int, std::tuple<float, double, int>> << std::endl;
    std::cout << typeid(metann::RemoveTypeFromContainer_t<
        int, std::tuple<double, int, float, int, long, double>>).name() << std::endl;
    std::cout << typeid(metann::Unique_t<std::tuple, int, double, float, int, long, double>).name() << std::endl;
    static_assert(metann::IsUnique_v<int, float, long, double>);
    std::cout << typeid(metann::Filter_t<true, std::is_integral, std::tuple, double, float, uint8_t, int, std::string>).
        name() << std::endl;
    std::cout << typeid(metann::TransformTo_t<std::tuple<std::vector<int>, std::vector<double>>, std::tuple, Oper>).
        name() << std::endl;
    std::cout << metann::Key2IDFromContainer_v<int, std::tuple<char, double, int>> << std::endl;
    std::cout << typeid(metann::ContainerTail_t<std::tuple<int, double, float, char>>).
        name() << std::endl;
}
