//
// Created by asus on 2025/1/20.
//

#include <iostream>

#include "../layers/grad_collector.hpp"
#include "../layers/initializer.hpp"
#include "../layers/layer_io.hpp"
#include "../layers/compose/compose_core.hpp"
#include "../layers/compose/linear_layer.hpp"
#include "../layers/compose/structure.hpp"
#include "../layers/elementary/add_layer.hpp"
#include "../layers/elementary/bias_layer.hpp"
#include "../layers/elementary/mul_layer.hpp"
#include "../layers/elementary/sigmoid_layer.hpp"
#include "../layers/elementary/tanh_layer.hpp"
#include "../layers/elementary/weight_layer.hpp"
#include "../layers/policies/single_layer_policy.hpp"
struct Tag1;
struct Tag2;
struct Tag3;
struct Tag4;
struct Tag5;
struct Tag6;
struct Input1;
struct Input2;
struct Input3;
struct Output1;
struct Output2;

using namespace metann;
using std::cout;
using std::endl;

void test_compose_kernel1()
{
    cout << "Test compose kernel case 1...\t";


    using check1 = SeparateParameters<SubLayer<Tag1, AddLayer>,
        SubLayer<Tag2, MulLayer>,
        SubLayer<Tag3, BiasLayer>,
        SubLayer<Tag4, TanhLayer>,
        SubLayer<Tag5, AddLayer>,
        InConnect<Input1, Tag1, AddLayerIn1>,
        InConnect<Input2, Tag1, AddLayerIn2>,
        InConnect<Input1, Tag2, MulLayerIn2>,
        InternalConnect<Tag1, LayerIO, Tag2, MulLayerIn1>,
        InternalConnect<Tag2, LayerIO, Tag3, LayerIO>,
        InternalConnect<Tag2, LayerIO, Tag4, LayerIO>,
        InternalConnect<Tag3, LayerIO, Tag5, AddLayerIn1>,
        InternalConnect<Tag4, LayerIO, Tag5, AddLayerIn2>,
        OutConnect<Tag5, LayerIO, Output1>>;
    static_assert(std::is_same_v<check1::SubLayerRes,
        SubLayerContainer<SubLayer<Tag1, AddLayer>,
        SubLayer<Tag2, MulLayer>,
        SubLayer<Tag3, BiasLayer>,
        SubLayer<Tag4, TanhLayer>,
        SubLayer<Tag5, AddLayer>>>, "Check Error");
    static_assert(std::is_same_v<check1::InterConnectRes,
        InterConnectContainer<InternalConnect<Tag1, LayerIO, Tag2, MulLayerIn1>,
        InternalConnect<Tag2, LayerIO, Tag3, LayerIO>,
        InternalConnect<Tag2, LayerIO, Tag4, LayerIO>,
        InternalConnect<Tag3, LayerIO, Tag5, AddLayerIn1>,
        InternalConnect<Tag4, LayerIO, Tag5, AddLayerIn2>>>,
        "Check Error");
    static_assert(std::is_same_v<check1::InConnectRes,
        InConnectContainer<InConnect<Input1, Tag1, AddLayerIn1>,
        InConnect<Input2, Tag1, AddLayerIn2>,
        InConnect<Input1, Tag2, MulLayerIn2>>>, "Check Error");

    static_assert(std::is_same_v<check1::OutConnectRes,
        OutConnectContainer<OutConnect<Tag5, LayerIO, Output1>>>, "Check Error");

    using check2 = SeparateParameters<SubLayer<Tag1, AddLayer>,
        InConnect<Input1, Tag1, AddLayerIn1>,
        InConnect<Input2, Tag1, AddLayerIn2>,
        SubLayer<Tag2, MulLayer>,
        InternalConnect<Tag1, LayerIO, Tag2, MulLayerIn1>,
        InternalConnect<Tag2, LayerIO, Tag3, LayerIO>,
        SubLayer<Tag3, BiasLayer>,
        InConnect<Input1, Tag2, MulLayerIn2>,
        InternalConnect<Tag2, LayerIO, Tag4, LayerIO>,
        SubLayer<Tag4, TanhLayer>,
        InternalConnect<Tag3, LayerIO, Tag5, AddLayerIn1>,
        OutConnect<Tag5, LayerIO, Output1>,
        SubLayer<Tag5, AddLayer>,
        InternalConnect<Tag4, LayerIO, Tag5, AddLayerIn2>>;

    static_assert(std::is_same_v<check2::SubLayerRes, check1::SubLayerRes>, "Check Error");
    static_assert(std::is_same_v<check2::InterConnectRes, check1::InterConnectRes>, "Check Error");
    static_assert(std::is_same_v<check2::InConnectRes, check1::InConnectRes>, "Check Error");
    static_assert(std::is_same_v<check2::OutConnectRes, check1::OutConnectRes>, "Check Error");
    cout << "done" << endl;
}

void test_compose_kernel2()
{
    cout << "Test compose kernel case 2...\t";

    static_assert(IsInPack_v<Tag1, Tag5, Tag4, Tag1, Tag3>, "Check Error");
    static_assert(!IsInPack_v<Tag1, Tag5, Tag4, Tag3>, "Check Error");

    static_assert(details::TagExistInLayerComps<Tag2, SubLayer<Tag1, AddLayer>,
        SubLayer<Tag2, MulLayer>,
        SubLayer<Tag3, BiasLayer>>::value, "Check Error");
    static_assert(!details::TagExistInLayerComps<Tag4, SubLayer<Tag1, AddLayer>,
        SubLayer<Tag2, MulLayer>,
        SubLayer<Tag3, BiasLayer>>::value, "Check Error");

    static_assert(details::TagExistInLayerComps<Tag2, InternalConnect<Tag2, LayerIO, Tag3, LayerIO>,
        InternalConnect<Tag2, LayerIO, Tag4, LayerIO>>::value, "Check Error");
    static_assert(details::TagExistInLayerComps<Tag3, InternalConnect<Tag2, LayerIO, Tag3, LayerIO>,
        InternalConnect<Tag2, LayerIO, Tag4, LayerIO>>::value, "Check Error");
    static_assert(details::TagExistInLayerComps<Tag4, InternalConnect<Tag2, LayerIO, Tag3, LayerIO>,
        InternalConnect<Tag2, LayerIO, Tag4, LayerIO>>::value, "Check Error");
    static_assert(!details::TagExistInLayerComps<Tag1, InternalConnect<Tag2, LayerIO, Tag3, LayerIO>,
        InternalConnect<Tag2, LayerIO, Tag4, LayerIO>>::value, "Check Error");

    static_assert(details::TagExistInLayerComps<Tag2, InConnect<Input1, Tag1, AddLayerIn1>,
        InConnect<Input2, Tag1, AddLayerIn2>,
        InConnect<Input1, Tag2, MulLayerIn2>>::value, "Check Error");
    static_assert(!details::TagExistInLayerComps<Tag3, InConnect<Input1, Tag1, AddLayerIn1>,
        InConnect<Input2, Tag1, AddLayerIn2>,
        InConnect<Input1, Tag2, MulLayerIn2>>::value, "Check Error");

    static_assert(details::TagExistInLayerComps<Tag5, OutConnect<Tag5, LayerIO, Output1>>::value, "Check Error");
    static_assert(!details::TagExistInLayerComps<Tag3, OutConnect<Tag5, LayerIO, Output1>>::value, "Check Error");
    cout << "done" << endl;
}

void test_compose_kernel3()
{
    cout << "Test compose kernel case 3...\t";
    static_assert(details::SubLayerChecker<SubLayerContainer<SubLayer<Tag1, AddLayer>,
        SubLayer<Tag2, MulLayer>,
        SubLayer<Tag3, BiasLayer>>>::IsUnique, "Check Error");
    static_assert(!details::SubLayerChecker<SubLayerContainer<SubLayer<Tag1, AddLayer>,
        SubLayer<Tag2, MulLayer>,
        SubLayer<Tag2, BiasLayer>>>::IsUnique, "Check Error");
    cout << "done" << endl;
}

void test_compose_kernel4()
{
    cout << "Test compose kernel case 4...\t";

    using check1 = details::InternalConnectChecker<InterConnectContainer<
        InternalConnect<Tag1, LayerIO, Tag2, MulLayerIn1>,
        InternalConnect<Tag2, LayerIO, Tag3, LayerIO>,
        InternalConnect<Tag2, LayerIO, Tag4, LayerIO>,
        InternalConnect<Tag3, LayerIO, Tag5, AddLayerIn1>,
        InternalConnect<Tag4, LayerIO, Tag5, AddLayerIn2>>>;
    static_assert(check1::NoSelfCycle, "Check Error");
    static_assert(check1::UniqueSource, "Check Error");

    using check2 = details::InternalConnectChecker<InterConnectContainer<
        InternalConnect<Tag1, LayerIO, Tag2, MulLayerIn1>,
        InternalConnect<Tag2, LayerIO, Tag3, LayerIO>,
        InternalConnect<Tag2, LayerIO, Tag2, LayerIO>,
        InternalConnect<Tag3, LayerIO, Tag5, AddLayerIn1>,
        InternalConnect<Tag4, LayerIO, Tag5, AddLayerIn2>>>;
    static_assert(!check2::NoSelfCycle, "Check Error");
    static_assert(check2::UniqueSource, "Check Error");

    using check3 = details::InternalConnectChecker<InterConnectContainer<
        InternalConnect<Tag1, LayerIO, Tag2, MulLayerIn1>,
        InternalConnect<Tag2, LayerIO, Tag3, LayerIO>,
        InternalConnect<Tag5, LayerIO, Tag3, LayerIO>,
        InternalConnect<Tag3, LayerIO, Tag5, AddLayerIn1>,
        InternalConnect<Tag4, LayerIO, Tag5, AddLayerIn2>>>;
    static_assert(check3::NoSelfCycle, "Check Error");
    static_assert(!check3::UniqueSource, "Check Error");
    cout << "done" << endl;
}

void test_compose_kernel5()
{
    cout << "Test compose kernel case 5...\t";

    using check1 = details::InputConnectChecker<InConnectContainer<InConnect<Input1, Tag1, AddLayerIn1>,
        InConnect<Input2, Tag1, AddLayerIn2>>>;
    static_assert(check1::UniqueSource, "Check Error");

    using check2 = details::InputConnectChecker<InConnectContainer<InConnect<Input1, Tag1, AddLayerIn1>,
        InConnect<Input2, Tag1, AddLayerIn1>>>;
    static_assert(!check2::UniqueSource, "Check Error");

    using check3 = details::InputConnectChecker<InConnectContainer<InConnect<Input1, Tag1, AddLayerIn1>,
        InConnect<Input1, Tag1, AddLayerIn2>>>;
    static_assert(check3::UniqueSource, "Check Error");
    cout << "done" << endl;
}

void test_compose_kernel6()
{
    cout << "Test compose kernel case 6...\t";

    using check1 = details::OutputConnectChecker<OutConnectContainer<OutConnect<Tag5, LayerIO, Output1>>>;
    static_assert(check1::UniqueSource, "Check Error");

    using check2 = details::OutputConnectChecker<OutConnectContainer<OutConnect<Tag5, LayerIO, Output1>,
        OutConnect<Tag5, LayerIO, Output2>>>;
    static_assert(check2::UniqueSource, "Check Error");

    using check3 = details::OutputConnectChecker<OutConnectContainer<OutConnect<Tag5, LayerIO, Output1>,
        OutConnect<Tag3, LayerIO, Output1>>>;
    static_assert(!check3::UniqueSource, "Check Error");
    cout << "done" << endl;
}

void test_compose_kernel7()
{
    cout << "Test compose kernel case 7...\t";

    using check1 = details::InternalTagInSublayer<
        InterConnectContainer<InternalConnect<Tag1, LayerIO, Tag2, MulLayerIn1>,
        InternalConnect<Tag2, LayerIO, Tag3, LayerIO>,
        InternalConnect<Tag5, LayerIO, Tag3, LayerIO>,
        InternalConnect<Tag3, LayerIO, Tag5, AddLayerIn1>,
        InternalConnect<Tag4, LayerIO, Tag5, AddLayerIn2>>,
        SubLayerContainer<SubLayer<Tag1, AddLayer>,
        SubLayer<Tag2, MulLayer>,
        SubLayer<Tag3, BiasLayer>,
        SubLayer<Tag4, TanhLayer>,
        SubLayer<Tag5, AddLayer>>>;
    static_assert(check1::value, "Check Error");

    using check2 = details::InternalTagInSublayer<
        InterConnectContainer<InternalConnect<Tag1, LayerIO, Tag2, MulLayerIn1>,
        InternalConnect<Tag2, LayerIO, Tag3, LayerIO>,
        InternalConnect<Tag5, LayerIO, Tag3, LayerIO>,
        InternalConnect<Tag3, LayerIO, Tag5, AddLayerIn1>,
        InternalConnect<Tag4, LayerIO, Tag5, AddLayerIn2>>,
        SubLayerContainer<SubLayer<Tag1, AddLayer>,
        SubLayer<Tag2, MulLayer>,
        SubLayer<Tag3, BiasLayer>,
        SubLayer<Tag4, TanhLayer>>>;
    static_assert(!check2::value, "Check Error");

    using check3 = details::InternalTagInSublayer<
        InterConnectContainer<InternalConnect<Tag1, LayerIO, Tag2, MulLayerIn1>,
        InternalConnect<Tag2, LayerIO, Tag3, LayerIO>,
        InternalConnect<Tag5, LayerIO, Tag3, LayerIO>>,
        SubLayerContainer<SubLayer<Tag1, AddLayer>,
        SubLayer<Tag2, MulLayer>,
        SubLayer<Tag3, BiasLayer>,
        SubLayer<Tag4, TanhLayer>,
        SubLayer<Tag5, AddLayer>>>;
    static_assert(check3::value, "Check Error");
    cout << "done" << endl;
}

void test_compose_kernel8()
{
    cout << "Test compose kernel case 8...\t";

    using check1 = details::InputTagInSubLayer<InConnectContainer<InConnect<Input1, Tag1, AddLayerIn1>,
        InConnect<Input2, Tag1, AddLayerIn2>,
        InConnect<Input1, Tag2, MulLayerIn2>>,
        SubLayerContainer<SubLayer<Tag1, AddLayer>,
        SubLayer<Tag2, MulLayer>,
        SubLayer<Tag3, BiasLayer>,
        SubLayer<Tag4, TanhLayer>,
        SubLayer<Tag5, AddLayer>>>;
    static_assert(check1::value, "Check Error");

    using check2 = details::InputTagInSubLayer<InConnectContainer<InConnect<Input1, Tag1, AddLayerIn1>,
        InConnect<Input2, Tag1, AddLayerIn2>,
        InConnect<Input1, Tag2, MulLayerIn2>>,
        SubLayerContainer<SubLayer<Tag1, AddLayer>,
        SubLayer<Tag3, BiasLayer>,
        SubLayer<Tag4, TanhLayer>,
        SubLayer<Tag5, AddLayer>>>;
    static_assert(!check2::value, "Check Error");
    cout << "done" << endl;
}

void test_compose_kernel9()
{
    cout << "Test compose kernel case 9...\t";

    using check1 = details::OutputTagInSubLayer<OutConnectContainer<OutConnect<Tag5, LayerIO, Output1>>,
        SubLayerContainer<SubLayer<Tag1, AddLayer>,
        SubLayer<Tag2, MulLayer>,
        SubLayer<Tag3, BiasLayer>,
        SubLayer<Tag4, TanhLayer>,
        SubLayer<Tag5, AddLayer>>>;
    static_assert(check1::value, "Check Error");

    using check2 = details::OutputTagInSubLayer<OutConnectContainer<OutConnect<Tag5, LayerIO, Output1>>,
        SubLayerContainer<SubLayer<Tag1, AddLayer>,
        SubLayer<Tag3, BiasLayer>,
        SubLayer<Tag4, TanhLayer>>>;
    static_assert(!check2::value, "Check Error");
    cout << "done" << endl;
}

void test_compose_kernel10()
{
    cout << "Test compose kernel case 10...\t";

    using check1 = details::SublayerTagInOtherArrays<
        InterConnectContainer<InternalConnect<Tag1, LayerIO, Tag2, MulLayerIn1>,
        InternalConnect<Tag2, LayerIO, Tag3, LayerIO>,
        InternalConnect<Tag2, LayerIO, Tag4, LayerIO>,
        InternalConnect<Tag3, LayerIO, Tag5, AddLayerIn1>,
        InternalConnect<Tag4, LayerIO, Tag5, AddLayerIn2>>,
        InConnectContainer<InConnect<Input1, Tag1, AddLayerIn1>,
        InConnect<Input2, Tag1, AddLayerIn2>,
        InConnect<Input1, Tag2, MulLayerIn2>>,
        OutConnectContainer<OutConnect<Tag5, LayerIO, Output1>>,
        SubLayerContainer<SubLayer<Tag1, AddLayer>,
        SubLayer<Tag2, MulLayer>,
        SubLayer<Tag3, BiasLayer>,
        SubLayer<Tag4, TanhLayer>,
        SubLayer<Tag5, AddLayer>>>;
    static_assert(check1::value, "Check Error");

    using check2 = details::SublayerTagInOtherArrays<
        InterConnectContainer<InternalConnect<Tag1, LayerIO, Tag2, MulLayerIn1>,
        InternalConnect<Tag2, LayerIO, Tag3, LayerIO>,
        InternalConnect<Tag2, LayerIO, Tag4, LayerIO>,
        InternalConnect<Tag3, LayerIO, Tag5, AddLayerIn1>,
        InternalConnect<Tag4, LayerIO, Tag5, AddLayerIn2>>,
        InConnectContainer<InConnect<Input1, Tag1, AddLayerIn1>,
        InConnect<Input2, Tag1, AddLayerIn2>,
        InConnect<Input1, Tag2, MulLayerIn2>>,
        OutConnectContainer<OutConnect<Tag5, LayerIO, Output1>>,
        SubLayerContainer<SubLayer<Tag1, AddLayer>,
        SubLayer<Tag2, MulLayer>,
        SubLayer<Tag3, BiasLayer>,
        SubLayer<Tag4, TanhLayer>,
        SubLayer<Tag5, AddLayer>,
        SubLayer<Tag6, AddLayer>>>;
    static_assert(!check2::value, "Check Error");
    cout << "done" << endl;
}

void test_compose_kernel11()
{
    cout << "Test compose kernel case 11...\t";

    using check1 = details::TagInInternalOut<Tag2, InternalConnect<Tag1, LayerIO, Tag2, MulLayerIn1>,
        InternalConnect<Tag2, LayerIO, Tag3, LayerIO>,
        InternalConnect<Tag2, LayerIO, Tag4, LayerIO>,
        InternalConnect<Tag3, LayerIO, Tag5, AddLayerIn1>,
        InternalConnect<Tag4, LayerIO, Tag5, AddLayerIn2>>;
    static_assert(check1::value, "Check Error");

    using check2 = details::TagInInternalOut<Tag5, InternalConnect<Tag1, LayerIO, Tag2, MulLayerIn1>,
        InternalConnect<Tag2, LayerIO, Tag3, LayerIO>,
        InternalConnect<Tag2, LayerIO, Tag4, LayerIO>,
        InternalConnect<Tag3, LayerIO, Tag5, AddLayerIn1>,
        InternalConnect<Tag4, LayerIO, Tag5, AddLayerIn2>>;
    static_assert(!check2::value, "Check Error");

    using check3 = details::TagInInternalIn<Tag2, InternalConnect<Tag1, LayerIO, Tag2, MulLayerIn1>,
        InternalConnect<Tag2, LayerIO, Tag3, LayerIO>,
        InternalConnect<Tag2, LayerIO, Tag4, LayerIO>,
        InternalConnect<Tag3, LayerIO, Tag5, AddLayerIn1>,
        InternalConnect<Tag4, LayerIO, Tag5, AddLayerIn2>>;
    static_assert(check3::value, "Check Error");

    using check4 = details::TagInInternalIn<Tag1, InternalConnect<Tag1, LayerIO, Tag2, MulLayerIn1>,
        InternalConnect<Tag2, LayerIO, Tag3, LayerIO>,
        InternalConnect<Tag2, LayerIO, Tag4, LayerIO>,
        InternalConnect<Tag3, LayerIO, Tag5, AddLayerIn1>,
        InternalConnect<Tag4, LayerIO, Tag5, AddLayerIn2>>;
    static_assert(!check4::value, "Check Error");
    cout << "done" << endl;
}

void test_compose_kernel12()
{
    cout << "Test compose kernel case 12...\t";

    using check1 = details::UsefulInternalPostLayer<
        InterConnectContainer<InternalConnect<Tag1, LayerIO, Tag2, MulLayerIn1>,
        InternalConnect<Tag2, LayerIO, Tag3, LayerIO>,
        InternalConnect<Tag2, LayerIO, Tag4, LayerIO>,
        InternalConnect<Tag3, LayerIO, Tag5, AddLayerIn1>,
        InternalConnect<Tag4, LayerIO, Tag5, AddLayerIn2>>,
        OutConnectContainer<OutConnect<Tag5, LayerIO, Output1>>>;
    static_assert(check1::value, "Check Error");

    // Error: Tag2 is useless
    using check2 = details::UsefulInternalPostLayer<InterConnectContainer<
        InternalConnect<Tag1, LayerIO, Tag2, MulLayerIn1>,
        InternalConnect<Tag3, LayerIO, Tag5, AddLayerIn1>,
        InternalConnect<Tag4, LayerIO, Tag5, AddLayerIn2>>,
        OutConnectContainer<OutConnect<Tag5, LayerIO, Output1>>>;
    static_assert(!check2::value, "Check Error");

    // Error: Tag5 is useless
    using check3 = details::UsefulInternalPostLayer<InterConnectContainer<
        InternalConnect<Tag1, LayerIO, Tag2, MulLayerIn1>,
        InternalConnect<Tag2, LayerIO, Tag3, LayerIO>,
        InternalConnect<Tag2, LayerIO, Tag4, LayerIO>,
        InternalConnect<Tag3, LayerIO, Tag5, AddLayerIn1>,
        InternalConnect<Tag4, LayerIO, Tag5, AddLayerIn2>>,
        OutConnectContainer<>>;
    static_assert(!check3::value, "Check Error");

    using check4 = details::UsefulInternalPostLayer<InterConnectContainer<
        InternalConnect<Tag1, LayerIO, Tag2, MulLayerIn1>,
        InternalConnect<Tag3, LayerIO, Tag5, AddLayerIn1>,
        InternalConnect<Tag4, LayerIO, Tag5, AddLayerIn2>>,
        OutConnectContainer<OutConnect<Tag5, LayerIO, Output1>,
        OutConnect<Tag2, LayerIO, Output2>>>;
    static_assert(check4::value, "Check Error");
    cout << "done" << endl;
}

void test_compose_kernel13()
{
    cout << "Test compose kernel case 13...\t";

    using check1 = details::UsefulInputLayer<InConnectContainer<InConnect<Input1, Tag1, AddLayerIn1>,
        InConnect<Input2, Tag1, AddLayerIn2>,
        InConnect<Input1, Tag2, MulLayerIn2>>,
        InterConnectContainer<InternalConnect<Tag1, LayerIO, Tag2, MulLayerIn1>,
        InternalConnect<Tag2, LayerIO, Tag3, LayerIO>,
        InternalConnect<Tag2, LayerIO, Tag4, LayerIO>,
        InternalConnect<Tag3, LayerIO, Tag5, AddLayerIn1>,
        InternalConnect<Tag4, LayerIO, Tag5, AddLayerIn2>>,
        OutConnectContainer<OutConnect<Tag5, LayerIO, Output1>>>;
    static_assert(check1::value, "Check Error");

    using check2 = details::UsefulInputLayer<InConnectContainer<InConnect<Input1, Tag1, AddLayerIn1>,
        InConnect<Input2, Tag1, AddLayerIn2>,
        InConnect<Input1, Tag5, MulLayerIn2>>,
        InterConnectContainer<InternalConnect<Tag1, LayerIO, Tag2, MulLayerIn1>,
        InternalConnect<Tag2, LayerIO, Tag3, LayerIO>,
        InternalConnect<Tag2, LayerIO, Tag4, LayerIO>,
        InternalConnect<Tag3, LayerIO, Tag5, AddLayerIn1>,
        InternalConnect<Tag4, LayerIO, Tag5, AddLayerIn2>>,
        OutConnectContainer<OutConnect<Tag5, LayerIO, Output1>>>;
    static_assert(check2::value, "Check Error");

    // Error: Tag1 is neither in InterConnect nor in OutConnect
    using check3 = details::UsefulInputLayer<InConnectContainer<InConnect<Input1, Tag1, AddLayerIn1>,
        InConnect<Input2, Tag1, AddLayerIn2>,
        InConnect<Input1, Tag5, MulLayerIn2>>,
        InterConnectContainer<InternalConnect<Tag2, LayerIO, Tag3, LayerIO>,
        InternalConnect<Tag2, LayerIO, Tag4, LayerIO>,
        InternalConnect<Tag3, LayerIO, Tag5, AddLayerIn1>,
        InternalConnect<Tag4, LayerIO, Tag5, AddLayerIn2>>,
        OutConnectContainer<OutConnect<Tag5, LayerIO, Output1>>>;
    static_assert(!check3::value, "Check Error");

    using check4 = details::UsefulInputLayer<InConnectContainer<InConnect<Input1, Tag1, AddLayerIn1>,
        InConnect<Input2, Tag1, AddLayerIn2>,
        InConnect<Input1, Tag5, MulLayerIn2>>,
        InterConnectContainer<InternalConnect<Tag1, LayerIO, Tag2, MulLayerIn1>,
        InternalConnect<Tag2, LayerIO, Tag3, LayerIO>,
        InternalConnect<Tag2, LayerIO, Tag4, LayerIO>,
        InternalConnect<Tag3, LayerIO, Tag5, AddLayerIn1>,
        InternalConnect<Tag4, LayerIO, Tag5, AddLayerIn2>>,
        OutConnectContainer<>>;
    static_assert(!check4::value, "Check Error");

    cout << "done" << endl;
}

void test_compose_kernel14()
{
    cout << "Test compose kernel case 14...\t";

    using InterConnects = InterConnectContainer<InternalConnect<Tag2, LayerIO, Tag3, LayerIO>,
        InternalConnect<Tag4, LayerIO, Tag5, AddLayerIn2>,
        InternalConnect<Tag1, LayerIO, Tag2, MulLayerIn1>,
        InternalConnect<Tag3, LayerIO, Tag5, AddLayerIn1>,
        InternalConnect<Tag2, LayerIO, Tag4, LayerIO>>;

    using check1 = TopologicalOrdering<SubLayerContainer<SubLayer<Tag3, BiasLayer>,
        SubLayer<Tag2, MulLayer>,
        SubLayer<Tag1, AddLayer>,
        SubLayer<Tag4, TanhLayer>,
        SubLayer<Tag6, AddLayer>,
        SubLayer<Tag5, AddLayer>>,
        InterConnects>::type;
    using comp1 = SubLayerContainer<SubLayer<Tag6, AddLayer>,
        SubLayer<Tag1, AddLayer>,
        SubLayer<Tag2, MulLayer>,
        SubLayer<Tag3, BiasLayer>,
        SubLayer<Tag4, TanhLayer>,
        SubLayer<Tag5, AddLayer>>;
    static_assert(std::is_same_v<check1, comp1>, "Check Error");

    using Policy1 = PolicyContainer<FeedbackOutputPolicy>;
    using Instantiation1 = details::SubLayerInstantiation<Policy1, check1, InterConnects>::type;
    static_assert(details::SubLayerInstantiation<Policy1, check1, InterConnects>::isPlainPolicyFeedbackOut, "yes");
    using InstantiationComp1 = std::tuple<
        details::InstantiatedSubLayer<Tag6, AddLayer<PolicyContainer<FeedbackOutputPolicy>>>,
        details::InstantiatedSubLayer<Tag1, AddLayer<PolicyContainer<FeedbackOutputPolicy>>>,
        details::InstantiatedSubLayer<Tag2, MulLayer<PolicyContainer<FeedbackOutputPolicy>>>,
        details::InstantiatedSubLayer<Tag3, BiasLayer<PolicyContainer<FeedbackOutputPolicy>>>,
        details::InstantiatedSubLayer<Tag4, TanhLayer<PolicyContainer<FeedbackOutputPolicy>>>,
        details::InstantiatedSubLayer<Tag5, AddLayer<PolicyContainer<FeedbackOutputPolicy>>>>;
    static_assert(std::is_same_v<Instantiation1, InstantiationComp1>, "Check Error");
    //
    using Policy2 = PolicyContainer<TanhAction, SubPolicyContainer<Tag3, BatchModelPolicy>>;
    using Instantiation2 = details::SubLayerInstantiation<Policy2, check1, InterConnects>::type;
    using InstantiationComp2 = std::tuple<details::InstantiatedSubLayer<Tag6, AddLayer<PolicyContainer<TanhAction>>>,
        details::InstantiatedSubLayer<Tag1, AddLayer<PolicyContainer<TanhAction>>>,
        details::InstantiatedSubLayer<Tag2, MulLayer<PolicyContainer<TanhAction>>>,
        details::InstantiatedSubLayer<
        Tag3, BiasLayer<PolicyContainer<BatchModelPolicy, TanhAction>>>,
        details::InstantiatedSubLayer<Tag4, TanhLayer<PolicyContainer<TanhAction>>>,
        details::InstantiatedSubLayer<Tag5, AddLayer<PolicyContainer<TanhAction>>>>;
    static_assert(std::is_same_v<Instantiation2, InstantiationComp2>, "Check Error");
    //
    using Policy3 = PolicyContainer<TanhAction, SubPolicyContainer<Tag2, UpdatePolicy>>;
    using Instantiation3 = details::SubLayerInstantiation<Policy3, check1, InterConnects>::type;
    using InstantiationComp3 = std::tuple<details::InstantiatedSubLayer<Tag6, AddLayer<PolicyContainer<TanhAction>>>,
        details::InstantiatedSubLayer<Tag1, AddLayer<PolicyContainer<TanhAction>>>,
        details::InstantiatedSubLayer<
        Tag2, MulLayer<PolicyContainer<UpdatePolicy, TanhAction>>>,
        details::InstantiatedSubLayer<
        Tag3, BiasLayer<PolicyContainer<TanhAction, FeedbackOutputPolicy>>>,
        details::InstantiatedSubLayer<
        Tag4, TanhLayer<PolicyContainer<TanhAction, FeedbackOutputPolicy>>>,
        details::InstantiatedSubLayer<
        Tag5, AddLayer<PolicyContainer<TanhAction, FeedbackOutputPolicy>>>>;
    static_assert(std::is_same_v<Instantiation3, InstantiationComp3>, "Check Error");

    using Policy4 = PolicyContainer<TanhAction, SubPolicyContainer<Tag3, UpdatePolicy>>;
    using Instantiation4 = details::SubLayerInstantiation<Policy4, check1, InterConnects>::type;
    using InstantiationComp4 = std::tuple<details::InstantiatedSubLayer<Tag6, AddLayer<PolicyContainer<TanhAction>>>,
        details::InstantiatedSubLayer<Tag1, AddLayer<PolicyContainer<TanhAction>>>,
        details::InstantiatedSubLayer<Tag2, MulLayer<PolicyContainer<TanhAction>>>,
        details::InstantiatedSubLayer<
        Tag3, BiasLayer<PolicyContainer<UpdatePolicy, TanhAction>>>,
        details::InstantiatedSubLayer<Tag4, TanhLayer<PolicyContainer<TanhAction>>>,
        details::InstantiatedSubLayer<
        Tag5, AddLayer<PolicyContainer<TanhAction, FeedbackOutputPolicy>>>>;
    static_assert(std::is_same_v<Instantiation4, InstantiationComp4>, "Check Error");
    cout << "done" << endl;
}
void test_linear_layer1()
{
    cout << "Test linear layer case 1 ...\t";
    using RootLayer = InjectPolicy_t<LinearLayer>;
    static_assert(!RootLayer::isUpdate, "Test Error");
    static_assert(!RootLayer::isFeedbackOutput, "Test Error");

    RootLayer layer("root", 2, 3);

    Matrix<float, CPU> w1(2, 3);
    w1.setValue(0, 0, 0.1f);  w1.setValue(1, 0, 0.2f);
    w1.setValue(0, 1, 0.3f);  w1.setValue(1, 1, 0.4f);
    w1.setValue(0, 2, 0.5f);  w1.setValue(1, 2, 0.6f);

    Matrix<float, CPU> b1(1, 3);
    b1.setValue(0, 0, 0.7f);  b1.setValue(0, 1, 0.8f); b1.setValue(0, 2, 0.9f);

    auto initializer = make_initializer<float>();
    initializer.setMatrix("root-weight", w1);
    initializer.setMatrix("root-bias", b1);
    std::map<std::string, Matrix<float, CPU>> params;
    layer.init(initializer, params);
    //
    Matrix<float, CPU> i(1, 2);
    i.setValue(0, 0, 0.1f); i.setValue(0, 1, 0.2f);
    //
    auto input = LayerIO::create().set<LayerIO>(i);
    auto out = evaluate(layer.feedForward(input).get<LayerIO>());
    //
    assert(fabs(out(0, 0) - 0.75f) < 0.00001);
    assert(fabs(out(0, 1) - 0.91f) < 0.00001);
    assert(fabs(out(0, 2) - 1.07f) < 0.00001);
    //
    auto out_grad = layer.feedBackward(LayerIO::create());
    auto fbOut = out_grad.get<LayerIO>();
    static_assert(std::is_same_v<decltype(fbOut), details::NullParameter>, "Test error");
    //
    params.clear();
    layer.saveWeights(params);
    assert(params.find("root-weight") != params.end());
    assert(params.find("root-bias") != params.end());

    cout << "done" << endl;
}
void test_linear_layer2()
{
    cout << "Test linear layer case 2 ...\t";
    using RootLayer = InjectPolicy_t<LinearLayer, UpdatePolicy>;
    static_assert(RootLayer::isUpdate, "Test Error");
    static_assert(!RootLayer::isFeedbackOutput, "Test Error");

    RootLayer layer("root", 2, 3);

    Matrix<float, CPU> w1(2, 3);
    w1.setValue(0, 0, 0.1f);  w1.setValue(1, 0, 0.2f);
    w1.setValue(0, 1, 0.3f);  w1.setValue(1, 1, 0.4f);
    w1.setValue(0, 2, 0.5f);  w1.setValue(1, 2, 0.6f);

    Matrix<float, CPU> b1(1, 3);
    b1.setValue(0, 0, 0.7f);  b1.setValue(0, 1, 0.8f); b1.setValue(0, 2, 0.9f);

    auto initializer = make_initializer<float>();
    initializer.setMatrix("root-weight", w1);
    initializer.setMatrix("root-bias", b1);
    std::map<std::string, Matrix<float, CPU>> params;
    layer.init(initializer, params);

    Matrix<float, CPU> i(1, 2);
    i.setValue(0, 0, 0.1f); i.setValue(0, 1, 0.2f);

    auto input = LayerIO::create().set<LayerIO>(i);
    auto out = evaluate(layer.feedForward(input).get<LayerIO>());
    assert(fabs(out(0, 0) - 0.75f) < 0.00001);
    assert(fabs(out(0, 1) - 0.91f) < 0.00001);
    assert(fabs(out(0, 2) - 1.07f) < 0.00001);

    Matrix<float, CPU> g(1, 3);
    g.setValue(0, 0, 0.1f);
    g.setValue(0, 1, 0.2f);
    g.setValue(0, 2, 0.3f);
    auto fbIn = LayerIO::create().set<LayerIO>(g);
    auto out_grad = layer.feedBackward(fbIn);
    auto fbOut = out_grad.get<LayerIO>();
    static_assert(std::is_same_v<decltype(fbOut), details::NullParameter>, "Test error");

    GradCollector<float, CPU> grad_collector;
    layer.gradCollect(grad_collector);
    assert(grad_collector.size() == 2);

    bool weight_update_valid = false;
    bool bias_update_valid = false;

    for (auto& p : grad_collector)
    {
        auto w = p.m_weight;
        auto info = evaluate(collapse(p.m_grad));
        if (w.rowNum() == w1.rowNum())
        {
            weight_update_valid = true;

            auto tmp = evaluate(dot(transpose(i), g));
            assert(tmp.rowNum() == info.rowNum());
            assert(tmp.colNum() == info.colNum());

            for (size_t i = 0; i < tmp.rowNum(); ++i)
            {
                for (size_t j = 0; j < tmp.colNum(); ++j)
                {
                    assert(fabs(info(i, j) - tmp(i, j)) < 0.0001f);
                }
            }
        }
        else if (w.rowNum() == b1.rowNum())
        {
            bias_update_valid = true;
            for (size_t i = 0; i < info.rowNum(); ++i)
            {
                for (size_t j = 0; j < info.colNum(); ++j)
                {
                    assert(fabs(info(i, j) - g(i, j)) < 0.0001f);
                }
            }
        }
        else
        {
            assert(false);
        }
    }
    assert(bias_update_valid);
    assert(weight_update_valid);

    params.clear();
    layer.saveWeights(params);
    assert(params.find("root-weight") != params.end());
    assert(params.find("root-bias") != params.end());

    cout << "done" << endl;
}

void test_linear_layer3()
{
    cout << "Test linear layer case 3 ...\t";
    using RootLayer = InjectPolicy_t<LinearLayer, UpdatePolicy,
        SubPolicyContainer<SubLayerOf<LinearLayer>::Weight, NoUpdatePolicy>>;
    // using Cont = PolicyContainer<UpdatePolicy, SubPolicyContainer<SubLayerOf<LinearLayer>::Weight, NoUpdatePolicy>>;
    // using PlanP = PlainPolicy_t<Cont, PolicyContainer<>>;
    // // using KeyType = RootLayer::PlainPolicies;
    // static_assert(!details::PolicySelect_t<FeedbackPolicy, PlanP>::isFeedbackOutput);
    // static_assert(details::PolicySelect_t<FeedbackPolicy, PlanP>::isUpdate);
    static_assert(!RootLayer::isFeedbackOutput, "Test Error");
    static_assert(RootLayer::isUpdate, "Test Error");
    //
    RootLayer layer("root", 2, 3);
    //
    Matrix<float, CPU> w1(2, 3);
    w1.setValue(0, 0, 0.1f);
    w1.setValue(1, 0, 0.2f);
    w1.setValue(0, 1, 0.3f);
    w1.setValue(1, 1, 0.4f);
    w1.setValue(0, 2, 0.5f);
    w1.setValue(1, 2, 0.6f);
    //
    Matrix<float, CPU> b1(1, 3);
    b1.setValue(0, 0, 0.7f);
    b1.setValue(0, 1, 0.8f);
    b1.setValue(0, 2, 0.9f);
    //
    auto initializer = make_initializer<float>();
    initializer.setMatrix("root-weight", w1);
    initializer.setMatrix("root-bias", b1);
    std::map<std::string, Matrix<float, CPU>> params;
    layer.init(initializer, params);
    //
    Matrix<float, CPU> i(1, 2);
    i.setValue(0, 0, 0.1f);
    i.setValue(0, 1, 0.2f);
    //
    auto input = LayerIO::create().set<LayerIO>(i);
    auto out = evaluate(layer.feedForward(input).get<LayerIO>());
    //
    assert(fabs(out(0, 0) - 0.75f) < 0.00001);
    assert(fabs(out(0, 1) - 0.91f) < 0.00001);
    assert(fabs(out(0, 2) - 1.07f) < 0.00001);
    //
    Matrix<float, CPU> g(1, 3);
    g.setValue(0, 0, 0.1f);
    g.setValue(0, 1, 0.2f);
    g.setValue(0, 2, 0.3f);
    auto fbIn = LayerIO::create().set<LayerIO>(g);
    auto out_grad = layer.feedBackward(fbIn);
    auto fbOut = out_grad.get<LayerIO>();
    static_assert(std::is_same_v<decltype(fbOut), details::NullParameter>, "Test error");
    //
    GradCollector<float, CPU> grad_collector;
    layer.gradCollect(grad_collector);
    assert(grad_collector.size() == 1);
    //
    auto w = (*grad_collector.begin()).m_weight;
    auto info = evaluate(collapse((*grad_collector.begin()).m_grad));
    assert(w == params.begin()->second);
    for (size_t i = 0; i < info.rowNum(); ++i)
    {
        for (size_t j = 0; j < info.colNum(); ++j)
        {
            assert(fabs(info(i, j) - g(i, j)) < 0.0001f);
        }
    }

    params.clear();
    layer.saveWeights(params);
    assert(params.find("root-weight") != params.end());
    assert(params.find("root-bias") != params.end());
    cout << "done" << endl;
}

void test_linear_layer4()
{
    cout << "Test linear layer case 4 ...\t";
    using RootLayer = InjectPolicy_t<LinearLayer, UpdatePolicy,
        SubPolicyContainer<SubLayerOf<LinearLayer>::Bias, NoUpdatePolicy>>;
    static_assert(RootLayer::isUpdate, "Test Error");
    static_assert(!RootLayer::isFeedbackOutput, "Test Error");

    RootLayer layer("root", 2, 3);
    std::map<std::string, Matrix<float, CPU>> params;

    Matrix<float, CPU> w1(2, 3);
    w1.setValue(0, 0, 0.1f);
    w1.setValue(1, 0, 0.2f);
    w1.setValue(0, 1, 0.3f);
    w1.setValue(1, 1, 0.4f);
    w1.setValue(0, 2, 0.5f);
    w1.setValue(1, 2, 0.6f);

    Matrix<float, CPU> b1(1, 3);
    b1.setValue(0, 0, 0.7f);
    b1.setValue(0, 1, 0.8f);
    b1.setValue(0, 2, 0.9f);

    auto initializer = make_initializer<float>();
    initializer.setMatrix("root-weight", w1);
    initializer.setMatrix("root-bias", b1);
    layer.init(initializer, params);

    Matrix<float, CPU> i(1, 2);
    i.setValue(0, 0, 0.1f);
    i.setValue(0, 1, 0.2f);

    auto input = LayerIO::create().set<LayerIO>(i);
    auto out = evaluate(layer.feedForward(input).get<LayerIO>());

    assert(fabs(out(0, 0) - 0.75f) < 0.00001);
    assert(fabs(out(0, 1) - 0.91f) < 0.00001);
    assert(fabs(out(0, 2) - 1.07f) < 0.00001);

    Matrix<float, CPU> g(1, 3);
    g.setValue(0, 0, 0.1f);
    g.setValue(0, 1, 0.2f);
    g.setValue(0, 2, 0.3f);
    auto fbIn = LayerIO::create().set<LayerIO>(g);
    auto out_grad = layer.feedBackward(fbIn);
    auto fbOut = out_grad.get<LayerIO>();
    static_assert(std::is_same_v<decltype(fbOut), details::NullParameter>, "Test error");

    GradCollector<float, CPU> grad_collector;
    layer.gradCollect(grad_collector);
    assert(grad_collector.size() == 1);

    auto w = (*grad_collector.begin()).m_weight;
    assert(w == params.rbegin()->second);

    auto check = collapse(grad_collector.begin()->m_grad);
    auto check2 = dot(transpose(i), g);

    auto handle1 = check.evalRegister();
    auto handle2 = check2.evalRegister();
    EvalPlan<CPU>::eval();

    auto info = handle1.data();
    auto tmp = handle2.data();
    assert(tmp.rowNum() == info.rowNum());
    assert(tmp.colNum() == info.colNum());

    for (size_t i = 0; i < tmp.rowNum(); ++i)
    {
        for (size_t j = 0; j < tmp.colNum(); ++j)
        {
            assert(fabs(info(i, j) - tmp(i, j)) < 0.0001f);
        }
    }

    params.clear();
    layer.saveWeights(params);
    assert(params.find("root-weight") != params.end());
    assert(params.find("root-bias") != params.end());
    cout << "done" << endl;
}

void test_linear_layer5()
{
    cout << "Test linear layer case 5 ...\t";
    using RootLayer = InjectPolicy_t<LinearLayer,
        SubPolicyContainer<SubLayerOf<LinearLayer>::Bias, UpdatePolicy>>;
    static_assert(RootLayer::isUpdate, "Test Error");
    static_assert(!RootLayer::isFeedbackOutput, "Test Error");

    RootLayer layer("root", 2, 3);
    std::map<std::string, Matrix<float, CPU>> params;

    Matrix<float, CPU> w1(2, 3);
    w1.setValue(0, 0, 0.1f);
    w1.setValue(1, 0, 0.2f);
    w1.setValue(0, 1, 0.3f);
    w1.setValue(1, 1, 0.4f);
    w1.setValue(0, 2, 0.5f);
    w1.setValue(1, 2, 0.6f);

    Matrix<float, CPU> b1(1, 3);
    b1.setValue(0, 0, 0.7f);
    b1.setValue(0, 1, 0.8f);
    b1.setValue(0, 2, 0.9f);

    auto initializer = make_initializer<float>();
    initializer.setMatrix("root-weight", w1);
    initializer.setMatrix("root-bias", b1);
    layer.init(initializer, params);

    Matrix<float, CPU> i(1, 2);
    i.setValue(0, 0, 0.1f);
    i.setValue(0, 1, 0.2f);

    auto input = LayerIO::create().set<LayerIO>(i);
    auto out = evaluate(layer.feedForward(input).get<LayerIO>());

    assert(fabs(out(0, 0) - 0.75f) < 0.00001);
    assert(fabs(out(0, 1) - 0.91f) < 0.00001);
    assert(fabs(out(0, 2) - 1.07f) < 0.00001);

    Matrix<float, CPU> g(1, 3);
    g.setValue(0, 0, 0.1f);
    g.setValue(0, 1, 0.2f);
    g.setValue(0, 2, 0.3f);
    auto fbIn = LayerIO::create().set<LayerIO>(g);
    auto out_grad = layer.feedBackward(fbIn);
    auto fbOut = out_grad.get<LayerIO>();
    static_assert(std::is_same_v<decltype(fbOut), details::NullParameter>, "Test error");

    GradCollector<float, CPU> grad_collector;
    layer.gradCollect(grad_collector);
    assert(grad_collector.size() == 1);

    auto w = (*grad_collector.begin()).m_weight;
    auto info = evaluate(collapse(grad_collector.begin()->m_grad));
    assert(w == params.begin()->second);
    for (size_t i = 0; i < info.rowNum(); ++i)
    {
        for (size_t j = 0; j < info.colNum(); ++j)
        {
            assert(fabs(info(i, j) - g(i, j)) < 0.0001f);
        }
    }

    params.clear();
    layer.saveWeights(params);
    assert(params.find("root-weight") != params.end());
    assert(params.find("root-bias") != params.end());
    cout << "done" << endl;
}

void test_linear_layer6()
{
    cout << "Test linear layer case 6 ...\t";
    using RootLayer = InjectPolicy_t<LinearLayer,
        SubPolicyContainer<SubLayerOf<LinearLayer>::Weight, UpdatePolicy>>;
    static_assert(RootLayer::isUpdate, "Test Error");
    static_assert(!RootLayer::isFeedbackOutput, "Test Error");

    RootLayer layer("root", 2, 3);
    std::map<std::string, Matrix<float, CPU>> params;

    Matrix<float, CPU> w1(2, 3);
    w1.setValue(0, 0, 0.1f);
    w1.setValue(1, 0, 0.2f);
    w1.setValue(0, 1, 0.3f);
    w1.setValue(1, 1, 0.4f);
    w1.setValue(0, 2, 0.5f);
    w1.setValue(1, 2, 0.6f);

    Matrix<float, CPU> b1(1, 3);
    b1.setValue(0, 0, 0.7f);
    b1.setValue(0, 1, 0.8f);
    b1.setValue(0, 2, 0.9f);

    auto initializer = make_initializer<float>();
    initializer.setMatrix("root-weight", w1);
    initializer.setMatrix("root-bias", b1);
    layer.init(initializer, params);

    Matrix<float, CPU> i(1, 2);
    i.setValue(0, 0, 0.1f);
    i.setValue(0, 1, 0.2f);

    auto input = LayerIO::create().set<LayerIO>(i);
    auto out = evaluate(layer.feedForward(input).get<LayerIO>());

    assert(fabs(out(0, 0) - 0.75f) < 0.00001);
    assert(fabs(out(0, 1) - 0.91f) < 0.00001);
    assert(fabs(out(0, 2) - 1.07f) < 0.00001);

    Matrix<float, CPU> g(1, 3);
    g.setValue(0, 0, 0.1f);
    g.setValue(0, 1, 0.2f);
    g.setValue(0, 2, 0.3f);
    auto fbIn = LayerIO::create().set<LayerIO>(g);
    auto out_grad = layer.feedBackward(fbIn);
    auto fbOut = out_grad.get<LayerIO>();
    static_assert(std::is_same_v<decltype(fbOut), details::NullParameter>, "Test error");

    GradCollector<float, CPU> grad_collector;
    layer.gradCollect(grad_collector);
    assert(grad_collector.size() == 1);

    auto w = (*grad_collector.begin()).m_weight;
    assert(w == params.rbegin()->second);
    auto check1 = collapse((*grad_collector.begin()).m_grad);
    auto check2 = dot(transpose(i), g);

    auto handle1 = check1.evalRegister();
    auto handle2 = check2.evalRegister();
    EvalPlan<CPU>::eval();

    auto info = handle1.data();
    auto tmp = handle2.data();

    assert(tmp.rowNum() == info.rowNum());
    assert(tmp.colNum() == info.colNum());

    for (size_t i = 0; i < tmp.rowNum(); ++i)
    {
        for (size_t j = 0; j < tmp.colNum(); ++j)
        {
            assert(fabs(info(i, j) - tmp(i, j)) < 0.0001f);
        }
    }

    params.clear();
    layer.saveWeights(params);
    assert(params.find("root-weight") != params.end());
    assert(params.find("root-bias") != params.end());
    cout << "done" << endl;
}

int main()
{
    std::cout << "Hello World!\n";
    // std::cout << typeid(Topology).name() << '\n';
    test_compose_kernel1();
    test_compose_kernel2();
    test_compose_kernel3();
    test_compose_kernel4();
    test_compose_kernel5();
    test_compose_kernel6();
    test_compose_kernel7();
    test_compose_kernel8();
    test_compose_kernel9();
    test_compose_kernel10();
    test_compose_kernel11();
    test_compose_kernel12();
    test_compose_kernel13();
    test_compose_kernel14();

    test_linear_layer1();
    test_linear_layer2();
    test_linear_layer3();
    test_linear_layer4();
    test_linear_layer5();
    test_linear_layer6();
}
