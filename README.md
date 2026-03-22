# aDeepLearning

一个基于 C++20 模板元编程实现的轻量级深度学习组件库（头文件库）。

项目核心目标是用强类型和编译期约束组织数据结构、算子、层与组合层，强调：

- 类型安全与概念约束（Concepts）
- 表达式式计算与延迟求值
- 头文件库形式，便于集成
- 通过大量单元测试验证基础行为

当前导出的库目标名为 `metann`。

## 项目结构

主要目录说明：

- `data/`：张量基础数据结构（`Matrix`、`Scalar`、`Batch`、`Array` 等）
- `operators/`：一元/二元/三元算子与辅助逻辑
- `layers/`：基础层、组合层、策略与初始化器
- `eval/`：求值相关设施（Eval Plan 等）
- `policy/`：通用策略定义
- `utils/`：类型工具、字典等元编程辅助
- `Tests/`：测试代码（按模块拆分）
- `chapter8/`：实验/示例章节代码（当前较简）

## 环境要求

- CMake >= 3.20
- C++20 编译器
- Ninja（推荐，与 Preset 配套）

建议编译器：

- Windows: MSVC (`cl`) 或 MinGW (`g++`)
- Linux: GCC 或 Clang

注意：项目测试和样例中使用了 `std::format`。若你的标准库实现不完整，可能需要更新编译器/标准库，或按需调整相关测试代码。

## 构建（推荐使用 CMake Presets）

项目已提供 `CMakePresets.json`，可直接使用。

### Windows + MSVC（Debug）

```bash
cmake --preset x64-debug
cmake --build --preset x64-debug
```

### Windows + MinGW（Debug）

```bash
cmake --preset mingw-debug
cmake --build --preset mingw-debug
```

### Linux + GCC（Debug）

```bash
cmake --preset gcc-debug
cmake --build --preset gcc-debug
```

构建产物默认位于：

- `out/build/<preset-name>/`

安装目录默认位于：

- `out/install/<preset-name>/`

## 如何在你的工程中使用

这是头文件库，最简单的集成方式是将本项目作为子目录加入：

```cmake
add_subdirectory(path/to/aDeepLearning)
target_link_libraries(your_target PRIVATE metann)
```

示例包含：

```cpp
#include <metann/data/matrix.hpp>
#include <metann/operators/binary_operators.hpp>

int main()
{
	using namespace metann;
	Matrix<float, CPU> a(2, 2), b(2, 2);
	auto c = a + b;
	auto r = evaluate(c);
	(void)r;
}
```

## 测试说明

`Tests/CMakeLists.txt` 中定义了大量测试可执行文件（如 `array_test`、`add_test`、`linear_layer_test` 等）。

目前这些测试已接入 CTest（通过 `add_test(...)` 注册），可统一发现与执行。

你可以在构建后手动运行对应可执行文件，例如（Windows PowerShell）：

```powershell
./out/build/x64-debug/Tests/array_test.exe
./out/build/x64-debug/Tests/add_test.exe
```

也可以使用 CTest 统一运行（推荐）：

```bash
# 查看测试列表
ctest --test-dir out/build/x64-debug -N

# 运行全部测试
ctest --test-dir out/build/x64-debug --output-on-failure

# 按名称筛选运行（示例）
ctest --test-dir out/build/x64-debug -R softmax_derivative --output-on-failure
```

## 安装与导出

顶层 `CMakeLists.txt` 已配置：

- 安装头文件到 `include/`
- 导出 CMake target：`metann::metann`（通过 `metannTargets.cmake`）

可按需执行安装：

```bash
cmake --install out/build/x64-debug
```

## 当前状态

- 项目主体为头文件实现，接口和测试仍在持续迭代。
- `include/metann` 目录由构建阶段从源码目录头文件同步生成。
- 部分目录（如 `chapter8/`）用于阶段性练习或实验。

## 后续建议

- 将关键测试逐步接入 `add_test(...)`，完善 `ctest` 流程
- 增加一个最小训练/推理示例，帮助新用户快速上手
- 增加 API 参考文档（按 `data/operators/layers` 分组）