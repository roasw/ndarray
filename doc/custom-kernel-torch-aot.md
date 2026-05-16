# 自定义 Torch Kernel 接入与 AOT 编译流程

本文说明如何在本项目中新增一个 C++ CPU 自定义算子，并同时让它可用于：

- `torch.ops` 直接调用
- `torch.export` 导出
- AOTI `.pt2` 编译与 C++ 运行时加载

## 目标结构

- 内核实现：`src/kernel/upsample_2d_fourier.cpp`
- Python 算法封装：`python/ndarry/algorithm/upsample_2d_fourier_kernel.py`
- AOT 编译工具：`tools/aoti-compile.py`
- CMake 接线：`CMakeLists.txt` + `cmake/AddAOTICompileTarget.cmake`
- C++ 运行时封装：`src/algorithm/upsample_2d_fourier_kernel.cpp`

## 在 C++ 注册自定义算子

在 `src/kernel/*.cpp` 中做三件事：

1. 定义 schema（namespace + op 名称）
1. 注册 CPU 实现
1. 注册 Meta 实现（供导出期形状推导）

示例（简化）：

```cpp
TORCH_LIBRARY(ndarray, m) {
    m.def("upsample_2d_fourier(Tensor x, Tensor factor_token) -> Tensor");
}

TORCH_LIBRARY_IMPL(ndarray, CPU, m) {
    m.impl("upsample_2d_fourier", kernel::Upsample2DFourierCpu);
}

TORCH_LIBRARY_IMPL(ndarray, Meta, m) {
    m.impl("upsample_2d_fourier", kernel::Upsample2DFourierMeta);
}
```

说明：

- CPU 实现是真实计算逻辑。
- Meta 实现必须返回正确形状的 meta tensor（不做真实计算）。
- 若缺少 Meta 实现，`torch.export` 通常无法正确处理动态形状。

## 让导出支持动态形状

如果算子依赖上采样倍率，建议把倍率设计成 `Tensor factor_token`（1D），而不是纯 `int`。

- 运行时倍率：`factor = factor_token.size(0)`
- Meta 中使用符号形状：
  - `x.sym_size(0) * factor_token.sym_size(0)`
  - 返回 `at::empty_symint(...)`

这样 `torch.export.Dim("F", ...)` 能参与动态约束。

## 在 Python 算法层封装 torch.ops

新增算法类（例如 `Upsample2DFourierKernel`）：

- `forward()` 内直接调用 `torch.ops.ndarray.upsample_2d_fourier(...)`
- `export(**config)` 中先 `torch.ops.load_library(kernel_lib)`
- 再执行 `torch.export.export(...)`

关键点：

- 不加载 `.so` 时，Python 侧看不到自定义 op。
- `export()` 可同时产出 f32/f64 两个 `ExportedProgram`。

## AOT 编译时注入 kernel 库路径

`tools/aoti-compile.py` 支持 `--config key=value`，所以可在 CMake 中传：

- `kernel_lib=$<TARGET_FILE:${NDARRAY_TORCH_KERNEL_DSO_TARGET}>`

然后在算法 `export(**config)` 里读取 `kernel_lib` 并 `load_library`。

## CMake 接线

需要三类 target：

1. kernel 共享库（注册算子，建议统一成单个 DSO）
1. kernel-backed 算法运行时封装库
1. AOT 导出目标（生成 `.pt2`）

示例要点：

- `set(NDARRAY_TORCH_KERNEL_DSO_TARGET ndarray_torch_kernels)`
- `add_torch_kernel_library(${NDARRAY_TORCH_KERNEL_DSO_TARGET} src/kernel/foo.cpp src/kernel/bar.cpp ...)`
- `add_aoti_compile_target(... ALGORITHM_FILE python/ndarry/algorithm/upsample_2d_fourier_kernel.py ... CONFIG kernel_lib=...)`
- 为 kernel-backed 版本单独定义 `.pt2` 输出名

补充：

- `ALGORITHM_CLASS` 可选；默认自动发现模块内唯一候选类（`nn.Module` 子类且定义 `export`）。
- `OUTPUT_DIR` 可选；默认 `${CMAKE_BINARY_DIR}/artifacts`。
- `DEPENDS` 默认会包含 `ALGORITHM_FILE`；仅在需要额外依赖（如 kernel DSO target）时追加。

## 自定义 op 命名策略

- 采用后端无关的 canonical schema 名：`upsample_2d_fourier`。
- CPU/Meta/CUDA 等实现通过 dispatcher key 分发（`TORCH_LIBRARY_IMPL`），不在 schema 名里编码后端。
- Python 侧统一引用 `python/ndarry/op_names.py` 中的常量，避免字符串散落在多处。

## C++ 运行时加载 `.pt2`

运行时封装类（如 `Upsample2DFourierKernel`）与现有 AOTI 路径一致：

- `AOTIModelPackageLoader(package_path)`
- `run({input_tensor, factor_token})`
- DLPack 与 `ndarray<T>` 互转

额外注意：

- 执行该 runtime 的可执行文件必须链接统一 kernel DSO（或以其他方式确保注册代码已加载）。
- 否则会报：`Could not find schema for ndarray::...`

## 测试建议

至少覆盖以下三类：

1. kernel 直接调用对齐测试（Python）
   - `torch.ops` 输出 vs Python eager 参考实现
1. kernel-backed AOT `.pt2` 对齐测试（Python）
   - `aoti_load_package` 输出 vs eager 参考实现
1. kernel-backed AOT C++ 运行时测试
   - 形状、有限值、f32/f64 行为一致

本项目对应示例：

- `tests/python/test_kernel_upsample_2d_fourier.py`
- `tests/python/test_upsample_2d_fourier_kernel.py`
- `tests/algorithm/upsample_2d_fourier_kernel_runtime.cpp`

## 常见问题

- 导出时报动态约束被常量化：
  - 检查是否有 Meta 实现；是否使用了 `sym_size` + `empty_symint`。
- C++ 运行时报找不到 schema：
  - 检查运行目标是否链接了 kernel 注册库。
- Python 导出找不到 op：
  - 检查 `torch.ops.load_library(kernel_lib)` 是否在 export 前执行。
