# Python 算法导出与 AOT 编译

本文说明本仓库中，Python 算法如何变成可在 C++ 里加载执行的 `.pt2` 包。

## 导出入口

以 `python/algorithm/upsample_2d_fourier.py` 为例：

- 类：`Upsample2DFourier`
- 导出方法：`Upsample2DFourier.export(**config)`
- 返回：`dict[str, ExportedProgram]`
  - `upsample_2d_fourier_cpu_f32_model`
  - `upsample_2d_fourier_cpu_f64_model`

`export()` 只产生导出图（`ExportedProgram`），还没有编译成可执行包。

## 编译脚本

脚本：`tools/aoti-compile.py`

- 加载算法类并调用 `export(**config)`
- 默认 `--dump`：写出 `<name>.exported.txt`
- 调用 `torch._inductor.aoti_compile_and_package(...)` 生成 `<name>.pt2`

当前产物目录：`build/Debug/artifacts`

```text
upsample_2d_fourier_cpu_f32_model.exported.txt
upsample_2d_fourier_cpu_f32_model.pt2
upsample_2d_fourier_cpu_f64_model.exported.txt
upsample_2d_fourier_cpu_f64_model.pt2
```

## `torch.export` 文本样例

文件：`build/Debug/artifacts/upsample_2d_fourier_cpu_f32_model.exported.txt`

```text
[graph]
graph():
    %x : [num_users=3] = placeholder[target=x]
    %factor_token : [num_users=1] = placeholder[target=factor_token]
    %sym_size_int_3 : [num_users=2] = call_function[target=torch.ops.aten.sym_size.int](args = (%x, 0), kwargs = {})
    %sym_size_int_4 : [num_users=2] = call_function[target=torch.ops.aten.sym_size.int](args = (%x, 1), kwargs = {})
    %sym_size_int_5 : [num_users=3] = call_function[target=torch.ops.aten.sym_size.int](args = (%factor_token, 0), kwargs = {})
    ...
    %fft_fft2 : [num_users=1] = call_function[target=torch.ops.aten.fft_fft2.default](args = (%x,), kwargs = {})
    %fft_fftshift : [num_users=1] = call_function[target=torch.ops.aten.fft_fftshift.default](args = (%fft_fft2,), kwargs = {})
    %pad : [num_users=1] = call_function[target=torch.ops.aten.pad.default](args = (%fft_fftshift, [...]), kwargs = {})
    %fft_ifftshift : [num_users=1] = call_function[target=torch.ops.aten.fft_ifftshift.default](args = (%pad,), kwargs = {})
    %fft_ifft2 : [num_users=1] = call_function[target=torch.ops.aten.fft_ifft2.default](args = (%fft_ifftshift,), kwargs = {})
    %real : [num_users=1] = call_function[target=torch.ops.aten.real.default](args = (%fft_ifft2,), kwargs = {})
    %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%real, %mul_4), kwargs = {})
    return (mul_5,)
```

这就是导出的可读 IR（图签名、graph、以及对应 Python code）。

## `.pt2` 包结构与 AOT 生成的 C++

`.pt2` 文件是 Zip 包（`archive_format = pt2`），内部示例：

```text
version
archive_format
data/aotinductor/model/c326...wrapper.cpp
data/aotinductor/model/co7i...kernel.cpp
data/aotinductor/model/c326...wrapper.so
data/aotinductor/model/c326...wrapper.json
data/aotinductor/model/c326...wrapper_metadata.json
data/aotinductor/model/co7i...kernel_metadata.json
data/aotinductor/model/custom_objs_config.json
```

可见 `.pt2` 包内同时包含：

- AOT 生成的 C++ 源码（`wrapper.cpp` / `kernel.cpp`）
- 已编译 DSO（`wrapper.so`）
- 元数据 JSON

`wrapper.cpp` 中 `run_impl` 片段示例：

```cpp
void AOTInductorModel::run_impl(
    AtenTensorHandle* input_handles,
    AtenTensorHandle* output_handles,
    DeviceStreamType stream,
    AOTIProxyExecutorHandle proxy_executor
) {
    __check_inputs_outputs(input_handles, output_handles);
    auto inputs = steal_from_raw_handles_to_raii_handles(input_handles, 2);
    auto arg0_1 = std::move(inputs[0]);
    auto arg1_1 = std::move(inputs[1]);
    ...
}
```

这里能直接看到：AOT runtime 入口按 `input_handles`/`output_handles` 执行图。

## CMake 接线

- helper：`cmake/AOTICompile.cmake` 的 `add_aoti_compile_target(...)`
- 调用点：`CMakeLists.txt`
- 当前输出：
  - `${CMAKE_BINARY_DIR}/artifacts/upsample_2d_fourier_cpu_f32_model.pt2`
  - `${CMAKE_BINARY_DIR}/artifacts/upsample_2d_fourier_cpu_f64_model.pt2`

## C++ 嵌入执行

实现：`src/algorithm/upsample_2d_fourier.cpp`

- 通过 `torch::inductor::AOTIModelPackageLoader(package_path)` 加载 `.pt2`
- 传入 `input_tensor` + `factor_token`
- 调用 `package.run({input_tensor, factor_token})`
- 用 DLPack 在 `at::Tensor` 和 `ndarray<T>` 间转换
- 对外提供：
  - `Run(const ndarray<float>&)`（f32 包）
  - `Run(const ndarray<double>&)`（f64 包）
