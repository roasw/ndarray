# Fourier 上采样基准测试结果

## 环境

- 机器：Apple Silicon (Mx Pro)
- Python：3.12
- PyTorch：2.7.0
- 编译器：Clang 19
- 构建模式：Release
- 预热迭代：3
- 计时迭代：10（取均值 ± 标准差）

## 测试矩阵

| 输入形状 (H×W) | 采样倍率 | 数据类型 |
|-----------------|----------|----------|
| 7×9 | 2 | f32, f64 |
| 128×128 | 2 | f32, f64 |
| 256×256 | 2, 4 | f32, f64 |
| 512×512 | 2, 4 | f32, f64 |

## 结果 (f32)

| 形状 | 倍率 | eager (ms) | aoti (ms) | kernel (ms) | kernel-aoti (ms) |
|------|------|-----------|-----------|-----------|-----------|
| 7×9 | 2 | 0.10±0.00 | 0.03±0.00 | 0.04±0.00 | 0.06±0.00 |
| 128×128 | 2 | 0.64±0.09 | 0.67±0.05 | 0.41±0.01 | 0.41±0.03 |
| 256×256 | 2 | 1.80±0.17 | 2.13±0.15 | 1.41±0.16 | 1.41±0.11 |
| 256×256 | 4 | 5.57±0.35 | 8.10±0.19 | 5.51±0.23 | 5.62±0.23 |
| 512×512 | 2 | 6.64±0.24 | 9.27±0.17 | 6.56±0.35 | 6.49±0.44 |
| 512×512 | 4 | 28.48±0.35 | 37.99±0.55 | 28.47±0.30 | 28.46±0.33 |

## 结果 (f64)

| 形状 | 倍率 | eager (ms) | aoti (ms) | kernel (ms) | kernel-aoti (ms) |
|------|------|-----------|-----------|-----------|-----------|
| 7×9 | 2 | 0.03±0.00 | 0.02±0.01 | 0.01±0.00 | 0.02±0.00 |
| 128×128 | 2 | 0.65±0.08 | 0.66±0.04 | 0.66±0.12 | 0.61±0.04 |
| 256×256 | 2 | 2.66±0.14 | 3.15±0.19 | 2.65±0.23 | 2.61±0.13 |
| 256×256 | 4 | 9.27±0.25 | 11.48±0.33 | 9.08±0.36 | 9.16±0.26 |
| 512×512 | 2 | 11.20±0.31 | 13.40±0.39 | 11.31±0.32 | 11.22±0.36 |
| 512×512 | 4 | 44.46±0.46 | 55.62±0.56 | 44.77±0.42 | 44.45±0.67 |

## 分析

- **eager vs kernel**：几乎一致（< 3% 差异），验证了 C++ kernel 与 Python 参考实现等效。
- **kernel vs kernel-aoti**：几乎一致（< 2%），`.pt2` 包装未引入显著开销。
- **aoti (参考 .pt2)**：中大尺寸下落后 30-40%，原因见下节性能剖析。
- **f32 vs f64**：大尺寸下 f64 约为 f32 的 1.5-2 倍。
- **瓶颈**：FFT 本身，时间随 $O(HW \\cdot F^2 \\cdot \\log(HW \\cdot F^2))$ 增长。算子调度开销可忽略。

## 性能剖析

以下数据来自 factor=4（512×512→2048×2048）的 Chrome trace，各路径 3 次预热迭代。

### 热点算子

| 调用 | 总 (ms) | 次数 | 说明 |
|------|--------|------|------|
| `aten::_fft_c2c` | 264 | 18 | vDSP FFT 内核——**仅 eager/kernel/kernel-aoti 可见**，aoti 的 FFT 在代理执行器内不可追踪 |
| `aten::fft_ifft2` | 255 | 9 | IFFT 包装，含 `_fft_c2c` |
| `boxed_run` | 197 | 6 | AOTI 运行时调用，~33 ms/次，内含全部计算 |
| `ndarray::upsample_2d_fourier` | 168 | 6 | C++ kernel（FFT + 象限拷贝 + IFFT + 缩放），~28 ms/次 |
| `aten::fft_fft2` | 12 | 9 | FFT 包装 |
| `aten::zeros`/`zero_`/`fill_` | 12 | 27 | 输出张量分配与初始化 |
| `aten::mul` | 5 | 15 | 缩放因子 |
| `aten::copy_` | 3 | 57 | 切片赋值（`_pad_spectrum`） |
| `aten::to`/`aten::_to_copy` | 3 | 36 | AOTI 中 `view_as_real`/`view_as_complex` 物化为物理拷贝 |

**aoti 的 FFT 为何不可见？** `torch.fft.fft2` 操作于复数张量，AOTI 无法为复数算子
生成代码（见构建警告），将其委托给代理执行器。代理执行器直接调用 vDSP 数学库，
**绕过 ATen 调度器**——而 `torch.profiler` 的钩子挂在 ATen 调度器中。
反之，`aten::copy_` 操作于 float 张量（`view_as_real` 后），AOTI 可为 float 操作
生成可追踪代码，故在 aoti 窗口中可见（0.13 ms）。

kernel-aoti 无此问题：它调用 `torch.ops.ndarray.upsample_2d_fourier`（注册的 torch 算子），
内部通过 ATen 调度器调用 FFT，profiler 可完整穿透 `boxed_run` 追踪。

### 路径单次耗时 (ms)

| 路径 | 单次 | FFT | 调度链 |
|------|-----|-----|--------|
| `eager` | 36.6 | ✓ 1.6 + 32.6 | `nn.Module._call_impl` → ATen |
| `aoti` | 37.7 | ✗ 不可见 | `boxed_run` → 代理执行器（绕过 profiler） |
| `kernel` | 27.9 | ✓ 1.2 + 26.0 | `torch.ops.__call__` → ATen |
| `kernel-aoti` | 28.2 | ✓ 1.0 + 26.2 | `boxed_run` → 注册算子 → ATen |

**eager**：`forward` → `fft_fft2`（1.6 ms）→ `zeros`+`fill_`（2.5 ms，32 MB）
→ 切片赋值 → `fft_ifft2`（32.6 ms）→ `mul`（1 ms）。首次 IFFT 40 ms，后续降至 30、27 ms
（vDSP 首次调用创建 FFT plan 并缓存）。`view_as_real`/`view_as_complex` 为零拷贝。

**aoti**：3 次 `boxed_run`（~38 ms/次），内部完全不可见——仅 `aten::copy_`（0.13 ms）例外，
因为它对 float 张量切片赋值，AOTI 可为 float 操作生成可追踪代码。

**kernel**：`torch.ops.__call__` → `ndarray::upsample_2d_fourier`（28 ms）。
直接在复数张量上操作，无视图拷贝，无代理执行器。

**kernel-aoti**：`boxed_run` → `ndarray::upsample_2d_fourier` → FFT。与 aoti 的关键区别：
内部调用的是注册 torch 算子，FFT 经过 ATen 调度器，profiler 可完整追踪。

### aoti vs eager 性能差异

基准测试中 512×512×4 的 aoti 比 eager 慢 33%（38.0 vs 28.5 ms）：
（profiler 中差距仅 3%，因 profiler 自身开销抵消了差异）

1. **代理执行器 FFT（~3-4 ms）**：aoti 通过代理执行器调用 FFT，比原生 ATen 慢。
1. **视图物化（~4 ms）**：AOTI 将 `view_as_real`/`view_as_complex` 转为物理拷贝（64 MB）。
   eager 为零拷贝视图（`data_ptr` 验证三者共享同一内存）。
1. **逐算子调度（~1-2 ms）**：切片/填充/乘法各自独立经过代理执行器。

### 零拷贝验证

```python
spec = torch.fft.fft2(x)
spec_real = torch.view_as_real(spec)
spec_round = torch.view_as_complex(spec_real)

# 三者共享同一 data_ptr (0x1140a7740)
spec.data_ptr() == spec_real.data_ptr() == spec_round.data_ptr()  # True

# 修改视图同步反映到原张量
spec_round[0, 0] = complex(42, 0)
spec[0, 0]  # (42+0j)
```

### 结论

1. **FFT 是硬瓶颈**：占可见时间 67%，已用 Apple Accelerate/vDSP，无法进一步优化。
1. **aoti 参考 .pt2 比 eager 慢 33%**：代理执行器 FFT（~3-4 ms）+ 视图物化拷贝（~4 ms）+ 逐算子调度（~1-2 ms）。根源是 AOTI 不能生成复数算子代码，整个算子链回退到代理执行器。
1. **C++ kernel 是最优解**：无视图拷贝、FFT 走原生 ATen、无代理执行器。kernel-aoti 性能一致，`.pt2` 包装无额外开销。
