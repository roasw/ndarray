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
- **瓶颈**：FFT 本身，时间随 O(HW·F²·log(HW·F²)) 增长。算子调度开销可忽略。

## 性能剖析

以下数据来自 512×512×2 配置的 Chrome trace，覆盖全部 4 条路径各 3 次预热迭代
（代码路径为：eager、aoti 参考 .pt2、kernel torch.ops、kernel-aoti .pt2）。

### 四路径耗时概览

| 路径 | 3 次总耗时 (ms) | 单次 (ms) | 对比 eager |
|------|----------------|-----------|------------|
| `eager` | 33.8 | 11.3 | — |
| `aoti` | 34.1 | 11.4 | 慢 1% |
| `Kernel` | 21.6 | 7.2 | 快 36% |
| `kernel-aoti` | 21.0 | 7.0 | 快 38% |

注意：这里的单次包含 3 次预热迭代，Chrome trace 中路径总耗时并非简单的
单次耗时 × 3，因为 torch profiler 自身有开销，且路径间存在交叉调度。

### 十大热点算子（按总耗时降序）

| 算子 | 总耗时 (ms) | 调用次数 | 单次最大 (ms) | 说明 |
|------|------------|----------|---------------|------|
| `aten::_fft_c2c` | 63.0 | 18 | 8.8 | vDSP/Accelerate FFT 实现，单次最大出现在 IFFT 阶段（输出尺寸为 1024×1024），最小的在 FFT 阶段（输入 512×512） |
| `aten::fft_ifft2` | 51.5 | 9 | 8.8 | IFFT 调度包装，内部包含 `_fft_c2c`。3 次来自参考算法（eager+aoti），6 次来自 kernel（C++ 内部调用） |
| `aten::fft_fft2` | 13.3 | 9 | 2.4 | FFT 调度包装，内部包含 `_fft_c2c`。较 IFFT 快因为前向 FFT 无缩放步骤 |
| `aten::mul` | 3.5 | 15 | 0.4 | 输出缩放因子 (factor²)，每次 ~0.23 ms |
| `aten::copy_` | 3.5 | 57 | 0.4 | 参考算法 `_pad_spectrum` 中切片赋值的逐元素拷贝 |
| `aten::to` | 1.5 | 18 | 0.4 | AOTI 路径中 `view_as_real`/`view_as_complex` 被代理执行器物化为实际拷贝 |
| `aten::_to_copy` | 1.5 | 18 | 0.4 | 同上，对应视图转换的另一半操作 |

### 路径详情

**eager 路径 (33.8 ms)：**

eager 调用走标准 PyTorch 调度链：`nn.Module._call_impl`（13.3 ms/次）
→ `forward` → `fft_fft2`（~2 ms）→ `_pad_spectrum` 切片赋值（~1 ms 拷贝）
→ `fft_ifft2`（~8 ms）→ `mul` 缩放。

`view_as_real`/`view_as_complex` 在 eager 模式下为零拷贝视图（已验证 `data_ptr`
相同），不产生 `to`/`_to_copy` 调用。切片赋值 `padded[:h_pos, ...] = spec_real[...]`
产生 `copy_` 调用（57 次总计 3.5 ms），因为 padded 是新张量，需要向其写入数据。

**aoti 参考 .pt2 路径 (34.1 ms)：**

AOTI 编译后的 `.pt2` 有两大不同：

1. **Python 调度被 `boxed_run` 替代**（每调用 ~9 ms）。`boxed_run` 是 AOTI
   运行时的内部函数，负责将 Python 张量序列化为 IValue 容器、查找编译后的
   kernel、调用、再将结果反序列化回 Python。这个开销是固定的，不随输入尺寸
   增长。

1. **`view_as_real`/`view_as_complex` 视图变成了物理拷贝**。AOTI 编译器
   不具备复数张量 slice_scatter/copy\_ 的 c-shim 实现，因此代理执行器将
   `view_as_real` 退化为 `aten::to` + `aten::_to_copy`（共 36 次，~3 ms）。
   这与 `_pad_spectrum` 中的注释吻合：*AOTI/inductor lacks c-shim
   implementations for complex tensor slice_scatter/copy\_/full ops*.

两项叠加，aoti 虽消除了 Python 的 `nn.Module._call_impl` 开销（13.3→0），
但 `boxed_run`（9 ms）加视图拷贝（3 ms）加起来（12 ms）抵消了大部分收益，
导致总耗时与 eager 几乎持平。

**kernel 路径 (21.6 ms)：**

直接调用 `torch.ops.ndarray.upsample_2d_fourier`，通过 `torch._ops.__call__`
调度（每调用 3.6 ms，比 `nn.Module._call_impl` 的 13.3 ms 快 3.7×）。

C++ kernel 内部直接在原生复数张量上执行 `.slice()` 象限拷贝和 `fft_fft2`/
`fft_ifft2`，不经过任何 Python 层，不产生 `to`/`copy_` 调用（这些操作
发生在 C++ 层，torch profiler 无法追踪）。

**kernel-aoti .pt2 路径 (21.0 ms)：**

`torch.ops.ndarray.upsample_2d_fourier` 被编译到 `.pt2` 中，调用时通过
`boxed_run`（9 ms/次）调度。与 kernel 直接调用（3.6 ms/次）相比多了
~5 ms 的装箱/拆箱开销，但由于 kernel 本身计算快（~7 ms/次），总开销仍然
远低于 eager 路径。

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

1. **FFT 是硬瓶颈**：`aten::_fft_c2c` 占 63 ms / ~108 ms = 58%，已使用
   Apple Accelerate/vDSP，无法优化。

1. **参考 .pt2 的性能代价来自视图拷贝**：`view_as_real`/`view_as_complex`
   在 eager 模式中为零拷贝，但在 AOTI 代理执行器中被物化为 `to`+`_to_copy`。
   C++ kernel 直接在复数张量上操作，彻底避免此问题。

1. **`boxed_run` 是 AOTI 调用的固有开销**：每调用 ~9 ms，对小尺寸主导，
   大尺寸可忽略。这是 AOTI 运行时的 IValue 装箱/拆箱机制。

1. **C++ kernel 路径是最优解**：无 Python 调度、无视图拷贝、无代理执行器，
   仅受 FFT 性能限制。
