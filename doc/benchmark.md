# 基准测试

## 设计

基准测试集成在 CTest 中，但默认不运行，通过 `benchmark` 标签区分。

## 运行方式

```bash
# 仅运行基准测试
ctest --test-dir build/Debug -L benchmark

# 运行除基准测试外的所有测试（日常使用）
ctest --test-dir build/Debug -LE benchmark

# 运行全部（包含基准测试）
ctest --test-dir build/Debug
```

## 当前覆盖

| 基准测试 | 脚本 |
|------|------|
| Fourier 上采样 | `benchmark/benchmark_upsample_fourier.py` |

## 被测路径

每个基准测试覆盖四条执行路径：

| 路径 | 别名 | 说明 |
|------|------|------|
| Python eager | `eager` | `Upsample2DFourier.forward()` 直接调用 |
| Python AOTI | `aoti` | 参考算法导出的 `.pt2` 包 |
| Kernel eager | `kernel` | `torch.ops.ndarray.upsample_2d_fourier` 直接调用 |
| Kernel AOTI | `kernel-aoti` | kernel 算法导出的 `.pt2` 包 |

## 指标

每条路径对每个 (shape, factor, dtype) 组合执行：

1. **预热**：10 次迭代（不计入结果）
1. **计时**：50 次迭代，取中位数（单位：微秒）
1. **吞吐量**：输入/输出元素数 × 迭代次数 / 总时间

## 添加新基准测试

1. 在 `benchmark/` 目录下创建脚本
1. 在 `tests/CMakeLists.txt` 中注册，确保 `LABELS benchmark`

脚本需支持以下参数：

- `--package-metadata PATH`：参考算法包元数据
- `--kernel-package-metadata PATH`：kernel 算法包元数据
- `--kernel-lib PATH`：torch kernel 共享库
