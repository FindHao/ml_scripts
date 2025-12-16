# Embedding Kernel 2D Tiling 性能分析

本文档分析了两个不同的 embedding kernel 实现，解释为什么 2D tiling 方式能显著提升 L2 cache hit rate，从而将 memory traffic 减少约一半。

## 背景

- **原始实现**: `run2.py` - 使用 1D tiling
- **优化实现**: `run_opt2_for_triton2.py` - 使用 2D tiling

通过 NCU profiler 分析发现：
- L2 cache hit rate 显著提高
- 实际 memory traffic 减少约 50%

## 问题规模

- Embedding Table: `8192 × 4096` (float32) = 128 MB
- Lookup Indices: `8 × 2048` = 16384 个 indices
- Output: `8 × 2048 × 4096` (float32) = 256 MB

## 两种实现的核心差异

### 1D Tiling (run2.py) - 原始实现

```python
# Grid: 65536 blocks (67108864 / 1024)
# XBLOCK = 1024

xoffset = tl.program_id(0) * XBLOCK
xindex = xoffset + tl.arange(0, XBLOCK)[:]
x1 = xindex // 4096   # 哪一行 (embedding lookup index)
x0 = xindex % 4096    # embedding vector 内的位置
```

**访问模式**：每个 block 处理连续的 1024 个输出元素，跨越不同的 embedding indices。

### 2D Tiling (run_opt2_for_triton2.py) - 优化实现

```python
# Grid: (128, 32, 1)
# YBLOCK = 128, XBLOCK = 128

yoffset = tl.program_id(0) * YBLOCK   # 控制 embedding indices
xoffset = tl.program_id(1) * XBLOCK   # 控制 embedding 维度

yindex = yoffset + tl.arange(0, YBLOCK)[None, :]  # shape: [1, YBLOCK]
xindex = xoffset + tl.arange(0, XBLOCK)[:, None]  # shape: [XBLOCK, 1]
```

**访问模式**：将输出划分为 YBLOCK × XBLOCK 的 2D tile，同一 Y-slice 内的 blocks 共享相同的 embedding indices。

## 为什么 L2 Cache Hit Rate 大幅提高

### 关键 Insight：跨 Block 的数据复用

2D tiling 的 grid 结构为 `(128, 32, 1)`：
- Y 方向有 128 个 blocks，每个处理 YBLOCK=128 个 embedding lookups
- X 方向有 32 个 blocks，每个处理 XBLOCK=128 个 embedding 维度

**核心优势**：X 方向的 32 个 blocks 都在处理相同的 128 个 embedding indices！

### 可视化说明

```
Embedding Table: 8192 rows × 4096 cols
Output: 16384 lookups × 4096 dims

2D Tiling Grid: (128, 32, 1)
                  ↑     ↑
                  Y     X
                  
Each Y-slice (128 indices) spans 32 X-blocks:
┌─────────────────────────────────────────────────────────────────────┐
│                    Embedding dimension (4096)                        │
├────────────┬────────────┬────────────┬─────┬───────────────────────┤
│  X-block 0 │  X-block 1 │  X-block 2 │ ... │     X-block 31        │
│ dims 0-127 │dims 128-255│dims 256-383│     │  dims 3968-4095       │
│            │            │            │     │                       │
│    128     │    128     │    128     │     │      128 indices      │
│  indices   │  indices   │  indices   │     │                       │
└────────────┴────────────┴────────────┴─────┴───────────────────────┘
                           ↑
                           |
                  这 32 个 blocks 访问相同的 embedding rows!
```

### Block 访问模式对比

#### 1D Tiling (block 访问模式)

```
Block 0:  [idx_0, dim 0-1023]                      → 读取 embedding_table[idx_0]
Block 1:  [idx_0, dim 1024-2047], [idx_1, dim 0-...] → 读取不同 rows
Block 2:  [idx_1, ...]                              → 完全独立的访问
...

结果：每个 embedding row 被读取 1 次，无法在 blocks 间复用
```

#### 2D Tiling (block 访问模式)

```
Y-Block 0:
  X-Block 0:  [idx 0-127, dim 0-127]      → 读取 rows[idx_0:127]  ← MISS
  X-Block 1:  [idx 0-127, dim 128-255]    → 读取 rows[idx_0:127]  ← HIT (L2)
  X-Block 2:  [idx 0-127, dim 256-383]    → 读取 rows[idx_0:127]  ← HIT (L2)
  ...
  X-Block 31: [idx 0-127, dim 3968-4095]  → 读取 rows[idx_0:127]  ← HIT (L2)

Y-Block 1:
  X-Block 0:  [idx 128-255, dim 0-127]    → 读取 rows[idx_128:255] ← MISS
  ...

结果：每组 128 个 embedding rows 被读取 32 次，但只有第一次 miss
```

## 量化分析

### 数据规模

| 数据 | 大小 |
|------|------|
| Embedding Table | 8192 × 4096 × 4 bytes = 128 MB |
| Output | 8 × 2048 × 4096 × 4 bytes = 256 MB |
| Indices | 8 × 2048 × 8 bytes = 128 KB |
| 每行 Embedding | 4096 × 4 bytes = 16 KB |

### Traffic 对比

| 指标 | 1D Tiling | 2D Tiling |
|------|-----------|-----------|
| Grid size | 65536 × 1 | 128 × 32 |
| Blocks 共享数据 | 无 | 32 个 X-blocks 共享相同 embedding rows |
| 理论 L2 复用率 | ~1× | ~32× (对于每组 Y-blocks) |
| 预期 traffic 减少 | baseline | 约 50% |

### 为什么 Traffic 减半而不是更多？

Traffic 减少约 50% 的原因分析：

1. **Output 写入不变**：输出写入的 256 MB 无法节省
2. **Indices 读取不变**：16384 × 8 bytes 的 index 数据
3. **Embedding table 读取优化**：
   - 原始：每次 lookup 独立读取，cache miss 频繁
   - 优化：32 个 X-blocks 复用相同的 L2 cache line

假设 embedding 读取占总 traffic 的约一半：
- 原始总 traffic ≈ Output + Embedding = 256 MB + 256 MB = 512 MB
- 优化后 ≈ 256 MB + 256 MB / 32 × (实际 miss ratio) ≈ 256 MB + ~128 MB

这与观察到的 "traffic 变成原来一半" 一致。

## L2 Cache 容量考虑

现代 GPU 的 L2 cache 容量：
- NVIDIA A100: 40 MB
- NVIDIA H100: 50 MB

每个 Y-slice (128 个随机 indices) 需要最多 128 × 16 KB = 2 MB 的 embedding 数据。

L2 cache 容量足以容纳多个 Y-slice 的数据，保证了 X-blocks 之间的数据复用。

## 总结

2D tiling 通过 **空间局部性优化** 实现了性能提升：

1. **数据复用**：将需要相同 embedding rows 的计算（不同维度）分配到相邻的 blocks
2. **L2 Cache 友好**：现代 GPU 有足够大的 L2 cache 容纳多个 Y-slice 的 embedding 数据
3. **减少 DRAM 访问**：每个 embedding row 从 DRAM 读取 1 次，后续通过 L2 cache 服务

这是一个经典的 **tile-based memory optimization** 策略，类似于矩阵乘法中的 blocking 优化。核心思想是通过调整计算顺序，使得同一数据在被驱逐出 cache 之前能被多次复用。

## 参考文件

- 原始 1D tiling 实现: `run2.py`
- 优化 2D tiling 实现: `run_opt2_for_triton2.py`
