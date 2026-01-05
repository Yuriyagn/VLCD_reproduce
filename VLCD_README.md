# VLCD 代码复现完成总结

## 概述

已成功在 ChangeCLIP 代码库基础上复现 VLCD (Vision-Language Change Detection) 模型的核心架构。

## 实现的文件

### 1. 核心模块

#### 新创建的文件:

1. **`mmseg/models/utils/learnable_prompt.py`** - CoOp 可学习提示模块
   - 实现了 Context Optimization (CoOp)
   - 包含 `LearnablePrompt` 和 `ContextOptimizedTextEncoder` 类
   - 16个可学习的上下文向量 (默认)

2. **`mmseg/models/backbones/side_fusion_network.py`** - Side Fusion Network
   - `RSFeatureNetwork`: 基于 ResNet50 的遥感特征网络
   - `BridgingModule`: CLIP 与 RS 特征的桥接融合模块  
   - `SideFusionCLIP`: 完整的旁路融合网络封装

3. **`mmseg/models/necks/cfc_module.py`** - CFC 变化特征计算
   - `CFCModule`: 完整的 CFC 实现 (含 rel_feature)
   - `SimpleCFCModule`: 简化版 CFC (不含 rel_feature)

4. **`mmseg/models/segmentors/VLCD.py`** - VLCD 主分割器
   - 集成所有模块的主模型
   - 实现参数冻结策略
   - 包含训练和推理逻辑

#### 配置文件:

5. **`configs/vlcd/vlcd_levir.py`** - VLCD 在 LEVIR-CD 上的配置
   - 参数冻结设置：CLIP 完全冻结
   - 分层学习率：RS Network + Bridging + CoOp 可训练
   - 批次大小调整为 16

#### 测试文件:

6. **`tools/test_vlcd_modules.py`** - 模块功能测试脚本

### 2. 修改的文件

- `mmseg/models/backbones/__init__.py` - 注册 Side Fusion Network
- `mmseg/models/necks/__init__.py` - 注册 CFC 模块
- `mmseg/models/segmentors/__init__.py` - 注册 VLCD 分割器

## 核心设计

### 架构对比

| 组件 | ChangeCLIP | VLCD |
|------|-----------|------|
| **文本端** | 固定模板 + 可学习上下文 | CoOp 可学习提示 (16个向量) |
| **图像端** | 单个 CLIP Backbone | CLIP (冻结) + RS Network + Bridging |
| **特征融合** | Concat + Cosine + Abs Diff | CFC: Concat + Abs Diff + Rel Feature |
| **参数策略** | 微调 CLIP | 冻结 CLIP, 仅训练 SFN + CoOp + Decoder |

### 关键差异

1. **参数冻结**:
   ```python
   # CLIP Vision + Text Encoder 完全冻结
   'backbone': dict(lr_mult=0.0, decay_mult=0.0)
   'text_encoder': dict(lr_mult=0.0, decay_mult=0.0)
   
   # 仅训练新增模块
   'rs_backbone': dict(lr_mult=1.0)
   'bridging_modules': dict(lr_mult=1.0)
   'learnable_prompt': dict(lr_mult=1.0)
   ```

2. **Side Fusion 流程**:
   ```
   Input Image
      ├─→ CLIP Encoder (frozen) ─┐
      │                           ├─→ Bridging Module ─→ Fused Features
      └─→ RS Network (trainable) ─┘
   ```

3. **CFC 变化计算**:
   ```
   Change Features = Concat(
       F1,                          # T1 features
       F2,                          # T2 features  
       |F1 - F2|,                   # Absolute difference
       norm(F_VL) * norm(F_visual)  # Relation feature (新增)
   )
   ```

## 使用方法

### 训练

```bash
# 单GPU训练
python tools/train.py configs/vlcd/vlcd_levir.py

# 多GPU训练
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torcrun --nproc_per_node=4 \
       tools/train.py configs/vlcd/vlcd_levir.py --launcher pytorch
```

### 测试

```bash
python tools/test.py configs/vlcd/vlcd_levir.py \
       work_dirs/vlcd_levir/latest.pth
```

### 验证模块功能

```bash
# 需要激活包含 PyTorch 的环境
conda activate your_env
python tools/test_vlcd_modules.py
```

## 参数统计

预期的参数分布 (以 RN50 为例):

- **CLIP Vision Encoder**: ~38M (冻结)
- **CLIP Text Encoder**: ~63M (冻结)
- **RS Network (ResNet50)**: ~25M (可训练)
- **Bridging Modules**: ~2M (可训练)
- **CoOp Context Vectors**: ~8K (可训练)
- **Decoder**: ~15M (可训练)

**总可训练参数**: ~42M (~30% 的总参数)

## 依赖项

确保已安装以下库:

```bash
pip install torch torchvision
pip install mmcv-full
pip install mmsegmentation
pip install timm  # 用于 ResNet50
pip install ftfy regex  # CLIP 依赖
```

## 注意事项

1. **预训练权重**: 
   - CLIP 预训练权重应放在 `pretrained/RN50.pt` 或 `pretrained/ViT-B-16.pt`
   - RS Network 会自动加载 torchvision 的 ImageNet 预训练权重

2. **内存占用**:
   - 由于同时运行 CLIP 和 RS Network，内存占用会增加
   - 建议使用至少 16GB GPU 显存
   - 可以调整 `batch_size` 来适应不同显存

3. **训练稳定性**:
   - 使用 warmup (前 1500 步) 提高稳定性
   - 冻结 CLIP 可以防止灾难性遗忘
   - Bridging Module 的 `alpha` 参数可调整融合强度

4. **扩展性**:
   - 可以替换 RS Network 为其他backbone (如 ViT)
   - CFC 模块支持简化版 `SimpleCFCModule`
   - CoOp 的上下文向量数量可调 (16 是默认值)

## 下一步工作

- [ ] 在实际数据集上训练验证
- [ ] 对比 ChangeCLIP 和 VLCD 的性能
- [ ] 消融实验：验证各模块的贡献
- [ ] 优化超参数 (学习率、batch size 等)
- [ ] 添加更多数据集的配置文件

## 参考

- VLCD 论文: Vision-Language Change Detection
- CoOp 论文: Learning to Prompt for Vision-Language Models
- ChangeCLIP: Change Detection with CLIP

---

**创建日期**: 2026-01-05  
**版本**: 1.0
