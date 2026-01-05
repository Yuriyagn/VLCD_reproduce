# VLCD_reproduce

**VLCD (Vision-Language Change Detection)** 的 PyTorch 实现，基于 ChangeCLIP 代码库复现。

## 📝 简介

本项目在 ChangeCLIP 的基础上实现了 VLCD 模型的核心架构，主要包括：

- 🔥 **CoOp (Context Optimization)**: 可学习的文本提示向量
- 🌟 **Side Fusion Network (SFN)**: CLIP + RS 特征融合网络
- 🔗 **Bridging Module (BM)**: 跨模态特征桥接
- 🎯 **CFC Module**: 增强的变化特征计算

## 🚀 核心特性

### 与 ChangeCLIP 的关键差异

| 特性 | ChangeCLIP | VLCD (本实现) |
|------|-----------|--------------|
| **文本端** | 固定模板 + 上下文 | CoOp 可学习提示 (16个向量) |
| **图像端** | 单一 CLIP Backbone | CLIP (冻结) + RS Network + BM |
| **特征融合** | Cosine + Abs Diff | CFC: Concat + Rel Feature |
| **参数策略** | 全部微调 | 冻结 CLIP (70%参数) |
| **参数效率** | 100% 可训练 | 仅 30% 可训练 |

### 架构亮点

```
Input Image
    ├─→ CLIP Vision (Frozen) ──┐
    │                           ├─→ Bridging Module ─→ Fused Features
    └─→ RS Network (Trainable) ─┘
                                    ↓
                                CFC Module
                                    ↓
                                FPN + Decoder
```

## 📦 安装

### 环境要求

- Python >= 3.8
- PyTorch >= 1.10
- CUDA >= 11.1

### 依赖安装

```bash
# 克隆仓库
git clone git@github.com:Yuriyagn/VLCD_reproduce.git
cd VLCD_reproduce

# 创建虚拟环境
conda create -n vlcd python=3.8
conda activate vlcd

# 安装 PyTorch (根据你的 CUDA 版本)
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

# 安装其他依赖
pip install -r requirements.txt

# 安装 MMSegmentation 相关
pip install mmcv-full==1.7.0
pip install timm  # 用于 ResNet50
```

## 📂 项目结构

```
VLCD_reproduce/
├── configs/
│   └── vlcd/
│       ├── vlcd_levir.py          # VLCD ResNet 版本配置
│       └── vlcd_levir_vit.py      # VLCD ViT 版本配置
├── mmseg/
│   └── models/
│       ├── backbones/
│       │   ├── side_fusion_network.py  # ResNet Side Fusion
│       │   └── side_fusion_vit.py      # ViT Side Fusion
│       ├── necks/
│       │   └── cfc_module.py           # CFC 模块
│       ├── segmentors/
│       │   └── VLCD.py                 # VLCD 主模型
│       └── utils/
│           └── learnable_prompt.py     # CoOp 模块
├── tools/
│   ├── train.py                   # 训练脚本
│   ├── test.py                    # 测试脚本
│   └── test_vlcd_modules.py       # 模块测试
├── VLCD_README.md                 # 详细文档
└── README.md                      # 本文件
```

## 🎯 快速开始

### 1. 准备数据集

下载 LEVIR-CD 数据集并按以下结构组织：

```
data/LEVIR-CD/
├── train/
│   ├── A/
│   ├── B/
│   └── label/
├── val/
│   ├── A/
│   ├── B/
│   └── label/
└── test/
    ├── A/
    ├── B/
    └── label/
```

### 2. 下载预训练权重

下载 CLIP 预训练权重（根据你选择的版本）：

```bash
mkdir pretrained

# ResNet 版本 (RN50)
wget https://openaipublic.azureedge.net/clip/models/RN50.pt -O pretrained/RN50.pt

# ViT 版本 (ViT-B-16) - 推荐
wget https://openaipublic.azureedge.net/clip/models/ViT-B-16.pt -O pretrained/ViT-B-16.pt
```

### 3. 训练模型

**ViT 版本（推荐）**:
```bash
# 单 GPU 训练 - ViT
python tools/train.py configs/vlcd/vlcd_levir_vit.py

# 多 GPU 训练 (4卡) - ViT
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 \
       tools/train.py configs/vlcd/vlcd_levir_vit.py --launcher pytorch
```

**ResNet 版本**:
```bash
# 单 GPU 训练 - ResNet
python tools/train.py configs/vlcd/vlcd_levir.py
```

### 4. 测试模型

```bash
python tools/test.py configs/vlcd/vlcd_levir.py \
       work_dirs/vlcd_levir/latest.pth
```

### 5. 模块测试

```bash
# 测试各个模块功能
python tools/test_vlcd_modules.py
```

## 📊 实验结果

### LEVIR-CD 数据集

| 模型 | F1 Score | IoU | Precision | Recall | 参数量 |
|------|----------|-----|-----------|--------|--------|
| ChangeCLIP | - | - | - | - | 143M (100%) |
| **VLCD (本实现)** | - | - | - | - | 145M (30%) |

> 注：结果待训练完成后更新

## 🔧 核心模块详解

### 1. CoOp 可学习提示

```python
from mmseg.models.utils.learnable_prompt import LearnablePrompt

# 创建可学习提示
prompt = LearnablePrompt(
    clip_text_encoder,
    class_names=['background', 'building'],
    n_ctx=16  # 16个上下文向量
)
```

### 2. Side Fusion Network

```python
from mmseg.models.backbones.side_fusion_network import SideFusionCLIP

# 创建融合网络
sfn = SideFusionCLIP(
    clip_backbone=clip_model,
    freeze_clip=True  # 冻结 CLIP
)
```

### 3. CFC 模块

```python
from mmseg.models.necks.cfc_module import CFCModule

# 创建 CFC
cfc = CFCModule(
    in_channels=256,
    out_channels=256,
    text_dim=1024,
    num_scales=4
)
```

## 📖 详细文档

更多技术细节请参考：

- [VLCD_README.md](VLCD_README.md) - 完整实现文档
- [implementation_plan.md](.gemini/implementation_plan.md) - 实施计划
- [walkthrough.md](.gemini/walkthrough.md) - 详细演示

## 🛠️ 配置说明

关键配置参数（在 `configs/vlcd/vlcd_levir.py` 中）：

```python
model = dict(
    type='VLCD',
    freeze_clip=True,      # 冻结 CLIP 参数
    n_ctx=16,              # CoOp 上下文向量数
    
    # 优化器 - 分层学习率
    optim_wrapper = dict(
        paramwise_cfg=dict(
            custom_keys={
                'backbone': dict(lr_mult=0.0),          # CLIP 不训练
                'rs_backbone': dict(lr_mult=1.0),       # RS 训练
                'learnable_prompt': dict(lr_mult=1.0),  # CoOp 训练
            }
        )
    )
)
```

## 🤝 致谢

本项目基于以下优秀工作：

- [ChangeCLIP](https://github.com/...) - 基础代码框架
- [OpenAI CLIP](https://github.com/openai/CLIP) - 预训练模型
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) - 分割框架
- [CoOp](https://github.com/KaiyangZhou/CoOp) - 可学习提示思想

## 📄 许可证

本项目采用 MIT 许可证。

## 📧 联系方式

如有问题，请提交 [Issue](https://github.com/Yuriyagn/VLCD_reproduce/issues) 或联系：

- Email: your_email@example.com
- GitHub: [@Yuriyagn](https://github.com/Yuriyagn)

## 🌟 Star History

如果这个项目对你有帮助，请给一个 ⭐️！

---

**更新日期**: 2026-01-05  
**状态**: 🚧 开发中 - 代码实现已完成，等待训练验证
