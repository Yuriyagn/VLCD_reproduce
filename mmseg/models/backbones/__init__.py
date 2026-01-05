# Copyright (c) OpenMMLab. All rights reserved.
# 精简版 - 仅包含 ChangeCLIP 所需的骨干网络

from .clip_backbone import (
    CLIPResNet,
    CLIPResNetWithAttention,
    CLIPVisionTransformer,
    CLIPTextEncoder,
    CLIPTextContextEncoder,
    ContextDecoder
)

from .side_fusion_network import (
    SideFusionCLIP,
    RSFeatureNetwork,
    BridgingModule
)

__all__ = [
    'CLIPResNet', 'CLIPResNetWithAttention', 'CLIPVisionTransformer',
    'CLIPTextEncoder', 'CLIPTextContextEncoder', 'ContextDecoder',
    'SideFusionCLIP', 'RSFeatureNetwork', 'BridgingModule'
]
