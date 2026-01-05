"""
Change Feature Calculation (CFC) Module for VLCD

This module replaces the DFC (Differential Feature Compensation) module
used in ChangeCLIP with VLCD's CFC approach.

Key differences from DFC:
- Uses concatenation instead of complex weighted fusion
- Incorporates rel_feature (Vision-Language correlation)
- Simpler architecture: Concat(F1, F2, abs(F1-F2), rel_feature)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS


@MODELS.register_module()
class CFCModule(nn.Module):
    """
    Change Feature Calculation Module for VLCD.
    
    Computes change features by concatenating:
    1. Features from time T1
    2. Features from time T2
    3. Absolute difference: |F1 - F2|
    4. Relation feature: normalized(F_VL) * normalized(visual_feat)
    
    Args:
        in_channels (int): Input feature channels
        out_channels (int): Output feature channels
        text_dim (int): Text embedding dimension
        num_scales (int): Number of feature scales
        conv_cfg (dict): Config for convolution
        norm_cfg (dict): Config for normalization
    """
    
    def __init__(
        self,
        in_channels,
        out_channels=256,
        text_dim=1024,
        num_scales=4,
        conv_cfg=None,
        norm_cfg=dict(type='SyncBN', requires_grad=True)
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.text_dim = text_dim
        self.num_scales = num_scales
        
        # Calculate concatenated feature channels:
        # F1 + F2 + |F1-F2| + rel_feature = 4 * in_channels
        concat_channels = in_channels * 4
        
        # Fusion convolutions for each scale
        self.fusion_convs = nn.ModuleList([
            ConvModule(
                in_channels=concat_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU')
            )
            for _ in range(num_scales)
        ])
        
        # Projection layers for rel_feature computation
        # Project text embedding to visual feature dimension
        self.text_proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(text_dim, in_channels),
                nn.LayerNorm(in_channels),
                nn.ReLU()
            )
            for _ in range(num_scales)
        ])
        
        print(f'[CFCModule] Initialized with {num_scales} scales, '
              f'in_channels={in_channels}, out_channels={out_channels}')
    
    def compute_rel_feature(self, text_embedding, visual_feat, scale_idx):
        """
        Compute relation feature between text and visual features.
        
        Implements: rel_feature = norm(F_VL) * norm(visual_feat)
        where F_VL is the vision-language fused feature.
        
        Args:
            text_embedding: Text embeddings [B, K, text_dim]
            visual_feat: Visual features [B, C, H, W]
            scale_idx: Index of the current scale
            
        Returns:
            Relation feature [B, C, H, W]
        """
        B, C, H, W = visual_feat.shape
        B, K, text_dim = text_embedding.shape
        
        # Project text embedding to visual dimension
        # [B, K, text_dim] -> [B, K, C]
        text_proj = self.text_proj[scale_idx](text_embedding)
        
        # Compute vision-language correlation
        # Normalize visual features: [B, C, H, W]
        visual_norm = F.normalize(visual_feat, dim=1, p=2)
        
        # Normalize text: [B, K, C]
        text_norm = F.normalize(text_proj, dim=2, p=2)
        
        # Compute correlation map: [B, K, H, W]
        # Einsum: [B, C, H, W] x [B, K, C] -> [B, K, H, W]
        correlation = torch.einsum('bchw,bkc->bkhw', visual_norm, text_norm)
        
        # Aggregate over classes (average or max)
        # Using average pooling over K dimension
        correlation_agg = correlation.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        
        # Expand to match visual feature channels
        rel_feature = correlation_agg.expand(-1, C, -1, -1)  # [B, C, H, W]
        
        # Element-wise multiplication with normalized visual feature
        rel_feature = visual_norm * rel_feature
        
        return rel_feature
    
    def forward(self, feat1, feat2, text_embedding1, text_embedding2=None):
        """
        Forward pass to compute change features.
        
        Args:
            feat1: Features from time T1, list of [B, C, H, W]
            feat2: Features from time T2, list of [B, C, H, W]
            text_embedding1: Text embeddings for T1 [B, K, text_dim]
            text_embedding2: Text embeddings for T2 (optional) [B, K, text_dim]
            
        Returns:
            List of fused change features
        """
        if text_embedding2 is None:
            text_embedding2 = text_embedding1
        
        change_features = []
        
        for i in range(self.num_scales):
            f1 = feat1[i]
            f2 = feat2[i]
            
            # 1. Compute absolute difference
            diff = torch.abs(f1 - f2)
            
            # 2. Compute relation features
            rel_feat1 = self.compute_rel_feature(text_embedding1, f1, i)
            rel_feat2 = self.compute_rel_feature(text_embedding2, f2, i)
            
            # Average relation features from both time steps
            rel_feat = (rel_feat1 + rel_feat2) / 2.0
            
            # 3. Concatenate: [F1, F2, |F1-F2|, rel_feature]
            concat_feat = torch.cat([f1, f2, diff, rel_feat], dim=1)
            
            # 4. Fusion through convolution
            fused = self.fusion_convs[i](concat_feat)
            
            change_features.append(fused)
        
        return change_features


@MODELS.register_module()
class SimpleCFCModule(nn.Module):
    """
    Simplified CFC Module without relation feature.
    
    This is a lighter version that only uses:
    Concat(F1, F2, |F1-F2|)
    
    Can be used for ablation studies or when text features are not available.
    """
    
    def __init__(
        self,
        in_channels,
        out_channels=256,
        num_scales=4,
        conv_cfg=None,
        norm_cfg=dict(type='SyncBN', requires_grad=True)
    ):
        super().__init__()
        
        self.num_scales = num_scales
        
        # F1 + F2 + |F1-F2| = 3 * in_channels
        concat_channels = in_channels * 3
        
        self.fusion_convs = nn.ModuleList([
            ConvModule(
                in_channels=concat_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU')
            )
            for _ in range(num_scales)
        ])
    
    def forward(self, feat1, feat2):
        """
        Forward pass.
        
        Args:
            feat1: Features from time T1, list of [B, C, H, W]
            feat2: Features from time T2, list of [B, C, H, W]
            
        Returns:
            List of fused change features
        """
        change_features = []
        
        for i in range(self.num_scales):
            f1 = feat1[i]
            f2 = feat2[i]
            
            # Compute absolute difference
            diff = torch.abs(f1 - f2)
            
            # Concatenate
            concat_feat = torch.cat([f1, f2, diff], dim=1)
            
            # Fusion
            fused = self.fusion_convs[i](concat_feat)
            
            change_features.append(fused)
        
        return change_features
