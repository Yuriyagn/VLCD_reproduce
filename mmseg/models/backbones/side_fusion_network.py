"""
Side Fusion Network (SFN) for VLCD

This module implements:
1. RS Feature Network (RFN): A lightweight ResNet for remote sensing features
2. Bridging Module (BM): Fusion mechanism between CLIP and RS features
3. SideFusionCLIP: Complete wrapper that combines CLIP + RFN + BM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import timm

from mmseg.registry import MODELS


class BridgingModule(nn.Module):
    """
    Bridging Module for fusing CLIP features with RS network features.
    
    Implements equations (8-13) from VLCD paper:
    1. Project CLIP features to RS feature dimension
    2. Compute attention map via dot product
    3. Fuse features through weighted sum
    
    Args:
        clip_dim (int): CLIP feature dimension
        rs_dim (int): RS network feature dimension
        spatial_size (tuple): Spatial size for interpolation
    """
    
    def __init__(self, clip_dim, rs_dim, use_layer_norm=True):
        super().__init__()
        
        self.clip_dim = clip_dim
        self.rs_dim = rs_dim
        
        # Layer normalization for CLIP features
        self.layer_norm = nn.LayerNorm(clip_dim) if use_layer_norm else nn.Identity()
        
        # Linear projection: CLIP dim -> RS dim
        self.proj = nn.Linear(clip_dim, rs_dim)
        
        # Optional: learnable fusion weight
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)
        
    def forward(self, clip_feat, rs_feat):
        """
        Forward pass of Bridging Module.
        
        Args:
            clip_feat: CLIP features [B, C_clip, H_clip, W_clip]
            rs_feat: RS network features [B, C_rs, H_rs, W_rs]
            
        Returns:
            Fused features [B, C_rs, H_rs, W_rs]
        """
        B, C_clip, H_clip, W_clip = clip_feat.shape
        B, C_rs, H_rs, W_rs = rs_feat.shape
        
        # 1. Normalize and project CLIP features
        # Reshape to [B, C_clip, H*W] -> [B, H*W, C_clip]
        clip_flat = clip_feat.view(B, C_clip, -1).permute(0, 2, 1)
        clip_norm = self.layer_norm(clip_flat)  # [B, H_clip*W_clip, C_clip]
        clip_proj = self.proj(clip_norm)  # [B, H_clip*W_clip, C_rs]
        
        # 2. Flatten RS features
        rs_flat = rs_feat.view(B, C_rs, -1).permute(0, 2, 1)  # [B, H_rs*W_rs, C_rs]
        
        # 3. Compute attention map
        # A = softmax(F_c^T @ F_rs)
        # [B, H_clip*W_clip, C_rs] @ [B, C_rs, H_rs*W_rs] = [B, H_clip*W_clip, H_rs*W_rs]
        attn_map = torch.bmm(clip_proj, rs_flat.permute(0, 2, 1))
        attn_map = F.softmax(attn_map, dim=1)  # Normalize over CLIP spatial locations
        
        # 4. Apply attention to CLIP features
        # [B, H_rs*W_rs, H_clip*W_clip] @ [B, H_clip*W_clip, C_rs] = [B, H_rs*W_rs, C_rs]
        attended_clip = torch.bmm(attn_map.permute(0, 2, 1), clip_proj)
        
        # 5. Reshape attended CLIP features to match RS spatial size
        attended_clip = attended_clip.permute(0, 2, 1).view(B, C_rs, H_rs, W_rs)
        
        # 6. Fuse with RS features
        # F_fused = F_rs + alpha * attended_clip
        fused = rs_feat + self.alpha * attended_clip
        
        return fused


class RSFeatureNetwork(nn.Module):
    """
    Lightweight RS Feature Network based on ResNet50.
    
    This network runs in parallel with CLIP and is trainable.
    
    Args:
        pretrained (bool): Whether to load ImageNet pretrained weights
        out_indices (tuple): Which stages to output
    """
    
    def __init__(self, pretrained=True, out_indices=(0, 1, 2, 3)):
        super().__init__()
        
        # Use timm to create ResNet50
        self.backbone = timm.create_model(
            'resnet50',
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices
        )
        
        self.out_indices = out_indices
        
        # Get feature dimensions for each stage
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            self.feature_dims = [f.shape[1] for f in features]
        
        print(f'[RSFeatureNetwork] Initialized ResNet50, '
              f'feature dims: {self.feature_dims}')
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input images [B, 3, H, W]
            
        Returns:
            List of multi-scale features
        """
        features = self.backbone(x)
        return features


@MODELS.register_module()
class SideFusionCLIP(nn.Module):
    """
    Side Fusion Network that combines CLIP and RS features.
    
    This is the main module for VLCD's image encoder, which:
    1. Extracts features from frozen CLIP
    2. Extracts features from trainable RS network
    3. Fuses them through Bridging Modules
    
    Args:
        clip_backbone: CLIP vision backbone (will be frozen)
        rs_backbone_cfg (dict): Config for RS backbone
        clip_feature_dims (list): CLIP feature dimensions at each stage
        rs_feature_dims (list): RS feature dimensions at each stage
    """
    
    def __init__(
        self,
        clip_backbone,
        rs_backbone_cfg=None,
        clip_feature_dims=None,
        rs_feature_dims=None,
        freeze_clip=True
    ):
        super().__init__()
        
        # CLIP backbone (frozen)
        self.clip_backbone = clip_backbone
        if freeze_clip:
            for param in self.clip_backbone.parameters():
                param.requires_grad = False
            print('[SideFusionCLIP] CLIP backbone frozen')
        
        # RS Feature Network (trainable)
        if rs_backbone_cfg is None:
            # Default: ResNet50
            self.rs_backbone = RSFeatureNetwork(pretrained=True)
        else:
            self.rs_backbone = MODELS.build(rs_backbone_cfg)
        
        # Infer dimensions if not provided
        if clip_feature_dims is None:
            # Common CLIP RN50 dimensions
            clip_feature_dims = [256, 512, 1024, 2048]
        if rs_feature_dims is None:
            # ResNet50 dimensions
            rs_feature_dims = [256, 512, 1024, 2048]
        
        # Create Bridging Modules for each stage
        self.bridging_modules = nn.ModuleList([
            BridgingModule(clip_dim, rs_dim)
            for clip_dim, rs_dim in zip(clip_feature_dims, rs_feature_dims)
        ])
        
        print(f'[SideFusionCLIP] Created {len(self.bridging_modules)} bridging modules')
    
    def forward(self, x):
        """
        Forward pass with side fusion.
        
        Args:
            x: Input images [B, 3, H, W]
            
        Returns:
            List of fused multi-scale features
        """
        # 1. Extract CLIP features (frozen, no grad)
        with torch.no_grad():
            clip_features = self.clip_backbone(x)
        
        # 2. Extract RS features (trainable)
        rs_features = self.rs_backbone(x)
        
        # 3. Fuse features at each stage through Bridging Modules
        fused_features = []
        for i, (clip_feat, rs_feat, bridge) in enumerate(
            zip(clip_features, rs_features, self.bridging_modules)
        ):
            # Handle different feature formats
            # CLIP may return tuple (global, local) at last layer
            if isinstance(clip_feat, (tuple, list)):
                clip_feat = clip_feat[1]  # Use local features
            if isinstance(rs_feat, (tuple, list)):
                rs_feat = rs_feat[1]
            
            # Fuse through bridging module
            fused = bridge(clip_feat, rs_feat)
            fused_features.append(fused)
        
        # Keep the same output format as CLIP backbone
        # Last layer should return (global, local) tuple
        if len(fused_features) > 0:
            # Use the last feature as both global and local
            last_feat = fused_features[-1]
            global_feat = F.adaptive_avg_pool2d(last_feat, (1, 1)).squeeze(-1).squeeze(-1)
            fused_features[-1] = [global_feat, last_feat]
        
        return fused_features
    
    def train(self, mode=True):
        """
        Override train mode to keep CLIP frozen.
        """
        super().train(mode)
        # Always keep CLIP in eval mode
        self.clip_backbone.eval()
        for param in self.clip_backbone.parameters():
            param.requires_grad = False
        return self
