"""
ViT-based RS Feature Network for VLCD

This module provides ViT-compatible feature extraction for Side Fusion Network.
It replaces the ResNet-based RSFeatureNetwork for ViT-based CLIP models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import timm

from mmseg.registry import MODELS


@MODELS.register_module()
class ViTFeatureNetwork(nn.Module):
    """
    ViT-compatible RS Feature Network.
    
    Uses a smaller ViT (DeiT-Small) as the side network to match 
    CLIP ViT-B-16's feature structure.
    
    Args:
        pretrained (bool): Whether to load pretrained weights
        embed_dim (int): Embedding dimension (768 to match CLIP ViT-B)
        out_indices (tuple): Which stages to output
    """
    
    def __init__(
        self,
        pretrained=True,
        embed_dim=768,
        out_indices=(3, 5, 7, 11),
        img_size=256,
        patch_size=16
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.out_indices = out_indices
        self.patch_size = patch_size
        self.img_size = img_size
        self.grid_size = img_size // patch_size
        
        # Use DeiT-Small or ViT-Small as side network
        # It's lighter than CLIP's ViT-B but has similar structure
        self.backbone = timm.create_model(
            'deit_small_patch16_224',  # or 'vit_small_patch16_224'
            pretrained=pretrained,
            img_size=img_size,
            num_classes=0,  # Remove classification head
        )
        
        # Get transformer blocks
        self.blocks = self.backbone.blocks
        self.patch_embed = self.backbone.patch_embed
        self.pos_embed = self.backbone.pos_embed
        self.cls_token = self.backbone.cls_token
        self.norm = self.backbone.norm
        
        # Feature dim of DeiT-Small is 384, project to match CLIP's 768
        self.proj = nn.Linear(384, embed_dim)
        
        # FPN-like upsampling for multi-scale features (same as CLIP ViT)
        self.fpn1 = nn.Sequential(
            nn.GroupNorm(1, embed_dim),
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
        )
        self.fpn2 = nn.Sequential(
            nn.GroupNorm(1, embed_dim),
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
        )
        self.fpn3 = nn.GroupNorm(1, embed_dim)
        self.fpn4 = nn.Sequential(
            nn.GroupNorm(1, embed_dim),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        print(f'[ViTFeatureNetwork] Initialized DeiT-Small, '
              f'output dim: {embed_dim}, indices: {out_indices}')
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input images [B, 3, H, W]
            
        Returns:
            List of multi-scale features matching CLIP ViT output format
        """
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add cls token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Get features from specified layers
        features = []
        H = W = self.grid_size
        
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.out_indices:
                # Remove cls token, reshape to spatial
                xp = x[:, 1:, :]  # [B, H*W, C]
                xp = self.proj(xp)  # Project to embed_dim [B, H*W, 768]
                xp = xp.permute(0, 2, 1).reshape(B, -1, H, W)  # [B, 768, H, W]
                features.append(xp.contiguous())
        
        # Apply FPN operations (same as CLIP ViT)
        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        for i in range(len(features)):
            features[i] = ops[i](features[i])
        
        return features


@MODELS.register_module()
class ViTBridgingModule(nn.Module):
    """
    Bridging Module optimized for ViT features.
    
    Handles the token-based structure of ViT features.
    
    Args:
        dim (int): Feature dimension (768 for CLIP ViT-B)
        num_heads (int): Number of attention heads
    """
    
    def __init__(self, dim=768, num_heads=8, use_layer_norm=True):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        
        # Layer normalization
        self.norm_clip = nn.LayerNorm(dim) if use_layer_norm else nn.Identity()
        self.norm_rs = nn.LayerNorm(dim) if use_layer_norm else nn.Identity()
        
        # Cross-attention
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        # Fusion weight
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)
        
        self.scale = (dim // num_heads) ** -0.5
    
    def forward(self, clip_feat, rs_feat):
        """
        Forward pass.
        
        Args:
            clip_feat: CLIP ViT features [B, C, H, W]
            rs_feat: RS ViT features [B, C, H, W]
            
        Returns:
            Fused features [B, C, H, W]
        """
        B, C, H, W = clip_feat.shape
        
        # Reshape to sequence format [B, H*W, C]
        clip_seq = clip_feat.flatten(2).permute(0, 2, 1)  # [B, HW, C]
        rs_seq = rs_feat.flatten(2).permute(0, 2, 1)      # [B, HW, C]
        
        # Normalize
        clip_norm = self.norm_clip(clip_seq)
        rs_norm = self.norm_rs(rs_seq)
        
        # Cross-attention: RS queries, CLIP keys/values
        q = self.q_proj(rs_norm).reshape(B, H*W, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(clip_norm).reshape(B, H*W, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(clip_norm).reshape(B, H*W, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention
        attended = (attn @ v).permute(0, 2, 1, 3).reshape(B, H*W, C)
        attended = self.out_proj(attended)
        
        # Fuse with RS features
        fused_seq = rs_seq + self.alpha * attended
        
        # Reshape back to spatial
        fused = fused_seq.permute(0, 2, 1).reshape(B, C, H, W)
        
        return fused


@MODELS.register_module()
class SideFusionViT(nn.Module):
    """
    Side Fusion Network for ViT-based CLIP.
    
    This module combines frozen CLIP ViT with trainable RS ViT features.
    
    Args:
        clip_backbone: CLIP ViT backbone (will be frozen)
        freeze_clip (bool): Whether to freeze CLIP
        embed_dim (int): Embedding dimension (768)
        out_indices (tuple): Output layer indices
    """
    
    def __init__(
        self,
        clip_backbone,
        freeze_clip=True,
        embed_dim=768,
        out_indices=(3, 5, 7, 11),
        img_size=256
    ):
        super().__init__()
        
        # CLIP ViT backbone (frozen)
        self.clip_backbone = clip_backbone
        if freeze_clip:
            for param in self.clip_backbone.parameters():
                param.requires_grad = False
            print('[SideFusionViT] CLIP ViT backbone frozen')
        
        # RS ViT Network (trainable)
        self.rs_backbone = ViTFeatureNetwork(
            pretrained=True,
            embed_dim=embed_dim,
            out_indices=out_indices,
            img_size=img_size
        )
        
        # Bridging Modules for each output stage
        self.bridging_modules = nn.ModuleList([
            ViTBridgingModule(dim=embed_dim, num_heads=8)
            for _ in range(len(out_indices))
        ])
        
        print(f'[SideFusionViT] Created {len(self.bridging_modules)} bridging modules')
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input images [B, 3, H, W]
            
        Returns:
            List of fused features
        """
        # 1. CLIP features (frozen)
        with torch.no_grad():
            clip_features = self.clip_backbone(x)
        
        # 2. RS features (trainable)
        rs_features = self.rs_backbone(x)
        
        # 3. Fuse features
        fused_features = []
        for i, (clip_feat, rs_feat, bridge) in enumerate(
            zip(clip_features[:-1], rs_features, self.bridging_modules)
        ):
            fused = bridge(clip_feat, rs_feat)
            fused_features.append(fused)
        
        # Keep global/local embedding from CLIP
        if len(clip_features) > len(rs_features):
            fused_features.append(clip_features[-1])
        
        return fused_features
    
    def train(self, mode=True):
        """Keep CLIP in eval mode."""
        super().train(mode)
        self.clip_backbone.eval()
        for param in self.clip_backbone.parameters():
            param.requires_grad = False
        return self
