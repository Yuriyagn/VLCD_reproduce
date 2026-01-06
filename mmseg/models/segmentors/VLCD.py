# Copyright (c) OpenMMLab. All rights reserved.
"""
VLCD (Vision-Language Change Detection) Segmentor

Key differences from ChangeCLIP:
1. Uses Side Fusion Network instead of direct CLIP features
2. Implements CFC (Change Feature Calculation) instead of DFC
3. Freezes CLIP parameters, only trains CoOp + SFN + Decoder
4. Integrates learnable prompts (CoOp) for text encoding
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmseg.models.utils import resize
from mmcv.cnn import ConvModule

from ..utils.untils import tokenize
from ..utils.learnable_prompt import LearnablePrompt

from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from .base import BaseSegmentor


@MODELS.register_module()
class VLCD(BaseSegmentor):
    """
    VLCD: Vision-Language Change Detection with Side Fusion Network.
    
    This model implements the VLCD architecture with:
    - Frozen CLIP encoder
    - Trainable Side Fusion Network (SFN) with Bridging Modules
    - Context Optimization (CoOp) for learnable prompts
    - Change Feature Calculation (CFC) module
    
    Args:
        backbone (ConfigType): CLIP vision backbone (will be frozen)
        text_encoder (ConfigType): CLIP text encoder (will be frozen)
        context_decoder (ConfigType): Context decoder for VL fusion
        decode_head (ConfigType): Decode head configuration
        rs_backbone (ConfigType): RS Feature Network configuration
        cfc_module (ConfigType): CFC module configuration
        class_names (list): Class names for text prompts
        n_ctx (int): Number of context vectors for CoOp
        freeze_clip (bool): Whether to freeze CLIP parameters
        ... (other args same as ChangeCLIP)
    """
    
    def __init__(self,
                 backbone: ConfigType,
                 text_encoder: ConfigType,
                 context_decoder: ConfigType,
                 decode_head: ConfigType,
                 rs_backbone: ConfigType = None,
                 cfc_module: ConfigType = None,
                 class_names=['background', 'change area'],
                 n_ctx=16,  # Number of context vectors for CoOp
                 context_length=77,
                 context_feature='attention',
                 freeze_clip=True,
                 text_dim=1024,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        
        # Load pretrained weights for CLIP
        if pretrained is not None:
            if 'RN50' not in pretrained and 'RN101' not in pretrained and 'ViT-B' not in pretrained:
                print('[VLCD] Warning: Unknown pretrained weight, using CLIP ViT-B-16')
                pretrained = 'pretrained/ViT-B-16.pt'
            backbone['pretrained'] = pretrained
            text_encoder['pretrained'] = pretrained
        
        # Build CLIP backbone (frozen)
        self.backbone = MODELS.build(backbone)
        if freeze_clip:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print('[VLCD] CLIP Vision Encoder frozen')
        
        # Build CLIP text encoder (frozen)
        self.text_encoder = MODELS.build(text_encoder)
        if freeze_clip:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            print('[VLCD] CLIP Text Encoder frozen')
        
        # Build RS Feature Network (trainable)
        if rs_backbone is not None:
            self.rs_backbone = MODELS.build(rs_backbone)
        else:
            # Default: Use ResNet50 as RS backbone
            from ..backbones.side_fusion_network import RSFeatureNetwork
            self.rs_backbone = RSFeatureNetwork(pretrained=True)
        print('[VLCD] RS Feature Network initialized (trainable)')
        
        # Build Bridging Modules
        # Detect backbone type and set dimensions accordingly
        backbone_type = backbone.get('type', '')
        if 'ViT' in backbone_type or 'VisionTransformer' in backbone_type:
            # ViT: all layers have same dimension (width=768 for ViT-B)
            vit_width = backbone.get('width', 768)
            clip_dims = [vit_width] * 4  # [768, 768, 768, 768]
            rs_dims = [vit_width] * 4
            print(f'[VLCD] Using ViT dimensions: {clip_dims}')
            # Use ViT-specific bridging module
            from ..backbones.side_fusion_vit import ViTBridgingModule
            self.bridging_modules = nn.ModuleList([
                ViTBridgingModule(dim=vit_width, num_heads=8)
                for _ in range(4)
            ])
        else:
            # ResNet: multi-scale dimensions
            clip_dims = [256, 512, 1024, 2048]
            rs_dims = [256, 512, 1024, 2048]
            print(f'[VLCD] Using ResNet dimensions: {clip_dims}')
            from ..backbones.side_fusion_network import BridgingModule
            self.bridging_modules = nn.ModuleList([
                BridgingModule(clip_dim, rs_dim)
                for clip_dim, rs_dim in zip(clip_dims, rs_dims)
            ])
        print(f'[VLCD] Created {len(self.bridging_modules)} Bridging Modules')
        
        # Context decoder
        self.context_decoder = MODELS.build(context_decoder)
        self.context_feature = context_feature
        
        # Learnable Prompts (CoOp)
        self.n_ctx = n_ctx
        self.learnable_prompt = LearnablePrompt(
            self.text_encoder,
            class_names,
            n_ctx=n_ctx
        )
        print(f'[VLCD] Initialized CoOp with {n_ctx} context vectors')
        
        # CFC Module (replaces DFC)
        if cfc_module is not None:
            self.cfc = MODELS.build(cfc_module)
        else:
            # Default CFC configuration
            from ..necks.cfc_module import CFCModule
            self.cfc = CFCModule(
                in_channels=256,
                out_channels=256,
                text_dim=text_dim,
                num_scales=4
            )
        print('[VLCD] CFC Module initialized')
        
        # Neck (FPN)
        if neck is not None:
            self.neck = MODELS.build(neck)
        
        # Decode head
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)
        
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.text_dim = text_dim
        
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
        # Gamma parameter for text-visual fusion
        self.gamma = nn.Parameter(torch.ones(text_dim) * 1e-4)
        
        assert self.with_decode_head
        
        print('[VLCD] Model initialization complete')
        self._print_trainable_params()
    
    def _print_trainable_params(self):
        """Print trainable parameter statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f'[VLCD] Parameter Statistics:')
        print(f'  Total: {total_params:,}')
        print(f'  Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)')
        print(f'  Frozen: {frozen_params:,} ({100*frozen_params/total_params:.2f}%)')
    
    def _init_decode_head(self, decode_head: ConfigType) -> None:
        """Initialize decode_head"""
        self.decode_head = MODELS.build(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
        self.out_channels = self.decode_head.out_channels
    
    def _init_auxiliary_head(self, auxiliary_head: ConfigType) -> None:
        """Initialize auxiliary_head"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(MODELS.build(head_cfg))
            else:
                self.auxiliary_head = MODELS.build(auxiliary_head)
    
    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        """
        Extract features from backbone.
        
        Required by BaseSegmentor abstract method.
        For VLCD, we use extract_feat_with_fusion instead.
        
        Args:
            inputs: Input images [B, 3, H, W]
            
        Returns:
            List of features from backbone
        """
        # Use CLIP backbone directly for basic feature extraction
        return self.backbone(inputs)
    
    def _forward(self, inputs: Tensor, data_samples: OptSampleList = None):
        """
        Network forward process without post-processing.
        
        Required by BaseSegmentor abstract method.
        
        Args:
            inputs: Input tensor
            data_samples: Data samples (optional)
            
        Returns:
            Tuple of tensors from decode head
        """
        # Split into T1 and T2
        inputsA = inputs[:, :3, :, :]
        inputsB = inputs[:, 3:, :, :]
        
        # Extract fused features
        xA = self.extract_feat_with_fusion(inputsA)
        xB = self.extract_feat_with_fusion(inputsB)
        
        # Get text prompts
        if data_samples is not None:
            textA, textB = self.get_cls_text(data_samples)
        else:
            # Use default text if no data_samples
            textA = torch.cat([tokenize(c, context_length=77) 
                              for c in ['background', 'change area']]).unsqueeze(0)
            textB = textA.clone()
            textA = textA.to(inputs.device)
            textB = textB.to(inputs.device)
        
        # Process with context decoder
        text_embA, x_clipA, _ = self.after_extract_feat_vlcd(xA, textA)
        text_embB, x_clipB, _ = self.after_extract_feat_vlcd(xB, textB)
        
        # Apply CFC module
        change_feats = self.cfc(x_clipA, x_clipB, text_embA, text_embB)
        
        # FPN neck
        if self.with_neck:
            change_feats = list(self.neck(change_feats))
        
        # Return features (for tensor mode)
        return change_feats
    
    def extract_feat_with_fusion(self, inputs: Tensor) -> List[Tensor]:
        """
        Extract features using Side Fusion Network.
        
        1. Extract CLIP features (frozen, no grad)
        2. Extract RS features (trainable)
        3. Fuse through Bridging Modules
        
        Args:
            inputs: Input images [B, 3, H, W]
            
        Returns:
            List of fused features
        """
        # 1. CLIP features (frozen)
        with torch.no_grad():
            clip_feats = self.backbone(inputs)
        
        # 2. RS features (trainable)
        rs_feats = self.rs_backbone(inputs)
        
        # 3. Fuse through Bridging Modules
        fused_feats = []
        for i, (clip_feat, rs_feat, bridge) in enumerate(
            zip(clip_feats, rs_feats, self.bridging_modules)
        ):
            # Handle tuple format from last layer
            if isinstance(clip_feat, (tuple, list)):
                clip_feat = clip_feat[1] if len(clip_feat) > 1 else clip_feat[0]
            if isinstance(rs_feat, (tuple, list)):
                rs_feat = rs_feat[1] if len(rs_feat) > 1 else rs_feat[0]
            
            # Fuse
            fused = bridge(clip_feat, rs_feat)
            fused_feats.append(fused)
        
        # Last layer format: [global_feat, local_feat]
        if len(fused_feats) > 0:
            last_feat = fused_feats[-1]
            global_feat = F.adaptive_avg_pool2d(last_feat, (1, 1)).squeeze(-1).squeeze(-1)
            fused_feats[-1] = [global_feat, last_feat]
        
        return fused_feats
    
    def get_text_embeddings(self, texts):
        """
        Get text embeddings with learnable context (CoOp).
        
        Args:
            texts: Tokenized text inputs
            
        Returns:
            Text embeddings [B, K, C]
        """
        # Get learnable context vectors
        ctx_vectors = self.learnable_prompt.get_context_vectors()
        
        # Expand for batch
        B = texts.shape[0]
        contexts = ctx_vectors.unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1)
        
        # Forward through frozen text encoder
        with torch.no_grad():
            text_embeddings = self.text_encoder(texts, contexts)
        
        return text_embeddings
    
    def after_extract_feat_vlcd(self, x, text):
        """
        Process features with context decoder (adapted from ChangeCLIP).
        
        Args:
            x: Extracted features
            text: Tokenized text
            
        Returns:
            text_embeddings, processed_features, score_map
        """
        x_orig = list(x[0:4])
        global_feat, visual_embeddings = x[4]
        
        B, C, H, W = visual_embeddings.shape
        
        # Visual context
        if self.context_feature == 'attention':
            visual_context = torch.cat([
                global_feat.reshape(B, C, 1),
                visual_embeddings.reshape(B, C, H*W)
            ], dim=2).permute(0, 2, 1)  # [B, N, C]
        
        # Text embeddings with CoOp
        text_embeddings = self.get_text_embeddings(text)
        text_embeddings = text_embeddings.expand(B, -1, -1)
        
        # Update text embeddings with visual context
        text_diff = self.context_decoder(text_embeddings, visual_context)
        text_embeddings = text_embeddings + self.gamma * text_diff
        
        # Compute score map
        B, K, C = text_embeddings.shape
        visual_norm = F.normalize(visual_embeddings, dim=1, p=2)
        text_norm = F.normalize(text_embeddings, dim=2, p=2)
        score_map = torch.einsum('bchw,bkc->bkhw', visual_norm, text_norm)
        
        # Concatenate score map (for compatibility)
        # x_orig[3] = torch.cat([x_orig[3], score_map], dim=1)
        
        return text_embeddings, x_orig, score_map
    
    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """
        Calculate losses (training).
        
        Key differences from ChangeCLIP:
        - Uses fused features from SFN
        - Uses CFC module for change calculation
        """
        # Split into T1 and T2
        inputsA = inputs[:, :3, :, :]
        inputsB = inputs[:, 3:, :, :]
        
        # Extract fused features
        xA = self.extract_feat_with_fusion(inputsA)
        xB = self.extract_feat_with_fusion(inputsB)
        
        # Get text prompts
        textA, textB = self.get_cls_text(data_samples)
        
        # Process with context decoder
        text_embA, x_clipA, _ = self.after_extract_feat_vlcd(xA, textA)
        text_embB, x_clipB, _ = self.after_extract_feat_vlcd(xB, textB)
        
        # Apply CFC module
        change_feats = self.cfc(x_clipA, x_clipB, text_embA, text_embB)
        
        # FPN neck
        if self.with_neck:
            change_feats = list(self.neck(change_feats))
        
        # Decode head forward
        losses = dict()
        loss_decode = self.decode_head.loss_changeclip(
            change_feats, text_embA, text_embB,
            data_samples, self.train_cfg
        )
        losses.update(add_prefix(loss_decode, 'decode'))
        
        return losses
    
    def encode_decode(self, inputs: Tensor, batch_img_metas: List[dict]) -> Tensor:
        """Inference encoding and decoding."""
        # Similar to loss() but for inference
        inputsA = inputs[:, :3, :, :]
        inputsB = inputs[:, 3:, :, :]
        
        xA = self.extract_feat_with_fusion(inputsA)
        xB = self.extract_feat_with_fusion(inputsB)
        
        textA, textB = self.get_cls_text(batch_img_metas, train=False)
        
        text_embA, x_clipA, _ = self.after_extract_feat_vlcd(xA, textA)
        text_embB, x_clipB, _ = self.after_extract_feat_vlcd(xB, textB)
        
        change_feats = self.cfc(x_clipA, x_clipB, text_embA, text_embB)
        
        if self.with_neck:
            change_feats = list(self.neck(change_feats))
        
        seg_logits = self.decode_head.predict_with_text(
            change_feats, text_embA, text_embB,
            batch_img_metas, self.test_cfg
        )
        
        return seg_logits
    
    def get_cls_text(self, img_infos, train=True):
        """Get class-specific text prompts (same as ChangeCLIP)."""
        textA = []
        textB = []
        for i in range(len(img_infos)):
            try:
                foreA = ', '.join(['remote sensing image foreground objects'] + img_infos[i].jsonA)
                foreB = ', '.join(['remote sensing image foreground objects'] + img_infos[i].jsonB)
            except:
                foreA = ', '.join(['remote sensing image foreground objects'] + img_infos[i]['jsonA'])
                foreB = ', '.join(['remote sensing image foreground objects'] + img_infos[i]['jsonB'])
            
            backA = ', '.join(['remote sensing image background objects'])
            backB = ', '.join(['remote sensing image background objects'])
            
            textA.append(torch.cat([tokenize(c, context_length=77) for c in [backA, foreA]]).unsqueeze(0))
            textB.append(torch.cat([tokenize(c, context_length=77) for c in [backB, foreB]]).unsqueeze(0))
        
        return torch.cat(textA, dim=0), torch.cat(textB, dim=0)
    
    def predict(self, inputs: Tensor, data_samples: OptSampleList = None) -> SampleList:
        """Predict segmentation results."""
        if data_samples is not None:
            batch_img_metas = [data_sample.metainfo for data_sample in data_samples]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0]
                )
            ] * inputs.shape[0]
        
        seg_logits = self.inference(inputs, batch_img_metas)
        return self.postprocess_result(seg_logits, data_samples)
    
    def inference(self, inputs: Tensor, batch_img_metas: List[dict]) -> Tensor:
        """Inference with slide/whole style."""
        ori_shape = batch_img_metas[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in batch_img_metas)
        
        if self.test_cfg.get('mode', 'whole') == 'slide':
            seg_logit = self.slide_inference(inputs, batch_img_metas)
        else:
            seg_logit = self.whole_inference(inputs, batch_img_metas)
        
        return seg_logit
    
    def whole_inference(self, inputs: Tensor, batch_img_metas: List[dict]) -> Tensor:
        """Inference with full image."""
        seg_logits = self.encode_decode(inputs, batch_img_metas)
        return seg_logits
    
    def slide_inference(self, inputs: Tensor, batch_img_metas: List[dict]) -> Tensor:
        """Inference by sliding window."""
        # Implementation similar to ChangeCLIP
        # Omitted for brevity
        raise NotImplementedError('Slide inference not yet implemented for VLCD')
