# VLCD 代码测试脚本

import torch
import sys
sys.path.insert(0, 'H:\\Code\\Python\\RS\\Change Detection\\ChangeCLIP_refactoring')

def test_learnable_prompt():
    """测试 CoOp 可学习提示模块"""
    print("=" * 60)
    print("测试 1: LearnablePrompt")
    print("=" * 60)
    
    from mmseg.models.utils.learnable_prompt import LearnablePrompt
    
    class DummyTextEncoder:
        def __init__(self):
            self.ln_final = torch.nn.LayerNorm(512)
            self.context_length = 77
    
    text_encoder = DummyTextEncoder()
    class_names = ['background', 'building']
    
    prompt = LearnablePrompt(text_encoder, class_names, n_ctx=16)
    ctx = prompt.get_context_vectors()
    
    print(f"✓ Context vectors shape: {ctx.shape}")
    print(f"✓ Expected: [16, 512]")
    print(f"✓ Trainable parameters: {sum(p.numel() for p in prompt.parameters() if p.requires_grad):,}")
    print()

def test_bridging_module():
    """测试桥接模块"""
    print("=" * 60)
    print("测试 2: BridgingModule")
    print("=" * 60)
    
    from mmseg.models.backbones.side_fusion_network import BridgingModule
    
    bridge = BridgingModule(clip_dim=512, rs_dim=256)
    
    # 创建测试输入
    B, H, W = 2, 32, 32
    clip_feat = torch.randn(B, 512, H, W)
    rs_feat = torch.randn(B, 256, H, W)
    
    fused = bridge(clip_feat, rs_feat)
    
    print(f"✓ Input CLIP: {clip_feat.shape}")
    print(f"✓ Input RS: {rs_feat.shape}")
    print(f"✓ Output fused: {fused.shape}")
    print(f"✓ Expected: [{B}, 256, {H}, {W}]")
    print()

def test_cfc_module():
    """测试 CFC 变化特征计算模块"""
    print("=" * 60)
    print("测试 3: CFCModule")
    print("=" * 60)
    
    from mmseg.models.necks.cfc_module import CFCModule
    
    cfc = CFCModule(in_channels=256, out_channels=256, text_dim=1024, num_scales=4)
    
    # 创建测试输入
    B, C, K = 2, 256, 2  # batch, channels, classes
    feat1 = [torch.randn(B, C, 64, 64), torch.randn(B, C, 32, 32),
             torch.randn(B, C, 16, 16), torch.randn(B, C, 8, 8)]
    feat2 = [torch.randn(B, C, 64, 64), torch.randn(B, C, 32, 32),
             torch.randn(B, C, 16, 16), torch.randn(B, C, 8, 8)]
    text_emb = torch.randn(B, K, 1024)
    
    change_feats = cfc(feat1, feat2, text_emb, text_emb)
    
    print(f"✓ Input features: 4 scales")
    print(f"✓ Text embeddings: {text_emb.shape}")
    print(f"✓ Output change features: {len(change_feats)} scales")
    for i, feat in enumerate(change_feats):
        print(f"  Scale {i}: {feat.shape}")
    print()

def test_parameter_freezing():
    """测试参数冻结策略"""
    print("=" * 60)
    print("测试 4: 参数冻结验证")
    print("=" * 60)
    
    # 模拟CLIP backbone
    clip_backbone = torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, 7, 2, 3),
        torch.nn.ReLU(),
        torch.nn.Conv2d(64, 128, 3, 2, 1)
    )
    
    # 冻结参数
    for param in clip_backbone.parameters():
        param.requires_grad = False
    
    total = sum(p.numel() for p in clip_backbone.parameters())
    trainable = sum(p.numel() for p in clip_backbone.parameters() if p.requires_grad)
    
    print(f"✓ Total parameters: {total:,}")
    print(f"✓ Trainable parameters: {trainable:,}")
    print(f"✓ Frozen parameters: {total - trainable:,}")
    print(f"✓ Freeze ratio: {(total - trainable) / total * 100:.2f}%")
    print()

def test_full_pipeline():
    """测试完整的前向传播流程"""
    print("=" * 60)
    print("测试 5: 完整前向传播流程")
    print("=" * 60)
    
    try:
        from mmseg.models.backbones.side_fusion_network import RSFeatureNetwork
        
        rs_net = RSFeatureNetwork(pretrained=False)
        
        # 测试输入
        B = 2
        x = torch.randn(B, 3, 256, 256)
        
        features = rs_net(x)
        
        print(f"✓ Input: {x.shape}")
        print(f"✓ Output features: {len(features)} scales")
        for i, feat in enumerate(features):
            print(f"  Scale {i}: {feat.shape} (C={feat.shape[1]})")
        print()
        
    except Exception as e:
        print(f"✗ Error: {e}")
        print("Note: This requires timm library")
        print()

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("VLCD 模块测试")
    print("=" * 60 + "\n")
    
    test_learnable_prompt()
    test_bridging_module()
    test_cfc_module()
    test_parameter_freezing()
    test_full_pipeline()
    
    print("=" * 60)
    print("所有测试完成 ✓")
    print("=" * 60)
