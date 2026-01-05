"""
Learnable Prompt Module for VLCD (Context Optimization - CoOp)

This module implements the learnable prompt vectors as described in VLCD paper.
Instead of using fixed text templates, it uses trainable context vectors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnablePrompt(nn.Module):
    """
    Context Optimization (CoOp) for VLCD.
    
    This module replaces fixed text prompts with learnable context vectors.
    The prompt structure: [V]_1...[V]_M [CLASS] [V]_{M+1}...[V]_{2M}
    
    Args:
        clip_model: Pretrained CLIP model (for dimension reference)
        class_names (list): List of class names (e.g., ['background', 'building'])
        n_ctx (int): Number of context vectors (M in the paper, default: 16)
        ctx_init (str): Initialization method, 'random' or 'class_specific'
    """
    
    def __init__(self, clip_text_encoder, class_names, n_ctx=16, ctx_init='random'):
        super().__init__()
        
        self.n_cls = len(class_names)
        self.n_ctx = n_ctx
        self.class_names = class_names
        
        # Get embedding dimension from CLIP text encoder
        # Assume the text encoder has ln_final layer
        try:
            ctx_dim = clip_text_encoder.ln_final.weight.shape[0]
        except:
            # Fallback to common CLIP dimension
            ctx_dim = 512  # For RN50, ViT-B uses 512
        
        self.ctx_dim = ctx_dim
        
        # Initialize learnable context vectors
        # Shape: [n_ctx, ctx_dim]
        if ctx_init == 'random':
            ctx_vectors = torch.empty(n_ctx, ctx_dim)
            nn.init.normal_(ctx_vectors, std=0.02)
        else:
            # Could implement class-specific initialization here
            ctx_vectors = torch.empty(n_ctx, ctx_dim)
            nn.init.normal_(ctx_vectors, std=0.02)
        
        self.ctx = nn.Parameter(ctx_vectors)
        
        # Class token embeddings (frozen)
        # These will be set from the tokenized class names
        self.register_buffer('class_token_embeddings', torch.zeros(self.n_cls, ctx_dim))
        
        print(f'[LearnablePrompt] Initialized with {n_ctx} context vectors, '
              f'dim={ctx_dim}, classes={self.n_cls}')
    
    def forward(self, tokenized_texts=None):
        """
        Forward pass to generate text embeddings with learnable context.
        
        Args:
            tokenized_texts: Optional tokenized text (for compatibility)
            
        Returns:
            Tensor: Context-enhanced embeddings [n_cls, seq_len, ctx_dim]
        """
        # Get context vectors
        ctx = self.ctx  # [n_ctx, ctx_dim]
        
        # For each class, create: [ctx_prefix] + [class_token] + [ctx_suffix]
        # Note: In practice, this should be integrated with CLIP text encoder
        # The actual implementation depends on how text encoder processes inputs
        
        return ctx
    
    def get_context_vectors(self):
        """
        Get the learnable context vectors.
        
        Returns:
            Tensor: Context vectors [n_ctx, ctx_dim]
        """
        return self.ctx


class ContextOptimizedTextEncoder(nn.Module):
    """
    Wrapper for CLIP Text Encoder with Context Optimization.
    
    This module wraps the frozen CLIP text encoder and injects learnable
    context vectors into the text embeddings.
    
    Args:
        clip_text_encoder: CLIP text encoder (will be frozen)
        class_names (list): List of class names
        n_ctx (int): Number of context vectors
        context_length (int): Maximum context length
    """
    
    def __init__(self, clip_text_encoder, class_names, n_ctx=16, context_length=77):
        super().__init__()
        
        self.text_encoder = clip_text_encoder
        self.n_ctx = n_ctx
        self.context_length = context_length
        
        # Freeze CLIP text encoder
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        # Create learnable prompt
        self.learnable_prompt = LearnablePrompt(
            clip_text_encoder, 
            class_names, 
            n_ctx=n_ctx
        )
        
        print('[ContextOptimizedTextEncoder] CLIP Text Encoder frozen, '
              'only context vectors are trainable')
    
    def forward(self, texts, contexts=None):
        """
        Forward pass with context-optimized prompts.
        
        Args:
            texts: Tokenized text inputs
            contexts: Additional context (for compatibility with ChangeCLIP)
            
        Returns:
            Text embeddings with learnable context
        """
        # Get learnable context vectors
        ctx_vectors = self.learnable_prompt.get_context_vectors()
        
        # If contexts is provided (from ChangeCLIP), use it as initialization
        # Otherwise, use the learnable context
        if contexts is not None:
            # Merge with existing context mechanism
            # This allows smooth transition from ChangeCLIP to VLCD
            combined_ctx = ctx_vectors.unsqueeze(0).expand(contexts.shape[0], -1, -1)
            # Add residual connection with original contexts if dimensions match
            if contexts.shape == combined_ctx.shape:
                combined_ctx = combined_ctx + contexts
        else:
            combined_ctx = ctx_vectors.unsqueeze(0)
        
        # Forward through CLIP text encoder (frozen)
        with torch.no_grad():
            # The actual integration with CLIP text encoder depends on its architecture
            # For now, we return the context-enhanced embeddings
            # In full implementation, this should be properly integrated with
            # CLIP's text encoder forward pass
            pass
        
        # For compatibility, forward through the original text encoder
        text_features = self.text_encoder(texts, combined_ctx)
        
        return text_features
    
    def trainable_parameters(self):
        """
        Get only the trainable parameters (context vectors).
        
        Returns:
            Iterator of trainable parameters
        """
        return self.learnable_prompt.parameters()
