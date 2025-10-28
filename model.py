import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (faster than LayerNorm)"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for better length generalization"""
    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None
        
    def _update_cache(self, seq_len, device):
        if seq_len != self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos()[None, None, :, :]
            self._sin_cached = emb.sin()[None, None, :, :]
            
    def forward(self, x):
        seq_len = x.shape[2]
        self._update_cache(seq_len, x.device)
        return self._cos_cached[:, :, :seq_len, :], self._sin_cached[:, :, :seq_len, :]


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embeddings to queries and keys"""
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SwiGLU(nn.Module):
    """SwiGLU activation (used in LLaMA, PaLM)"""
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class FlashMultiHeadAttention(nn.Module):
    """Modern attention using F.scaled_dot_product_attention (Flash Attention 2)"""
    def __init__(self, dim, num_heads, dropout=0.0, use_rope=False):
        super().__init__()
        assert dim % num_heads == 0
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_rope = use_rope
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout
        
        if use_rope:
            self.rope = RotaryEmbedding(self.head_dim)
        
    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        # Apply RoPE if enabled
        if self.use_rope:
            cos, sin = self.rope(q)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Use Flash Attention (automatic kernel selection)
        # is_causal handles the causal mask efficiently
        is_causal = mask is not None and mask.dim() == 2
        attn_mask = mask if not is_causal else None
        
        x = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal
        )
        
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention (GQA) for efficient inference"""
    def __init__(self, dim, num_heads, num_kv_heads=None, dropout=0.0, use_rope=False):
        super().__init__()
        assert dim % num_heads == 0
        
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads  # Default to MHA
        self.num_queries_per_kv = num_heads // self.num_kv_heads
        self.head_dim = dim // num_heads
        self.use_rope = use_rope
        
        self.q = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.k = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout
        
        if use_rope:
            self.rope = RotaryEmbedding(self.head_dim)
        
    def forward(self, x, encoder_out=None, mask=None):
        B, N, C = x.shape
        
        # Query from decoder, Key/Value from encoder (if provided) or self
        kv_input = encoder_out if encoder_out is not None else x
        M = kv_input.shape[1]
        
        q = self.q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(kv_input).view(B, M, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v(kv_input).view(B, M, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE if enabled
        if self.use_rope:
            cos, sin = self.rope(q)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Repeat KV heads to match query heads (for GQA)
        if self.num_kv_heads != self.num_heads:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=1)
        
        # Flash Attention
        is_causal = encoder_out is None and (mask is None or mask.dim() == 2)
        attn_mask = None if is_causal else mask
        
        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal
        )
        
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class PatchEmbedding(nn.Module):
    """Modern patch embedding with optional registers"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, num_registers=0):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.num_registers = num_registers
        
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = RMSNorm(embed_dim)
        
        # Register tokens (Vision Transformers Need Registers, 2023)
        if num_registers > 0:
            self.registers = nn.Parameter(torch.zeros(1, num_registers, embed_dim))
            nn.init.trunc_normal_(self.registers, std=0.02)
        
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        
        if self.num_registers > 0:
            B = x.shape[0]
            registers = self.registers.expand(B, -1, -1)
            x = torch.cat([registers, x], dim=1)
        
        return x


class ViTEncoderBlock(nn.Module):
    """Modern ViT block with RMSNorm, Flash Attention, and SwiGLU"""
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = FlashMultiHeadAttention(dim, num_heads, dropout, use_rope=False)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, int(dim * mlp_ratio), dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class GPT2DecoderBlock(nn.Module):
    """Modern decoder block with GQA, RoPE, and SwiGLU"""
    def __init__(self, dim, num_heads, num_kv_heads=None, mlp_ratio=4.0, dropout=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.self_attn = GroupedQueryAttention(dim, num_heads, num_kv_heads, dropout, use_rope=True)
        self.norm2 = RMSNorm(dim)
        self.cross_attn = GroupedQueryAttention(dim, num_heads, num_kv_heads, dropout, use_rope=False)
        self.norm3 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, int(dim * mlp_ratio), dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        
    def forward(self, x, encoder_out, encoder_mask=None):
        # Self-attention with causal mask
        x = x + self.drop_path(self.self_attn(self.norm1(x)))
        # Cross-attention to encoder
        x = x + self.drop_path(self.cross_attn(self.norm2(x), encoder_out, encoder_mask))
        # FFN
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        return x


class DropPath(nn.Module):
    """Stochastic Depth (Drop Path) for regularization"""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class ViTEncoder(nn.Module):
    """Modern Vision Transformer encoder with all optimizations"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, 
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, 
                 dropout=0.0, drop_path_rate=0.1, num_registers=4):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim, num_registers)
        
        # No learned positional embeddings (patches have implicit spatial position)
        # Register tokens serve as global context
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        self.blocks = nn.ModuleList([
            ViTEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout, dpr[i])
            for i in range(depth)
        ])
        self.norm = RMSNorm(embed_dim)
        
    def forward(self, x):
        x = self.patch_embed(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        return x


class GPT2Decoder(nn.Module):
    """Modern GPT-2 decoder with GQA and RoPE"""
    def __init__(self, vocab_size, max_seq_len=256, embed_dim=768, 
                 depth=12, num_heads=12, num_kv_heads=4, mlp_ratio=4.0, 
                 dropout=0.0, drop_path_rate=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        
        # No positional embeddings - using RoPE instead
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        self.blocks = nn.ModuleList([
            GPT2DecoderBlock(embed_dim, num_heads, num_kv_heads, mlp_ratio, dropout, dpr[i])
            for i in range(depth)
        ])
        self.norm = RMSNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        
        # Weight tying
        self.head.weight = self.token_embed.weight
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        nn.init.normal_(self.token_embed.weight, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
    def forward(self, x, encoder_out, encoder_mask=None):
        x = self.token_embed(x)
        
        for block in self.blocks:
            x = block(x, encoder_out, encoder_mask)
        
        x = self.norm(x)
        logits = self.head(x)
        return logits


class ModernImageCaptioningModel(nn.Module):
    """
    State-of-the-art Image Captioning Model (2024)
    
    Features:
    - Vision Transformer encoder with register tokens
    - GPT-2 decoder with Grouped-Query Attention (GQA)
    - RoPE for positional encoding
    - Flash Attention 2 via F.scaled_dot_product_attention
    - RMSNorm instead of LayerNorm
    - SwiGLU activation
    - Stochastic Depth regularization
    - No learned positional embeddings (implicit in patches + RoPE)
    """
    def __init__(self, 
                 vocab_size,
                 img_size=224,
                 patch_size=16,
                 max_seq_len=256,
                 embed_dim=768,
                 encoder_depth=12,
                 decoder_depth=12,
                 num_heads=12,
                 num_kv_heads=4,  # GQA: fewer KV heads than Q heads
                 mlp_ratio=4.0,
                 dropout=0.0,
                 drop_path_rate=0.1,
                 num_registers=4):
        super().__init__()
        
        self.encoder = ViTEncoder(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
            num_registers=num_registers
        )
        
        self.decoder = GPT2Decoder(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            embed_dim=embed_dim,
            depth=decoder_depth,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            drop_path_rate=drop_path_rate
        )
        
    def forward(self, images, captions):
        """
        Args:
            images: (B, 3, H, W)
            captions: (B, T) token indices
        Returns:
            logits: (B, T, vocab_size)
        """
        encoder_out = self.encoder(images)
        logits = self.decoder(captions, encoder_out)
        return logits
    
    @torch.no_grad()
    def generate(self, images, max_length=50, temperature=1.0, top_k=50, 
                 top_p=0.9, eos_token_id=None, bos_token_id=0):
        """
        Generate captions using top-k and top-p (nucleus) sampling
        
        Args:
            images: (B, 3, H, W)
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            eos_token_id: End-of-sequence token
            bos_token_id: Beginning-of-sequence token
        """
        self.eval()
        B = images.shape[0]
        device = images.device
        
        encoder_out = self.encoder(images)
        generated = torch.full((B, 1), bos_token_id, dtype=torch.long, device=device)
        
        for _ in range(max_length):
            logits = self.decoder(generated, encoder_out)
            logits = logits[:, -1, :] / temperature
            
            # Top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            
            # Early stopping if all sequences generated EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        
        return generated


# Utility function to compile the model for maximum performance
def create_optimized_model(vocab_size, compile_model=True, **kwargs):
    """
    Create an optimized model with optional torch.compile
    
    Args:
        vocab_size: Size of vocabulary
        compile_model: Whether to apply torch.compile (requires PyTorch 2.0+)
        **kwargs: Additional model arguments
    
    Returns:
        Optimized model instance
    """
    model = ModernImageCaptioningModel(vocab_size, **kwargs)
    
    if compile_model and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile for optimal performance...")
        model = torch.compile(model, mode='max-autotune')
    
    return model


# Example usage and performance tips
if __name__ == "__main__":
    # Create model with modern defaults
    model = create_optimized_model(
        vocab_size=50257,  # GPT-2 vocab size
        img_size=224,
        embed_dim=768,
        encoder_depth=12,
        decoder_depth=12,
        num_heads=12,
        num_kv_heads=4,  # 3x fewer KV heads for GQA efficiency
        compile_model=True
    )
    
    # Use bfloat16 for training (better than float16)
    model = model.to(dtype=torch.bfloat16)
    
    # Dummy forward pass
    images = torch.randn(2, 3, 224, 224, dtype=torch.bfloat16)
    captions = torch.randint(0, 50257, (2, 20))
    
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        logits = model(images, captions)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"Output shape: {logits.shape}")
    
    # Generate captions
    generated = model.generate(images, max_length=50, temperature=0.8, top_k=50, top_p=0.9)
    print(f"Generated shape: {generated.shape}")