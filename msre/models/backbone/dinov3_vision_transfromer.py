import torch
import torch.nn as nn
import numpy as np
import math
from typing import Optional
from mmengine.model import BaseModule

from mmseg.registry import MODELS
from mmcv.cnn.bricks import build_activation_layer
"""
refer: https://github.com/huggingface/transformers/blob/47b0e478f324b54f177ea7998a0791870fdd0324/src/transformers/models/dinov3_vit/modeling_dinov3_vit.py#L133
many changes
"""

class DINOv3ViTEmbeddings(nn.Module):
    """
    Construct the CLS token, mask token, position and patch embeddings.
    """

    def __init__(self, num_channels, hidden_size, patch_size, num_register_tokens):
        super().__init__()
        # self.config = config
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.register_tokens = nn.Parameter(torch.empty(1, num_register_tokens, hidden_size))
        self.patch_embeddings = nn.Conv2d(
            num_channels, hidden_size, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, pixel_values: torch.Tensor, bool_masked_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embeddings.weight.dtype

        # (batch_size, num_channels, height, width) -> (batch_size, num_patches, hidden_size)
        patch_embeddings = self.patch_embeddings(pixel_values.to(dtype=target_dtype))
        patch_embeddings = patch_embeddings.flatten(2).transpose(1, 2)

        if bool_masked_pos is not None:
            mask_token = self.mask_token.to(patch_embeddings.dtype)
            patch_embeddings = torch.where(bool_masked_pos.unsqueeze(-1), mask_token, patch_embeddings)

        # Add CLS and register tokens
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        register_tokens = self.register_tokens.expand(batch_size, -1, -1)
        embeddings = torch.cat([cls_token, register_tokens, patch_embeddings], dim=1)

        return embeddings


def get_patches_center_coordinates(
        num_patches_h: int, num_patches_w: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """
    Computes the 2D coordinates of the centers of image patches, normalized to the range [-1, +1].
    The center of each patch is exactly halfway between its top-left and bottom-right corners.

    Args:
        num_patches_h (int): Number of patches along the vertical (height) axis.
        num_patches_w (int): Number of patches along the horizontal (width) axis.
        dtype (torch.dtype): The desired data type of the returned tensor.

    Returns:
        torch.Tensor: A tensor of shape (height * width, 2), where each row contains the (y, x)
            coordinates of a patch center, normalized to [-1, +1].
    """
    coords_h = torch.arange(0.5, num_patches_h, dtype=dtype, device=device)
    coords_w = torch.arange(0.5, num_patches_w, dtype=dtype, device=device)
    coords_h = coords_h / num_patches_h
    coords_w = coords_w / num_patches_w
    # (height, width, 2) -> (height * width, 2)
    coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)
    coords = coords.flatten(0, 1)
    # Shift range [0, 1] to [-1, +1]
    coords = 2.0 * coords - 1.0
    return coords


def augment_patches_center_coordinates(
        coords: torch.Tensor,
        shift: Optional[float] = None,
        jitter: Optional[float] = None,
        rescale: Optional[float] = None,
) -> torch.Tensor:
    # Shift coords by adding a uniform value in [-shift, shift]
    if shift is not None:
        shift_hw = torch.empty((1, 2), device=coords.device, dtype=coords.dtype)
        shift_hw = shift_hw.uniform_(-shift, shift)
        coords = coords + shift_hw

    # Jitter coords by multiplying the range [-1, 1] by a log-uniform value in [1/jitter, jitter]
    if jitter is not None:
        jitter_range = np.log(jitter)
        jitter_hw = torch.empty((1, 2), device=coords.device, dtype=coords.dtype)
        jitter_hw = jitter_hw.uniform_(-jitter_range, jitter_range).exp()
        coords = coords * jitter_hw

    # Rescale coords by multiplying the range [-1, 1] by a log-uniform value in [1/rescale, rescale]
    if rescale is not None:
        rescale_range = np.log(rescale)
        rescale_hw = torch.empty(1, device=coords.device, dtype=coords.dtype)
        rescale_hw = rescale_hw.uniform_(-rescale_range, rescale_range).exp()
        coords = coords * rescale_hw

    return coords


class DINOv3ViTRopePositionEmbedding(nn.Module):
    inv_freq: torch.Tensor
    """
    DINOv3ViTRopePositionEmbedding
   
    
    Args:        
        rope_theta (`float`, *optional*, defaults to 100.0):
            The base period of the RoPE embeddings.
        hidden_size (`int`, *optional*, defaults to 384):
            Dimensionality of the encoder layers and the pooler layer.
        num_attention_heads (`int`, *optional*, defaults to 6):
            Number of attention heads for each attention layer in the Transformer encoder.    
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        pos_embed_shift (`float`, *optional*):
            Amount to randomly shift position embedding coordinates in [-shift, shift],
            applied only in training mode if not `None`.
        pos_embed_jitter (`float`, *optional*):
            Amount to randomly jitter position embedding coordinates in log-uniform value in [1/jitter, jitter],
            applied only in training mode if not `None`.
        pos_embed_rescale (`float`, *optional*, defaults to 2.0):
            Amount to randomly rescale position embedding coordinates in log-uniform value in [1/rescale, rescale],
            applied only in training mode if not `None`.
        
    """

    def __init__(
            self, rope_theta,
            hidden_size, num_attention_heads,
            image_size, patch_size,
            pos_embed_shift, pos_embed_jitter, pos_embed_rescale
    ):
        super().__init__()

        # self.config = config
        self.base = rope_theta
        self.head_dim = hidden_size // num_attention_heads
        self.num_patches_h = image_size // patch_size
        self.num_patches_w = image_size // patch_size

        self.patch_size = patch_size
        self.pos_embed_shift = pos_embed_shift
        self.pos_embed_jitter = pos_embed_jitter
        self.pos_embed_rescale = pos_embed_rescale

        inv_freq = 1 / self.base ** torch.arange(0, 1, 4 / self.head_dim, dtype=torch.float32)  # (head_dim / 4,)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, height, width = pixel_values.shape
        # num_patches_h = height // self.config.patch_size
        # num_patches_w = width // self.config.patch_size
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size

        device = pixel_values.device
        device_type = device.type if isinstance(device.type, str) and device.type != "mps" else "cpu"

        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            # Although we could precompute static patch_coords from image_size and patch_size in the config,
            # the model was trained with random_scale, so it can process images of varying sizes.
            # Therefore, it's better to compute patch_coords dynamically (with lru_cache).
            patch_coords = get_patches_center_coordinates(
                num_patches_h, num_patches_w, dtype=torch.float32, device=device
            )
            if self.training:
                patch_coords = augment_patches_center_coordinates(
                    patch_coords,
                    shift=self.pos_embed_shift,
                    jitter=self.pos_embed_jitter,
                    rescale=self.pos_embed_rescale,
                )

            # (height * width, 2, head_dim / 4) -> (height * width, head_dim / 2) -> (height * width, head_dim)
            angles = 2 * math.pi * patch_coords[:, :, None] * self.inv_freq[None, None, :]
            angles = angles.flatten(1, 2)
            angles = angles.tile(2)

            cos = torch.cos(angles)
            sin = torch.sin(angles)

        dtype = pixel_values.dtype
        return cos.to(dtype=dtype), sin.to(dtype=dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def eager_attention_forward(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float = 0.0,
        **kwargs,
):
    # Take the dot product between "query" and "key" to get the raw attention scores.
    attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling

    # Normalize the attention scores to probabilities.
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    # Mask heads if we want to
    if attention_mask is not None:
        attn_weights = attn_weights * attention_mask

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def apply_rotary_pos_emb(
        q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embedding to the query and key tensors, but only to the patch tokens,
    ignoring the prefix tokens (cls token and register tokens).

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.

    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """

    num_tokens = q.shape[-2]
    num_patches = sin.shape[-2]
    num_prefix_tokens = num_tokens - num_patches  # cls token + register tokens

    q_prefix_tokens, q_patches = q.split((num_prefix_tokens, num_patches), dim=-2)
    k_prefix_tokens, k_patches = k.split((num_prefix_tokens, num_patches), dim=-2)

    # apply rope only to patch tokens
    q_patches = (q_patches * cos) + (rotate_half(q_patches) * sin)
    k_patches = (k_patches * cos) + (rotate_half(k_patches) * sin)

    q = torch.cat((q_prefix_tokens, q_patches), dim=-2)
    k = torch.cat((k_prefix_tokens, k_patches), dim=-2)

    return q, k


class DINOv3ViTAttention(nn.Module):
    """
    Multi-headed attention compatible with ALL_ATTENTION_FUNCTIONS.
    """

    def __init__(self,
                 hidden_size, num_attention_heads, attention_dropout,
                 query_bias, key_bias, value_bias, proj_bias):
        super().__init__()
        # self.config = config
        self.embed_dim = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.is_causal = False

        self.scaling = self.head_dim ** -0.5
        self.is_causal = False

        self.dropout = attention_dropout
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=key_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=value_bias)

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=query_bias)
        self.o_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=proj_bias)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
            # **kwargs: Unpack[TransformersKwargs],
            **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""

        batch_size, patches, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, patches, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, patches, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, patches, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # attention_interface: Callable = eager_attention_forward
        # if self.config._attn_implementation != "eager":
        #     attention_interface = ALL_ATTENTION_FUNCTIONS['sdpa']
        attn_output, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(batch_size, patches, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class DINOv3ViTLayerScale(nn.Module):
    def __init__(self, layerscale_value, hidden_size) -> None:
        super().__init__()
        self.lambda1 = nn.Parameter(layerscale_value * torch.ones(hidden_size))

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return hidden_state * self.lambda1


def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # binarize
    output = input.div(keep_prob) * random_tensor
    return output


class DINOv3ViTDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f"p={self.drop_prob}"


class DINOv3ViTMLP(nn.Module):
    """

    Args:
        hidden_act:
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            Defaults to ``dict(type='GELU')``
    """

    def __init__(self, hidden_size, intermediate_size, mlp_bias, hidden_act=dict(type='GELU')):
        super().__init__()
        # self.config = config
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=mlp_bias)
        # self.act_fn = ACT2FN[config.hidden_act]
        self.act_fn = build_activation_layer(hidden_act)

    def forward(self, x):
        return self.down_proj(self.act_fn(self.up_proj(x)))


class DINOv3ViTGatedMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, mlp_bias, hidden_act=dict(type='GELU')):
        super().__init__()
        # self.config = config
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=mlp_bias)
        # self.act_fn = ACT2FN[config.hidden_act]
        self.act_fn = build_activation_layer(hidden_act)

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class DINOv3ViTLayer(nn.Module):
    """This corresponds to the Block class in the original implementation.
    Args:
        hidden_size:
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 1536):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer.
        num_attention_heads:
            Number of attention heads for each attention layer in the Transformer encoder.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        query_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the query projection.
        key_bias (`bool`, *optional*, defaults to `False`):
            Whether to add a bias to the key projection.
        value_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the value projection.
        proj_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the output projection.
        mlp_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the MLP layers.
        layerscale_value (`float`, *optional*, defaults to 1.0):
            Initial value to use for layer scale.
        drop_path_rate (`float`, *optional*, defaults to 0.0):
            Stochastic depth rate per sample (when applied in the main path of residual layers).
        use_gated_mlp (`bool`, *optional*, defaults to `False`):
            Whether to use the SwiGLU feedforward neural network.
    """

    def __init__(self,
                 hidden_size, intermediate_size, num_attention_heads,
                 attention_dropout=0., layer_norm_eps=1e-5,
                 query_bias=True, key_bias=False, value_bias=True, proj_bias=True, mlp_bias=True,
                 layerscale_value=1.0, drop_path_rate=0.,
                 use_gated_mlp=False, hidden_act=dict(type='GELU'),
                 ):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.attention = DINOv3ViTAttention(
            hidden_size, num_attention_heads, attention_dropout,
            query_bias, key_bias, value_bias, proj_bias
        )
        self.layer_scale1 = DINOv3ViTLayerScale(layerscale_value, hidden_size)
        self.drop_path = DINOv3ViTDropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        self.norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        if use_gated_mlp:
            self.mlp = DINOv3ViTGatedMLP(hidden_size, intermediate_size, mlp_bias, hidden_act)
        else:
            self.mlp = DINOv3ViTMLP(hidden_size, intermediate_size, mlp_bias, hidden_act)
        self.layer_scale2 = DINOv3ViTLayerScale(layerscale_value, hidden_size)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        # Attention with residual connection
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states, _ = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        )
        hidden_states = self.layer_scale1(hidden_states)
        hidden_states = self.drop_path(hidden_states) + residual

        # MLP with residual connection
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.layer_scale2(hidden_states)
        hidden_states = self.drop_path(hidden_states) + residual

        return hidden_states


@MODELS.register_module()
class DINOv3ViT(BaseModule):
    def __init__(
            self,
            patch_size: int = 16,
            hidden_size: int = 384,
            intermediate_size: int = 1536,
            num_hidden_layers: int = 12,
            num_attention_heads: int = 6,
            hidden_act: dict = dict(type='GELU'),
            attention_dropout: float = 0.0,
            initializer_range: float = 0.02,
            layer_norm_eps: float = 1e-5,
            rope_theta: float = 100.0,
            image_size: int = 224,
            num_channels: int = 3,
            query_bias: bool = True,
            key_bias: bool = False,
            value_bias: bool = True,
            proj_bias: bool = True,
            mlp_bias: bool = True,
            layerscale_value: float = 1.0,
            drop_path_rate: float = 0.0,
            use_gated_mlp: bool = False,
            num_register_tokens: int = 0,
            # train augs
            pos_embed_shift: Optional[float] = None,
            pos_embed_jitter: Optional[float] = None,
            pos_embed_rescale: Optional[float] = 2.0,

            out_features=[5, 11, 17, 23], freeze= False,
            init_cfg=None
    ):
        super().__init__(init_cfg=init_cfg)
        self.embeddings = DINOv3ViTEmbeddings(
            num_channels, hidden_size, patch_size, num_register_tokens)
        self.rope_embeddings = DINOv3ViTRopePositionEmbedding(
            rope_theta,
            hidden_size, num_attention_heads,
            image_size, patch_size,
            pos_embed_shift, pos_embed_jitter, pos_embed_rescale)
        _layer_cfg = dict(
            hidden_size=hidden_size, intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout, layer_norm_eps=layer_norm_eps,
            query_bias=query_bias, key_bias=key_bias, value_bias=value_bias,
            proj_bias=proj_bias, mlp_bias=mlp_bias,
            layerscale_value=layerscale_value, drop_path_rate=drop_path_rate,
            use_gated_mlp=use_gated_mlp, hidden_act=hidden_act,
        )
        self.layer = nn.ModuleList([DINOv3ViTLayer(**_layer_cfg) for _ in range(num_hidden_layers)])
        self.norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens

        self.out_features = out_features
        if freeze:
            for p in self.parameters():
                p.requires_grad_(False)

    def forward(
            self,
            pixel_values: torch.Tensor,
            bool_masked_pos: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,

    ) -> tuple:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, sequence_length)`):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0). Only relevant for
            pre-training.
        """
        batch_size, _, height, width = pixel_values.shape
        patch_size = self.patch_size
        num_extra_tokens = self.num_register_tokens + 1

        pixel_values = pixel_values.to(self.embeddings.patch_embeddings.weight.dtype)
        hidden_states = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)
        position_embeddings = self.rope_embeddings(pixel_values)

        out = []
        for i, layer_module in enumerate(self.layer):
            layer_head_mask = head_mask[i] if head_mask is not None else None
            hidden_states = layer_module(
                hidden_states,
                attention_mask=layer_head_mask,
                position_embeddings=position_embeddings,
            )
            if i in self.out_features:
                out_f = self.norm(hidden_states) if i == len(self.layer) - 1 else hidden_states
                out_f = out_f[:, num_extra_tokens:,]
                out_f = out_f.reshape(batch_size, height // patch_size, width // patch_size, -1)
                out_f = out_f.permute(0, 3, 1, 2).contiguous()
                out.append(out_f)

        return tuple(out)

