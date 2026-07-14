import torch
import torch.nn as nn
import torch.nn.functional as F
import math


from torch import Tensor
from typing import Union, List, Tuple, Optional
from functools import reduce
from operator import mul

from mmseg.registry import MODELS

from ..backbone.dinov3_vision_transfromer import DINOv3ViT

from mmseg.utils.typing_utils import ConfigType

class MsRE(nn.Module):
    """

    Args:
        num_layers (int): depth of transformer.
        embed_dims (int): embedding dimension.
        patch_size (int): the patch size.
        token_length (int): feature token lenth. default: 100.
        use_softmax (bool): use softmax or not. default: True.
        link_token_to_query (bool): use feature token to generate (decoder) query or not. default: True.
        scale_init (float): feature scaling factor init value. default: 0.001.
    """  # noqa

    def __init__(
            self,
            num_layers: int,
            embed_dims: int,
            patch_size: int,
            token_length: int = 100,
            use_softmax: bool = True,
            scale_init: float = 0.001,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.embed_dims = embed_dims
        self.patch_size = patch_size
        self.token_length = token_length

        self.scale_init = scale_init
        self.use_softmax = use_softmax
        self.create_model()

    def create_model(self):
        self.learnable_tokens = nn.Parameter(
            torch.empty([self.num_layers, self.token_length, self.embed_dims])
        )
        self.level_encoding = nn.Embedding(3, self.embed_dims)
        self.scale = nn.Parameter(torch.tensor(self.scale_init))
        self.mlp_token2feat = nn.Linear(self.embed_dims, self.embed_dims)
        val = math.sqrt(6.0 / float(3 * reduce(mul, (self.patch_size, self.patch_size), 1) + self.embed_dims))
        nn.init.uniform_(self.learnable_tokens.data, -val, val)
        nn.init.kaiming_uniform_(self.mlp_token2feat.weight, a=math.sqrt(5))


        self.conv1 = nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=3, padding=3 // 2, groups=self.embed_dims)
        self.conv2 = nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=5, padding=5 // 2, groups=self.embed_dims)
        self.conv3 = nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=7, padding=7 // 2, groups=self.embed_dims)

        self.projector = nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=1, )


    def forward(
            self, feats: Tensor, layer: int,
            batch_first=True, has_cls_token=True, num_reg_token=0,
    ) -> Tensor:
        if not batch_first: # N B C ->  B N C
            feats = feats.permute(1, 0, 2)
        if has_cls_token:
            cls_token, feats = torch.tensor_split(feats, [1], dim=1)
        if num_reg_token > 0:
            reg_tokens, feats = torch.split(feats, [num_reg_token, feats.shape[1]-num_reg_token], dim=1)
        tokens = self.learnable_tokens[layer]

        b, n, c = feats.shape
        h = w = int(n**0.5)

        identity = feats
        # feat   B N C -> B C H W
        feats_hw = feats.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()

        conv1_x = self.conv1(feats_hw)
        conv2_x = self.conv2(feats_hw)
        conv3_x = self.conv3(feats_hw)
        # B 3 C H W
        conv_all = torch.stack([conv1_x, conv2_x, conv3_x], dim=1)
        feats_all = conv_all.permute(0, 1, 3, 4, 2).contiguous().reshape(b, 3, n, c)
        level_embed = self.level_encoding.weight[None,:,None,:]
        feats_all = feats_all * level_embed

        attn = torch.einsum("bsnc,mc->bsnm", feats_all, tokens)

        attn = attn * (self.embed_dims ** -0.5)
        attn = F.softmax(attn, dim=-1)

        delta_f = torch.einsum(
            "bsnm,mc->bsnc",
            attn[..., 1:],
            self.mlp_token2feat(tokens[1:, :]),
        )
        # avg
        delta_f = delta_f.sum(dim=1) / 3.0 + identity

        identity = delta_f
        delta_f = delta_f.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        delta_f = self.projector(delta_f)
        delta_f = delta_f.permute(0, 2, 3, 1).reshape(b, n, c).contiguous()

        delta_f = delta_f + identity

        delta_feat = delta_f * self.scale + feats

        cat_feat = []
        if has_cls_token:
            cat_feat.append(cls_token)
        if num_reg_token > 0:
            cat_feat.append(reg_tokens)
        cat_feat.append(delta_feat)
        feats_out = torch.cat(cat_feat, dim=1)
        if not batch_first:
            feats_out = feats_out.permute(1, 0, 2)
        return feats_out

class MsREConv(nn.Module):
    """

    Args:
        num_layers (int): depth of transformer.
        embed_dims (int): embedding dimension.
        patch_size (int): the patch size.
        token_length (int): feature token lenth. default: 100.
        use_softmax (bool): use softmax or not. default: True.
        link_token_to_query (bool): use feature token to generate (decoder) query or not. default: True.
        scale_init (float): feature scaling factor init value. default: 0.001.
    """  # noqa

    def __init__(
            self,
            num_layers: int,
            embed_dims: int,
            patch_size: int,
            token_length: int = 100,
            use_softmax: bool = True,
            scale_init: float = 0.001,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.embed_dims = embed_dims
        self.patch_size = patch_size
        self.token_length = token_length

        self.scale_init = scale_init
        self.use_softmax = use_softmax
        self.create_model()

    def create_model(self):

        self.scale = nn.Parameter(torch.tensor(self.scale_init))


        self.conv1 = nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=3, padding=3 // 2, groups=self.embed_dims)
        self.conv2 = nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=5, padding=5 // 2, groups=self.embed_dims)
        self.conv3 = nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=7, padding=7 // 2, groups=self.embed_dims)

        self.projector = nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=1, )


    def forward(
            self, feats: Tensor, layer: int,
            batch_first=True, has_cls_token=True, num_reg_token=0,
    ) -> Tensor:
        if not batch_first: # N B C ->  B N C
            feats = feats.permute(1, 0, 2)
        if has_cls_token: # TODO 试试 不处理cls token
            cls_token, feats = torch.tensor_split(feats, [1], dim=1)
        if num_reg_token > 0:
            reg_tokens, feats = torch.split(feats, [num_reg_token, feats.shape[1]-num_reg_token], dim=1)
        # tokens = self.learnable_tokens[layer]

        b, n, c = feats.shape
        h = w = int(n**0.5)

        identity = feats
        # feat   B N C -> B C H W
        feats_hw = feats.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()

        conv1_x = self.conv1(feats_hw)
        conv2_x = self.conv2(feats_hw)
        conv3_x = self.conv3(feats_hw)
        # B 3 C H W
        conv_all = torch.stack([conv1_x, conv2_x, conv3_x], dim=1)
        feats_all = conv_all.permute(0, 1, 3, 4, 2).contiguous().reshape(b, 3, n, c)
        delta_f = feats_all
        # avg
        delta_f = delta_f.sum(dim=1) / 3.0 + identity

        identity = delta_f
        delta_f = delta_f.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        delta_f = self.projector(delta_f)
        delta_f = delta_f.permute(0, 2, 3, 1).reshape(b, n, c).contiguous()

        delta_f = delta_f + identity

        delta_feat = delta_f * self.scale + feats

        cat_feat = []
        if has_cls_token:
            cat_feat.append(cls_token)
        if num_reg_token > 0:
            cat_feat.append(reg_tokens)
        cat_feat.append(delta_feat)
        feats_out = torch.cat(cat_feat, dim=1)
        if not batch_first:
            feats_out = feats_out.permute(1, 0, 2)
        return feats_out

@MODELS.register_module()
class MsReDINOv3(DINOv3ViT):
    def __init__(self, msre_config: ConfigType, **kwargs):
        super().__init__(**kwargs)
        tp = msre_config.pop('type', None)
        if tp is None or tp == 'vanilla':
            self.msre: MsRE = MsRE(**msre_config)
        if tp == 'conv':
            self.msre: MsREConv = MsREConv(**msre_config)

    def forward(self,  pixel_values: torch.Tensor,
            bool_masked_pos: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,) -> Tensor:
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
            hidden_states = self.msre(hidden_states, i, batch_first=True, has_cls_token=True,
                                       num_reg_token=self.num_register_tokens)
            if i in self.out_features:
                out_f = self.norm(hidden_states) if i == len(self.layer) - 1 else hidden_states
                out_f = out_f[:, num_extra_tokens:,]
                out_f = out_f.reshape(batch_size, height // patch_size, width // patch_size, -1)
                out_f = out_f.permute(0, 3, 1, 2).contiguous()
                out.append(out_f)

        return tuple(out)


