import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp
from timm.models.layers import DropPath
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import repeat


class _SelectiveStateSpace(nn.Module):
    """轻量版 Mamba Mixer，遵循官方 selective scan 接口。"""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto", conv_bias=True, bias=False):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        self.out_proj = nn.Linear(self.d_inner * 2, self.d_model, bias=bias)

        self.conv_x = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            bias=conv_bias,
        )

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = repeat(torch.arange(1, self.d_state + 1, dtype=torch.float32), "n -> d n", d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

    def forward(self, x):
        # x: [B, L, d_model]
        B, L, _ = x.shape
        xz = self.in_proj(x)
        x_inner, z_inner = xz.chunk(2, dim=-1)

        # depthwise conv
        x_inner = x_inner.transpose(1, 2)  # [B, d_inner, L]
        x_inner = self.conv_x(x_inner)[..., :L]
        x_inner = F.silu(x_inner).transpose(1, 2)  # [B, L, d_inner]

        params = self.x_proj(x_inner)
        dt, B_ssm, C_ssm = torch.split(params, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt).transpose(1, 2).contiguous()  # [B, d_inner, L]
        B_ssm = B_ssm.transpose(1, 2).contiguous()
        C_ssm = C_ssm.transpose(1, 2).contiguous()
        A = -torch.exp(self.A_log.float())

        y_ssm = selective_scan_fn(
            x_inner.transpose(1, 2).contiguous(),  # [B, d_inner, L]
            dt,
            A,
            B_ssm,
            C_ssm,
            self.D.float(),
            z=None,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
        )
        y_ssm = y_ssm.transpose(1, 2)  # [B, L, d_inner]

        z_inner = F.silu(z_inner)
        y = torch.cat([y_ssm, z_inner], dim=-1)
        return self.out_proj(y)


class MambaBlock(nn.Module):
    """带有 LayerNorm + FFN 的残差 Mamba 模块，兼容 padding_mask。"""

    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.mixer = _SelectiveStateSpace(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = Mlp(in_features=d_model, hidden_features=int(d_model * mlp_ratio), act_layer=nn.GELU, drop=drop)

    def forward(self, x, padding_mask=None):
        # x: [B, L, C], padding_mask: [B, L]
        residual = x
        h = self.norm1(x)
        if padding_mask is not None:
            h = h.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        h = self.mixer(h)
        if padding_mask is not None:
            h = h.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        x = residual + self.drop_path(h)

        residual = x
        h = self.norm2(x)
        h = self.mlp(h)
        if padding_mask is not None:
            h = h.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        x = residual + self.drop_path(h)
        return x


class BidirectionalMambaBlock(nn.Module):
    """双向 Mamba 模块：同时进行前向和后向扫描，然后融合特征。

    这个模块通过在两个方向上运行 selective state space 来捕获双向的上下文信息，
    类似于 Transformer 的双向自注意力，但保持了 Mamba 的线性复杂度优势。
    """

    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
    ):
        super().__init__()
        self.d_model = d_model

        # 前向和后向两个独立的 Mamba mixer
        self.norm1 = nn.LayerNorm(d_model)
        self.mixer_forward = _SelectiveStateSpace(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mixer_backward = _SelectiveStateSpace(d_model, d_state=d_state, d_conv=d_conv, expand=expand)

        # 融合前向和后向特征
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(d_model * 2, d_model),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # FFN
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = Mlp(
            in_features=d_model,
            hidden_features=int(d_model * mlp_ratio),
            act_layer=nn.GELU,
            drop=drop
        )

    def forward(self, x, padding_mask=None):
        """
        Args:
            x: [B, L, C] 输入序列
            padding_mask: [B, L] padding mask，True 表示需要被 mask 的位置

        Returns:
            x: [B, L, C] 输出序列
        """
        # 第一阶段：双向 Mamba
        residual = x
        h = self.norm1(x)

        # 应用 padding mask
        if padding_mask is not None:
            h = h.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        # 前向扫描
        h_forward = self.mixer_forward(h)

        # 后向扫描：先翻转序列，扫描后再翻转回来
        h_backward = torch.flip(h, dims=[1])
        h_backward = self.mixer_backward(h_backward)
        h_backward = torch.flip(h_backward, dims=[1])

        # 融合双向特征
        h = self.fusion(torch.cat([h_forward, h_backward], dim=-1))

        # 应用 padding mask
        if padding_mask is not None:
            h = h.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        x = residual + self.drop_path(h)

        # 第二阶段：FFN
        residual = x
        h = self.norm2(x)
        h = self.mlp(h)

        if padding_mask is not None:
            h = h.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        x = residual + self.drop_path(h)

        return x