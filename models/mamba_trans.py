# 文件名: res_mamba_trans.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp
from timm.models.layers import DropPath
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import rearrange, repeat
import math


# --- MambaVision Mixer 核心模块 ---
class MambaVisionMixer(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto", conv_bias=True, bias=False, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        self.conv1d_x = nn.Conv1d(in_channels=self.d_inner // 2, out_channels=self.d_inner // 2, bias=conv_bias,
                                  kernel_size=d_conv, groups=self.d_inner // 2, padding=d_conv - 1)
        self.conv1d_z = nn.Conv1d(in_channels=self.d_inner // 2, out_channels=self.d_inner // 2, bias=conv_bias,
                                  kernel_size=d_conv, groups=self.d_inner // 2, padding=d_conv - 1)
        self.x_proj = nn.Linear(self.d_inner // 2, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner // 2, bias=True)
        A = repeat(torch.arange(1, self.d_state + 1, dtype=torch.float32), "n -> d n", d=self.d_inner // 2)
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.D = nn.Parameter(torch.ones(self.d_inner // 2))

    def forward(self, hidden_states):
        B, L, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        x, z = xz.chunk(2, dim=2)
        x_ssm_input = x.permute(0, 2, 1).contiguous()
        x_conv_output = self.conv1d_x(x_ssm_input)[:, :, :L]
        x_ssm_input = F.silu(x_conv_output)
        x_for_proj = x_ssm_input.permute(0, 2, 1)
        x_dbl = self.x_proj(x_for_proj)
        dt, B_ssm, C_ssm = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt).permute(0, 2, 1)
        B_ssm = B_ssm.permute(0, 2, 1).contiguous()
        C_ssm = C_ssm.permute(0, 2, 1).contiguous()
        A = -torch.exp(self.A_log.float())
        y_ssm = selective_scan_fn(x_ssm_input, dt, A, B_ssm, C_ssm, self.D.float(), z=None,
                                  delta_bias=self.dt_proj.bias.float(), delta_softplus=True)
        z_conv_input = z.permute(0, 2, 1).contiguous()
        z_conv_output = self.conv1d_z(z_conv_input)[:, :, :L]
        z_non_ssm = F.silu(z_conv_output)
        y = torch.cat([y_ssm, z_non_ssm], dim=1).permute(0, 2, 1)
        return self.out_proj(y)


# --- 自注意力模块 ---
class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()

        # 使用 PyTorch 官方的 MHA 模块
        # 它已经包含了 qkv 投射、注意力计算、和输出投射 (self.proj)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            bias=qkv_bias,  # 控制 qkv 投射的偏置
            dropout=attn_drop,  # 注意力权重的 dropout
            batch_first=True  # 关键！确保输入/输出是 (B, N, C)
        )

        # nn.MultiheadAttention 不包含最后的 proj_drop，所以我们单独保留它
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # MHA 需要 (query, key, value)
        # 在自注意力中, 它们都是 x
        # 它返回 (attn_output, attn_weights)
        attn_output, _ = self.attn(x, x, x)

        # 应用您的原始代码中的最后一步 dropout
        x = self.proj_drop(attn_output)
        return x


# --- 混合模块 Block (支持窗口化) ---
class ResidualHybridBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size, is_transformer_layer=False, mlp_ratio=4., drop=0., attn_drop=0.,
                 drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.is_transformer_layer = is_transformer_layer
        self.window_size = window_size
        if self.is_transformer_layer:
            self.mixer = SelfAttention(dim, num_heads=num_heads, qkv_bias=True, attn_drop=attn_drop, proj_drop=drop)
        else:
            self.mixer = MambaVisionMixer(
                d_model=dim,
                expand=4,  # <-- 尝试新值
                d_state=32,  # <-- 尝试新值
                d_conv=5  # <-- 也可以尝试调整
            )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=nn.GELU, drop=drop)

    def forward(self, x):
        shortcut = x
        x_normed = self.norm1(x)
        if self.is_transformer_layer:
            B, L, C = x_normed.shape
            # 对于一维动作数据，无法假设 L 是完美平方数，此处不再进行窗口化
            # 回退到全局注意力
            x_processed = self.mixer(x_normed)
        else:
            x_processed = self.mixer(x_normed)
        x = shortcut + self.drop_path(x_processed)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# --- 最终骨干网络 ---
class MambaTransBackbone(nn.Module):
    def __init__(self, num_layers, latent_dim, num_heads, ff_size, dropout=0.1, drop_path_rate=0.1):
        super().__init__()
        num_mamba_layers = 1
        # num_mamba_layers = int(num_layers * 0.75)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            is_transformer = i >= num_mamba_layers
            print('trans:',end=" ")
            print(is_transformer)
            self.blocks.append(
                # 注意：由于动作数据是一维的，window_size 参数在此简化版中不再需要
                ResidualHybridBlock(
                    dim=latent_dim,
                    num_heads=num_heads,
                    window_size=0,  # 设为0或移除
                    is_transformer_layer=is_transformer,
                    mlp_ratio=ff_size / latent_dim if latent_dim > 0 else 4.0,
                    drop=dropout,
                    attn_drop=dropout,
                    drop_path=dpr[i],
                )
            )

    def forward(self, x, src_key_padding_mask=None):
        x = x.permute(1, 0, 2)
        for blk in self.blocks:
            # 简化后的全局注意力暂不处理 mask
            x = blk(x)
        x = x.permute(1, 0, 2)
        return x