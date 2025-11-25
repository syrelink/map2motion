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
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        x = x.transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
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
        
        # 生成 Drop path rate 列表
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        self.blocks = nn.ModuleList()
        
        for i in range(num_layers):
            # --- 修改核心逻辑 ---
            # 如果 i 是最后一个索引 (num_layers - 1)，则是 Mamba
            # 否则 (前 num_layers - 1 层) 都是 Transformer
            if i == num_layers - 1:
                is_transformer = False 
            else:
                is_transformer = True
            
            # 打印日志方便确认结构
            print(f'Layer {i} type: {"Transformer" if is_transformer else "Mamba"}')
            
            self.blocks.append(
                ResidualHybridBlock(
                    dim=latent_dim,
                    num_heads=num_heads,
                    window_size=0, 
                    is_transformer_layer=is_transformer, # 传入判断结果
                    mlp_ratio=ff_size / latent_dim if latent_dim > 0 else 4.0,
                    drop=dropout,
                    attn_drop=dropout,
                    drop_path=dpr[i],
                )
            )

    def forward(self, x, src_key_padding_mask=None):
        # x input from cmdm.py is: [Length, Batch, Dim] (L, B, C)
        # src_key_padding_mask is: [Batch, Length] (B, L)

        # 1. 维度转换: (L, B, C) -> (B, L, C)
        # 因为 Mamba 和 我们的 SelfAttention 代码都期望 Batch First
        x = x.permute(1, 0, 2)

        # 2. 逐层处理
        for blk in self.blocks:

            x = blk(x)

        # 3. 维度还原: (B, L, C) -> (L, B, C)
        # 为了配合 cmdm.py 后续的处理
        x = x.permute(1, 0, 2)

        return x


