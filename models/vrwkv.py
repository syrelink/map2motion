# Copyright (c) Shanghai AI Lab. All rights reserved.

from typing import Sequence
import math, os

import logging
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.checkpoint as cp

from mmcv.runner.base_module import BaseModule, ModuleList
from mmcv.cnn.bricks.transformer import PatchEmbed
from mmcls.models.builder import BACKBONES
from mmcls.models.utils import resize_pos_embed
from mmcls.models.backbones.base_backbone import BaseBackbone

from models.tools import DropPath

logger = logging.getLogger(__name__)

from torch.utils.cpp_extension import load

wkv_cuda = load(name="bi_wkv", sources=["/home/supermicro/syr/afford-motion/models/tools/cuda_new/bi_wkv.cpp",
                                        "/home/supermicro/syr/afford-motion/models/tools/cuda_new/bi_wkv_kernel.cu"],
                verbose=True,
                extra_cuda_cflags=['-res-usage', '--maxrregcount 60', '--use_fast_math', '-O3', '-Xptxas -O3',
                                   '-gencode arch=compute_86,code=sm_86'])


class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, u, k, v):
        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        ctx.save_for_backward(w, u, k, v)
        w = w.float().contiguous()
        u = u.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        y = wkv_cuda.bi_wkv_forward(w, u, k, v)
        if half_mode:
            y = y.half()
        elif bf_mode:
            y = y.bfloat16()
        return y

    @staticmethod
    def backward(ctx, gy):
        w, u, k, v = ctx.saved_tensors
        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        gw, gu, gk, gv = wkv_cuda.bi_wkv_backward(w.float().contiguous(),
                                                  u.float().contiguous(),
                                                  k.float().contiguous(),
                                                  v.float().contiguous(),
                                                  gy.float().contiguous())
        if half_mode:
            return (gw.half(), gu.half(), gk.half(), gv.half())
        elif bf_mode:
            return (gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())
        else:
            return (gw, gu, gk, gv)


def RUN_CUDA(w, u, k, v):
    return WKV.apply(w.cuda(), u.cuda(), k.cuda(), v.cuda())


def q_shift(input, shift_pixel=1, gamma=1 / 4, patch_resolution=None):
    assert gamma <= 1 / 4
    B, N, C = input.shape
    # 步骤1：将一维序列还原为二维图像特征图
    # 输入的 shape 是 [批大小, 序列长度, 通道数]
    # 这里将其变形成 [批大小, 通道数, 高度, 宽度]，以恢复图像的空间结构
    input = input.transpose(1, 2).reshape(B, C, patch_resolution[0], patch_resolution[1])
    B, C, H, W = input.shape
    output = torch.zeros_like(input)

    # 步骤2：将通道分为四个部分，分别向不同方向平移
    # (gamma 默认为 1/4，所以 C*gamma 就是 1/4 的通道数)

    # 第 1/4 通道：向右平移一个像素
    # 将 [:, :, :, 0:W-1] 的内容，赋值给 [:, :, :, 1:W]
    output[:, 0:int(C * gamma), :, shift_pixel:W] = input[:, 0:int(C * gamma), :, 0:W - shift_pixel]

    # 第 2/4 通道：向左平移一个像素
    output[:, int(C * gamma):int(C * gamma * 2), :, 0:W - shift_pixel] = input[
        :, int(C * gamma):int(C * gamma * 2), :, shift_pixel:W]

    # 第 3/4 通道：向下平移一个像素
    output[:, int(C * gamma * 2):int(C * gamma * 3), shift_pixel:H, :] = input[
        :, int(C * gamma * 2):int(C * gamma * 3), 0:H - shift_pixel, :]

    # 第 4/4 通道：向上平移一个像素
    output[:, int(C * gamma * 3):int(C * gamma * 4), 0:H - shift_pixel, :] = input[
        :, int(C * gamma * 3):int(C * gamma * 4), shift_pixel:H, :]

    # (可选) 剩余通道保持不变
    output[:, int(C * gamma * 4):, ...] = input[:, int(C * gamma * 4):, ...]

    # 步骤3：将二维特征图重新展平为一维序列
    return output.flatten(2).transpose(1, 2)


def token_shift(x):
    """
    适用于一维序列的 Token Shift 操作。
    它的作用是让每个时间步的 token 与其上一个时间步的 token 进行信息混合。

    Args:
        x (torch.Tensor): 输入张量，形状为 [批大小, 序列长度, 特征维度]

    Returns:
        torch.Tensor: 返回一个“过去”信息的张量，形状与输入相同。
    """
    # 沿着序列长度（时间）维度，将所有 token 向后平移一位
    # 使用 F.pad 在序列的开头填充 0，并舍弃最后一个元素
    # [B, T, C] -> [B, T+1, C] -> [B, T, C]
    shifted_x = F.pad(x, (0, 0, 1, 0))[:, :-1, :]

    return shifted_x


class VRWKV_SpatialMix(BaseModule):
    def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                 channel_gamma=1 / 4, shift_pixel=1, init_mode='fancy',
                 key_norm=False, with_cp=False):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.device = None
        attn_sz = n_embd
        self._init_weights(init_mode)
        self.shift_pixel = shift_pixel
        self.shift_mode = shift_mode
        if shift_pixel > 0:
            self.shift_func = eval(shift_mode)
            self.channel_gamma = channel_gamma
        else:
            self.spatial_mix_k = None
            self.spatial_mix_v = None
            self.spatial_mix_r = None

        self.key = nn.Linear(n_embd, attn_sz, bias=False)
        self.value = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(n_embd, attn_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(n_embd)
        else:
            self.key_norm = None
        self.output = nn.Linear(attn_sz, n_embd, bias=False)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

        self.with_cp = with_cp

    def _init_weights(self, init_mode):
        if init_mode == 'fancy':
            with torch.no_grad():  # fancy init
                ratio_0_to_1 = (self.layer_id / (self.n_layer - 1))  # 0 to 1
                ratio_1_to_almost0 = (1.0 - (self.layer_id / self.n_layer))  # 1 to ~0

                # fancy time_decay
                decay_speed = torch.ones(self.n_embd)
                for h in range(self.n_embd):
                    decay_speed[h] = -5 + 8 * (h / (self.n_embd - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
                self.spatial_decay = nn.Parameter(decay_speed)

                # fancy time_first
                zigzag = (torch.tensor([(i + 1) % 3 - 1 for i in range(self.n_embd)]) * 0.5)
                self.spatial_first = nn.Parameter(torch.ones(self.n_embd) * math.log(0.3) + zigzag)

                # fancy time_mix
                x = torch.ones(1, 1, self.n_embd)
                for i in range(self.n_embd):
                    x[0, 0, i] = i / self.n_embd
                self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
                self.spatial_mix_v = nn.Parameter(torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
                self.spatial_mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))
        elif init_mode == 'local':
            self.spatial_decay = nn.Parameter(torch.ones(self.n_embd))
            self.spatial_first = nn.Parameter(torch.ones(self.n_embd))
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]))
            self.spatial_mix_v = nn.Parameter(torch.ones([1, 1, self.n_embd]))
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]))
        elif init_mode == 'global':
            self.spatial_decay = nn.Parameter(torch.zeros(self.n_embd))
            self.spatial_first = nn.Parameter(torch.zeros(self.n_embd))
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
            self.spatial_mix_v = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
        else:
            raise NotImplementedError

    def jit_func(self, x, patch_resolution):
        # Mix x with the previous timestep to produce xk, xv, xr
        B, T, C = x.size()
        if self.shift_pixel > 0:
            xx = self.shift_func(x, self.shift_pixel, self.channel_gamma, patch_resolution)
            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xv = x * self.spatial_mix_v + xx * (1 - self.spatial_mix_v)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
        else:
            xk = x
            xv = x
            xr = x

        # Use xk, xv, xr to produce k, v, r
        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        sr = torch.sigmoid(r)

        return sr, k, v

    def forward(self, x, patch_resolution=None):
        def _inner_forward(x):
            B, T, C = x.size()
            self.device = x.device

            sr, k, v = self.jit_func(x, patch_resolution)
            x = RUN_CUDA(self.spatial_decay / T, self.spatial_first / T, k, v)
            if self.key_norm is not None:
                x = self.key_norm(x)
            x = sr * x
            x = self.output(x)
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


class VRWKV_ChannelMix(BaseModule):
    def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                 channel_gamma=1 / 4, shift_pixel=1, hidden_rate=4, init_mode='fancy',
                 key_norm=False, with_cp=False):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.with_cp = with_cp
        self._init_weights(init_mode)
        self.shift_pixel = shift_pixel
        self.shift_mode = shift_mode
        if shift_pixel > 0:
            self.shift_func = eval(shift_mode)
            self.channel_gamma = channel_gamma
        else:
            self.spatial_mix_k = None
            self.spatial_mix_r = None

        hidden_sz = hidden_rate * n_embd
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(hidden_sz)
        else:
            self.key_norm = None
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)

        self.value.scale_init = 0
        self.receptance.scale_init = 0

    def _init_weights(self, init_mode):
        if init_mode == 'fancy':
            with torch.no_grad():  # fancy init of time_mix
                ratio_1_to_almost0 = (1.0 - (self.layer_id / self.n_layer))  # 1 to ~0
                x = torch.ones(1, 1, self.n_embd)
                for i in range(self.n_embd):
                    x[0, 0, i] = i / self.n_embd
                self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
                self.spatial_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
        elif init_mode == 'local':
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]))
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]))
        elif init_mode == 'global':
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
        else:
            raise NotImplementedError

    def forward(self, x, patch_resolution=None):
        def _inner_forward(x):
            if self.shift_pixel > 0:
                xx = self.shift_func(x, self.shift_pixel, self.channel_gamma, patch_resolution)
                xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
                xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
            else:
                xk = x
                xr = x

            k = self.key(xk)
            k = torch.square(torch.relu(k))
            if self.key_norm is not None:
                k = self.key_norm(k)
            kv = self.value(k)
            x = torch.sigmoid(self.receptance(xr)) * kv
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


class BiRWKV_TemporalMix(BaseModule):
    """
    [最终修正版] 时间混合模块 (替代自注意力)。
    - 实现了有界指数，提升训练稳定性。
    - 恢复使用独立的 K, V, R 投影层以保证逻辑正确。
    - 接受 'x_prev' 作为参数，避免重复计算。
    """

    def __init__(self, n_embd, n_layer, layer_id, init_mode='fancy'):
        super().__init__()
        # ... (init 部分与您之前的代码相同，无需改动)
        self.n_embd = n_embd
        self.layer_id = layer_id
        self.n_layer = n_layer

        with torch.no_grad():
            # (您的 'fancy' 初始化逻辑保持不变)
            ratio_0_to_1 = layer_id / (n_layer - 1) if n_layer > 1 else 0
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)

            decay_speed = torch.ones(n_embd)
            for h in range(n_embd):
                decay_speed[h] = -5 + 8 * (h / (n_embd - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed)

            zigzag = torch.tensor([(i + 1) % 3 - 1 for i in range(n_embd)]) * 0.5
            self.time_first = nn.Parameter(torch.ones(n_embd) * math.log(0.3) + zigzag)

            x = torch.ones(1, 1, n_embd)
            for i in range(n_embd):
                x[0, 0, i] = i / n_embd
            self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))

        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.output = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x, x_prev):
        # --- 新增: 从输入 x 获取序列长度 T ---
        B, T, C = x.size()

        # 使用学习到的系数混合当前和过去的 token
        xk = torch.lerp(x_prev, x, self.time_mix_k)
        xv = torch.lerp(x_prev, x, self.time_mix_v)
        xr = torch.lerp(x_prev, x, self.time_mix_r)

        # 分别计算 k, v, r
        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)

        sr = torch.sigmoid(r)

        # --- 修正点: 调用 RUN_CUDA 时进行除法操作 ---
        wkv_out = RUN_CUDA(self.time_decay / T, self.time_first / T, k, v)

        return self.output(sr * wkv_out)


class BiRWKV_ChannelMix(BaseModule):
    """
    [已优化] 通道混合模块 (替代 FFN)。
    - 接受 'x_prev' 作为参数，避免重复计算。
    """

    def __init__(self, n_embd, n_layer, layer_id, hidden_rate=4, init_mode='fancy'):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)
            x = torch.ones(1, 1, n_embd)
            for i in range(n_embd):
                x[0, 0, i] = i / n_embd
            self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))

        hidden_sz = int(hidden_rate * n_embd)
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)

    def forward(self, x, x_prev):
        # 优化点 1: 'x_prev' 作为参数传入。
        xk = torch.lerp(x_prev, x, self.time_mix_k)
        xr = torch.lerp(x_prev, x, self.time_mix_r)

        k = self.key(xk)
        k = torch.square(torch.relu(k))
        kv = self.value(k)

        return torch.sigmoid(self.receptance(xr)) * kv


class Block(BaseModule):
    def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                 channel_gamma=1 / 4, shift_pixel=1, drop_path=0., hidden_rate=4,
                 init_mode='fancy', init_values=None, post_norm=False, key_norm=False,
                 with_cp=False):
        super().__init__()
        self.layer_id = layer_id
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(n_embd)

        self.att = VRWKV_SpatialMix(n_embd, n_layer, layer_id, shift_mode,
                                    channel_gamma, shift_pixel, init_mode,
                                    key_norm=key_norm)

        self.ffn = VRWKV_ChannelMix(n_embd, n_layer, layer_id, shift_mode,
                                    channel_gamma, shift_pixel, hidden_rate,
                                    init_mode, key_norm=key_norm)
        self.layer_scale = (init_values is not None)
        self.post_norm = post_norm
        if self.layer_scale:
            self.gamma1 = nn.Parameter(init_values * torch.ones((n_embd)), requires_grad=True)
            self.gamma2 = nn.Parameter(init_values * torch.ones((n_embd)), requires_grad=True)
        self.with_cp = with_cp

    def forward(self, x, patch_resolution=None):
        def _inner_forward(x):
            if self.layer_id == 0:
                x = self.ln0(x)
            if self.post_norm:
                if self.layer_scale:
                    x = x + self.drop_path(self.gamma1 * self.ln1(self.att(x, patch_resolution)))
                    x = x + self.drop_path(self.gamma2 * self.ln2(self.ffn(x, patch_resolution)))
                else:
                    x = x + self.drop_path(self.ln1(self.att(x, patch_resolution)))
                    x = x + self.drop_path(self.ln2(self.ffn(x, patch_resolution)))
            else:
                if self.layer_scale:
                    x = x + self.drop_path(self.gamma1 * self.att(self.ln1(x), patch_resolution))
                    x = x + self.drop_path(self.gamma2 * self.ffn(self.ln2(x), patch_resolution))
                else:
                    x = x + self.drop_path(self.att(self.ln1(x), patch_resolution))
                    x = x + self.drop_path(self.ffn(self.ln2(x), patch_resolution))
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


class Block_time(BaseModule):
    """
    [已优化] 组合了时间混合和通道混合的主模块。
    - 每个 Block 内的子模块共享计算好的 token_shift 结果。
    """

    def __init__(self, n_embd, n_layer, layer_id, drop_path=0., hidden_rate=4,
                 init_mode='fancy', init_values=None, post_norm=False, with_cp=False, **kwargs):
        super().__init__()
        self.layer_id = layer_id
        self.post_norm = post_norm

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.att = BiRWKV_TemporalMix(n_embd, n_layer, layer_id, init_mode)
        self.ffn = BiRWKV_ChannelMix(n_embd, n_layer, layer_id, hidden_rate, init_mode)

        self.layer_scale = (init_values is not None)
        if self.layer_scale:
            self.gamma1 = nn.Parameter(init_values * torch.ones(n_embd), requires_grad=True)
            self.gamma2 = nn.Parameter(init_values * torch.ones(n_embd), requires_grad=True)
        self.with_cp = with_cp

    def forward(self, x):
        """
        前向传播函数。

        Args:
            x (torch.Tensor): 输入张量，形状为 [批大小, 序列长度, 特征维度]

        Returns:
            torch.Tensor: 输出张量，形状与输入相同。
        """

        def _inner_forward(x):
            # --- Pre-LN (前置归一化) 结构 ---
            if not self.post_norm:
                # 第一个子层: Temporal Mix (Attention)
                residual = x
                x_normalized = self.ln1(x)
                x_prev = token_shift(x_normalized)  # 对归一化后的x进行shift
                att_out = self.att(x_normalized, x_prev)

                if self.layer_scale:
                    att_out = self.gamma1 * att_out
                x = residual + self.drop_path(att_out)

                # 第二个子层: Channel Mix (FFN)
                residual = x
                x_normalized = self.ln2(x)
                x_prev = token_shift(x_normalized)  # 对归一化后的x进行shift
                ffn_out = self.ffn(x_normalized, x_prev)

                if self.layer_scale:
                    ffn_out = self.gamma2 * ffn_out
                x = residual + self.drop_path(ffn_out)

            # --- Post-LN (后置归一化) 结构 ---
            else:
                residual = x
                x_prev = token_shift(x)  # 对原始x进行shift
                att_out = self.att(x, x_prev)
                if self.layer_scale:
                    att_out = self.gamma1 * att_out
                x = self.ln1(residual + self.drop_path(att_out))  # 残差连接后进行LayerNorm

                residual = x
                x_prev = token_shift(x)  # 对新的x进行shift
                ffn_out = self.ffn(x, x_prev)
                if self.layer_scale:
                    ffn_out = self.gamma2 * ffn_out
                x = self.ln2(residual + self.drop_path(ffn_out))  # 残差连接后进行LayerNorm

            return x

        if self.with_cp and x.requires_grad:
            return cp.checkpoint(_inner_forward, x)
        else:
            return _inner_forward(x)