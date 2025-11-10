import torch
import torch.nn as nn
from omegaconf import DictConfig

from models.base import Model
from models.modules import PositionalEncoding, TimestepEmbedder
from models.modules import SceneMapEncoderDecoder, SceneMapEncoder
from models.functions import load_and_freeze_clip_model, encode_text_clip, \
    load_and_freeze_bert_model, encode_text_bert, get_lang_feat_dim_type
from utils.misc import compute_repr_dimesion
from models.mamba_trans import *
from models.vrwkv import *


@Model.register()
class CMDM(nn.Module):

    def __init__(self, cfg: DictConfig, *args, **kwargs):
        super().__init__()
        self.device = kwargs['device'] if 'device' in kwargs else 'cpu'
        
        self.motion_type = cfg.data_repr
        self.motion_dim = cfg.input_feats
        self.latent_dim = cfg.latent_dim
        self.mask_motion = cfg.mask_motion
        
        self.arch = cfg.arch

        ## time embedding
        self.time_emb_dim = cfg.time_emb_dim
        self.timestep_embedder = TimestepEmbedder(self.latent_dim, self.time_emb_dim, max_len=1000)

        ## contact
        self.contact_type = cfg.contact_model.contact_type
        self.contact_dim = compute_repr_dimesion(self.contact_type)
        self.planes = cfg.contact_model.planes
        if self.arch == 'trans_enc':
            SceneMapModule = SceneMapEncoder
            self.contact_adapter = nn.Linear(self.planes[-1], self.latent_dim, bias=True)
        elif self.arch == 'trans_rwkv':
            SceneMapModule = SceneMapEncoder
            self.contact_adapter = nn.Linear(self.planes[-1], self.latent_dim, bias=True)
        elif self.arch == 'trans_dec':
            SceneMapModule = SceneMapEncoderDecoder
        elif self.arch == 'trans_mamba':
            SceneMapModule = SceneMapEncoder
            self.contact_adapter = nn.Linear(self.planes[-1], self.latent_dim, bias=True)
        else:
            raise NotImplementedError
        self.contact_encoder = SceneMapModule(
            point_feat_dim=self.contact_dim,
            planes=self.planes,
            blocks=cfg.contact_model.blocks,
            num_points=cfg.contact_model.num_points,
        )
        
        ## text
        self.text_model_name = cfg.text_model.version
        self.text_max_length = cfg.text_model.max_length
        self.text_feat_dim, self.text_feat_type = get_lang_feat_dim_type(self.text_model_name)
        if self.text_feat_type == 'clip':
            self.text_model = load_and_freeze_clip_model(self.text_model_name)
        elif self.text_feat_type == 'bert':
            self.tokenizer, self.text_model = load_and_freeze_bert_model(self.text_model_name)
        else:
            raise NotImplementedError
        self.language_adapter = nn.Linear(self.text_feat_dim, self.latent_dim, bias=True)

        ## model architecture
        self.motion_adapter = nn.Linear(self.motion_dim, self.latent_dim, bias=True)
        self.positional_encoder = PositionalEncoding(self.latent_dim, dropout=0.1, max_len=5000)

        self.num_layers = cfg.num_layers
        if self.arch == 'trans_enc':
            self.self_attn_layer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.latent_dim,
                    nhead=cfg.num_heads,
                    dim_feedforward=cfg.dim_feedforward,
                    dropout=cfg.dropout,
                    activation='gelu',
                    batch_first=True,
                ),
                enable_nested_tensor=False,
                num_layers=sum(cfg.num_layers),
            )
        elif self.arch == 'trans_rwkv':
            rwkv_blocks = []
            for i in range(sum(cfg.num_layers)):
                rwkv_blocks.append(
                    Block_time(
                        n_embd=self.latent_dim,
                        n_layer=sum(cfg.num_layers),  # RWKV 'fancy init' 需要总层数
                        layer_id=i,           # 当前层的 ID (从 0 到 total_depth-1)
                        hidden_rate=cfg.dim_feedforward // self.latent_dim,
                        drop_path=cfg.dropout, # 复用 cfg 中的 dropout 作为 drop_path
                        # init_values, post_norm, key_norm 等高级参数
                        # 也可以在这里从 cfg 传入
                    )

                )
            # 4 将所有块打包成一个 nn.Sequential 模块
            self.self_attn_layer = nn.Sequential(*rwkv_blocks)

            # self.self_attn_layer = nn.Sequential(
            #         *[Block_time(
            #             n_embd=self.latent_dim,
            #             n_layer=sum(self.num_layers), # 总层数，用于fancy init
            #             layer_id=sum(self.num_layers[:i]) + layer_idx, # 当前块的全局ID
            #             hidden_rate=4, # FFN的隐藏层倍率，可设为超参数
            #             # drop_path, init_values 等参数也可以根据需要添加
            #         ) for layer_idx in range(n)]
            #     )
        elif self.arch == 'trans_mamba':
            self.self_attn_layer = MambaTransBackbone(
                    num_layers=sum(cfg.num_layers),
                    latent_dim=self.latent_dim,
                    num_heads=cfg.num_heads,
                    ff_size=cfg.dim_feedforward,
                    dropout=0.1,
                    drop_path_rate=0.1
                )
        elif self.arch == 'trans_dec':
            self.self_attn_layers = nn.ModuleList()
            self.kv_mappling_layers = nn.ModuleList()
            self.cross_attn_layers = nn.ModuleList()
            for i, n in enumerate(self.num_layers):
                self.self_attn_layers.append(
                    nn.TransformerEncoder(
                        nn.TransformerEncoderLayer(
                            d_model=self.latent_dim,
                            nhead=cfg.num_heads,
                            dim_feedforward=cfg.dim_feedforward,
                            dropout=cfg.dropout,
                            activation='gelu',
                            batch_first=True,
                        ),
                        num_layers=n,
                    )
                )

                if i != len(self.num_layers) - 1:
                    self.kv_mappling_layers.append(
                        nn.Sequential(
                            nn.Linear(self.planes[-1-i], self.latent_dim, bias=True),
                            nn.LayerNorm(self.latent_dim),
                        )
                    )
                    self.cross_attn_layers.append(
                        nn.TransformerDecoderLayer(
                            d_model=self.latent_dim,
                            nhead=cfg.num_heads,
                            dim_feedforward=cfg.dim_feedforward,
                            dropout=cfg.dropout,
                            activation='gelu',
                            batch_first=True,
                        )
                    )
        else:
            raise NotImplementedError
        self.motion_layer = nn.Linear(self.latent_dim, self.motion_dim, bias=True)

    def forward(self, x, timesteps, **kwargs):
        """ Forward pass of the model.

        Args:
            x: input motion, [bs, seq_len, motion_dim]
            kwargs: other inputs, e.g., contact, text
        
        Return:
            Output motion, [bs, seq_len, motion_dim]
        """
        ## time embedding
        time_emb = self.timestep_embedder(timesteps) # [bs, 1, latent_dim]
        time_mask = torch.zeros((x.shape[0], 1), dtype=torch.bool, device=self.device)

        ## text embedding
        if self.text_feat_type == 'clip':
            text_emb = encode_text_clip(self.text_model, kwargs['c_text'], max_length=self.text_max_length, device=self.device)
            text_emb = text_emb.unsqueeze(1).detach().float()
            text_mask = torch.zeros((x.shape[0], 1), dtype=torch.bool, device=self.device)
        elif self.text_feat_type == 'bert':
            text_emb, text_mask = encode_text_bert(self.tokenizer, self.text_model, kwargs['c_text'], max_length=self.text_max_length, device=self.device)
            text_mask = ~(text_mask.to(torch.bool)) # 0 for valid, 1 for invalid
        else:
            raise NotImplementedError
        if 'c_text_mask' in kwargs:
            text_mask = torch.logical_or(text_mask, kwargs['c_text_mask'].repeat(1, text_mask.shape[1]))
        if 'c_text_erase' in kwargs:
            text_emb = text_emb * (1. - kwargs['c_text_erase'].unsqueeze(-1).float())
        text_emb = self.language_adapter(text_emb) # [bs, 1, latent_dim]

        ## encode contact
        cont_emb = self.contact_encoder(kwargs['c_pc_xyz'], kwargs['c_pc_contact'])
        if hasattr(self, 'contact_adapter'): # trans_enc
            cont_mask = torch.zeros((x.shape[0], cont_emb.shape[1]), dtype=torch.bool, device=self.device)
            if 'c_pc_mask' in kwargs:
                cont_mask = torch.logical_or(cont_mask, kwargs['c_pc_mask'].repeat(1, cont_mask.shape[1]))
            if 'c_pc_erase' in kwargs:
                cont_emb = cont_emb * (1. - kwargs['c_pc_erase'].unsqueeze(-1).float())
            cont_emb = self.contact_adapter(cont_emb) # [bs, num_groups, latent_dim], for trans_enc

        ## motion embedding
        x = self.motion_adapter(x) # [bs, seq_len, latent_dim]
        if self.arch == 'trans_enc':
            x = torch.cat([time_emb, text_emb, cont_emb, x], dim=1) # [bs, 2 + num_groups + seq_len, latent_dim]
            x = self.positional_encoder(x.permute(1, 0, 2)).permute(1, 0, 2)

            x_mask = None
            if self.mask_motion:
                x_mask = torch.cat([time_mask, text_mask, cont_mask, kwargs['x_mask']], dim=1) # [bs, 2 + num_groups + seq_len]
            x = self.self_attn_layer(x, src_key_padding_mask=x_mask)

            non_motion_token = time_mask.shape[1] + text_mask.shape[1] + cont_mask.shape[1]
            x = x[:, non_motion_token:, :]
        if self.arch == 'trans_mamba':
            # 1. 将所有序列拼接成一个长序列：[时间, 文本, 接触, 运动]
            x = torch.cat([time_emb, text_emb, cont_emb, x],
                          dim=1)  # [bs, total_len, latent_dim] (B, L, C)

            # 2. 准备 MambaTransBackbone 所需的 (L, B, C) 格式
            #    我们假设 self.positional_encoder 期望 L, B, C 并返回 L, B, C
            x_permuted = x.permute(1, 0, 2)
            x_with_pos = self.positional_encoder(x_permuted)  # [total_len, bs, latent_dim] (L, B, C)

            # 3. 准备掩码
            x_mask = None
            if self.mask_motion:
                # 拼接对应的掩码
                x_mask = torch.cat([time_mask, text_mask, cont_mask, kwargs['x_mask']], dim=1)  # (B, L)

            # 4. 通过 MambaTransBackbone
            # 它的 forward 接受 (L, B, C) 输入和 (B, L) 掩码，并返回 (L, B, C)
            x_processed = self.self_attn_layer(x_with_pos, src_key_padding_mask=x_mask)

            # 5. 转换回 (B, L, C) 格式，以便进行后续切片
            x = x_processed.permute(1, 0, 2)  # [bs, total_len, latent_dim] (B, L, C)

            # 6. 从输出序列中提取出属于运动的部分
            non_motion_token = time_mask.shape[1] + text_mask.shape[1] + cont_mask.shape[1]
            x = x[:, non_motion_token:, :]

        elif self.arch == 'trans_dec':
            x = torch.cat([time_emb, text_emb, x], dim=1) # [bs, 2 + seq_len, latent_dim]
            x = self.positional_encoder(x.permute(1, 0, 2)).permute(1, 0, 2)

            x_mask = None
            if self.mask_motion:
                x_mask = torch.cat([time_mask, text_mask, kwargs['x_mask']], dim=1) # [bs, 2 + seq_len]
            for i in range(len(self.num_layers)):
                x = self.self_attn_layers[i](x, src_key_padding_mask=x_mask) # self attention
                if i != len(self.num_layers) - 1: # cross attention
                    mem = cont_emb[i]
                    mem_mask = torch.zeros((x.shape[0], mem.shape[1]), dtype=torch.bool, device=self.device)
                    if 'c_pc_mask' in kwargs:
                        mem_mask = torch.logical_or(mem_mask, kwargs['c_pc_mask'].repeat(1, mem_mask.shape[1]))
                    if 'c_pc_erase' in kwargs:
                        mem = mem * (1. - kwargs['c_pc_erase'].unsqueeze(-1).float())
                    mem = self.kv_mappling_layers[i](mem)
                    x = self.cross_attn_layers[i](x, mem, tgt_key_padding_mask=x_mask, memory_key_padding_mask=mem_mask)

            non_motion_token = time_mask.shape[1] + text_mask.shape[1]
            x = x[:, non_motion_token:, :]
        elif self.arch == 'trans_rwkv':
            # 1. 拼接所有序列：[时间, 文本, 接触, 运动]
            # (与 trans_enc 完全相同)
            x = torch.cat([time_emb, text_emb, cont_emb, x],
                          dim=1)  # 形状: [bs, total_len, latent_dim]

            # 2. (关键区别) RWKV 不需要位置编码
            # 您的 Block_time 及其内部的 token_shift 和 BiRWKV_TemporalMix
            # 已经隐式地处理了时序/位置信息，因此 self.positional_encoder 是多余的。
            # x = self.positional_encoder(x.permute(1, 0, 2)).permute(1, 0, 2) # <-- 不需要这一行

            # 3. 准备掩码 (与 trans_enc 完全相同)
            x_mask = None
            if self.mask_motion:
                # x_mask 形状: [bs, total_len], True/1 表示该位置是 padding
                x_mask = torch.cat([time_mask, text_mask, cont_mask, kwargs['x_mask']], dim=1)

            # 4. (关键区别) 手动应用掩码
            # RWKV (nn.Sequential) 不接受 src_key_padding_mask
            # 我们必须在输入 RWKV 之前，手动将所有 padding 的 token 置零
            if x_mask is not None:
                # x_mask 形状 [B, L], x 形状 [B, L, C]
                # 我们需要将 x_mask 扩展到 [B, L, 1] 以便广播
                x_mask_expanded = x_mask.unsqueeze(-1) # 形状: [bs, total_len, 1]
                # masked_fill_ 会在 x_mask_expanded 为 True 的地方填充 0.0
                x = x.masked_fill(x_mask_expanded, 0.0)

            # 5. 通过 RWKV 堆叠层
            # self.self_attn_layer 是一个 nn.Sequential(Block_time, ...)
            # Block_time.forward(x) 接受 (B, L, C) 并返回 (B, L, C)
            x = self.self_attn_layer(x)

            # 6. (可选但推荐) 再次应用掩码
            # 确保 padding 位置的输出特征也是零，不会影响后续的 motion_layer
            if x_mask is not None:
                x = x.masked_fill(x_mask_expanded, 0.0)

            # 7. 提取运动序列 (与 trans_enc 完全相同)
            non_motion_token = time_mask.shape[1] + text_mask.shape[1] + cont_mask.shape[1]
            x = x[:, non_motion_token:, :]
        else:
            raise NotImplementedError

        x = self.motion_layer(x)
        return x
