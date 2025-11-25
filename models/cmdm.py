import torch
import torch.nn as nn
from omegaconf import DictConfig

from models.base import Model
from models.modules import PositionalEncoding, TimestepEmbedder
from models.modules import SceneMapEncoderDecoder, SceneMapEncoder
from models.functions import load_and_freeze_clip_model, encode_text_clip, \
    load_and_freeze_bert_model, encode_text_bert, get_lang_feat_dim_type
from models.try_models.mamba_block import MambaBlock, BidirectionalMambaBlock
from utils.misc import compute_repr_dimesion

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

        # 冻结参数相关配置
        self.freeze_transformer_layers = getattr(cfg, 'freeze_transformer_layers', False)
        self.frozen_layers = getattr(cfg, 'frozen_layers', [])  # 要冻结的层索引列表

        ## time embedding
        self.time_emb_dim = cfg.time_emb_dim
        self.timestep_embedder = TimestepEmbedder(self.latent_dim, self.time_emb_dim, max_len=1000)

        ## contact
        self.contact_type = cfg.contact_model.contact_type
        self.contact_dim = compute_repr_dimesion(self.contact_type)
        self.planes = cfg.contact_model.planes
        if self.arch in ['trans_enc', 'trans_mamba', 'trans_double_mamba']:
            SceneMapModule = SceneMapEncoder
            self.contact_adapter = nn.Linear(self.planes[-1], self.latent_dim, bias=True)
        elif self.arch == 'trans_dec':
            SceneMapModule = SceneMapEncoderDecoder
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
        elif self.arch == 'trans_mamba':
            total_layers = sum(cfg.num_layers)
            mamba_layers = getattr(cfg, 'mamba_layers', 1)
            if mamba_layers > total_layers:
                raise ValueError("cfg.mamba_layers 不能大于总层数")
            mlp_ratio = cfg.dim_feedforward / float(self.latent_dim)
            self.encoder_layers = nn.ModuleList()
            for idx in range(total_layers):
                if idx < total_layers - mamba_layers:
                    layer = nn.TransformerEncoderLayer(
                        d_model=self.latent_dim,
                        nhead=cfg.num_heads,
                        dim_feedforward=cfg.dim_feedforward,
                        dropout=cfg.dropout,
                        activation='gelu',
                        batch_first=True,
                    )
                else:
                    layer = MambaBlock(
                        d_model=self.latent_dim,
                        d_state=getattr(cfg, 'mamba_d_state', 16),
                        d_conv=getattr(cfg, 'mamba_d_conv', 4),
                        expand=getattr(cfg, 'mamba_expand', 2),
                        mlp_ratio=mlp_ratio,
                        drop=cfg.dropout,
                        drop_path=getattr(cfg, 'mamba_drop_path', 0.0),
                    )
                self.encoder_layers.append(layer)
        elif self.arch == 'trans_double_mamba':
            # 双向 Mamba 架构：前面用 Transformer，后面用双向 Mamba
            total_layers = sum(cfg.num_layers)
            mamba_layers = getattr(cfg, 'mamba_layers', 2)  # 默认最后 2 层用双向 Mamba
            if mamba_layers > total_layers:
                raise ValueError("cfg.mamba_layers 不能大于总层数")
            mlp_ratio = cfg.dim_feedforward / float(self.latent_dim)
            self.encoder_layers = nn.ModuleList()
            for idx in range(total_layers):
                if idx < total_layers - mamba_layers:
                    # 前面的层使用标准 Transformer
                    layer = nn.TransformerEncoderLayer(
                        d_model=self.latent_dim,
                        nhead=cfg.num_heads,
                        dim_feedforward=cfg.dim_feedforward,
                        dropout=cfg.dropout,
                        activation='gelu',
                        batch_first=True,
                    )
                else:
                    # 后面的层使用双向 Mamba
                    layer = BidirectionalMambaBlock(
                        d_model=self.latent_dim,
                        d_state=getattr(cfg, 'mamba_d_state', 16),
                        d_conv=getattr(cfg, 'mamba_d_conv', 4),
                        expand=getattr(cfg, 'mamba_expand', 2),
                        mlp_ratio=mlp_ratio,
                        drop=cfg.dropout,
                        drop_path=getattr(cfg, 'mamba_drop_path', 0.0),
                    )
                self.encoder_layers.append(layer)
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

        # 冻结指定的层
        if self.freeze_transformer_layers and hasattr(self, 'encoder_layers'):
            self._freeze_layers()

    def _freeze_layers(self):
        """冻结指定的层参数"""
        if not hasattr(self, 'encoder_layers'):
            return

        for idx, layer in enumerate(self.encoder_layers):
            if idx in self.frozen_layers:
                for param in layer.parameters():
                    param.requires_grad = False
                print(f"Frozen layer {idx}: {type(layer).__name__}")

    def load_pretrained_weights(self, path: str):
        """加载预训练权重，支持部分参数加载（跳过冻结的参数）"""
        import torch
        assert torch.cuda.is_available() or torch.backends.mps.is_available(), 'CUDA or MPS is required for loading pretrained weights.'

        saved_state_dict = torch.load(path, map_location='cpu')
        model_state_dict = self.state_dict()

        unchanged_weights = []
        used_weights = []
        skipped_frozen_weights = []

        for key in model_state_dict:
            # 检查参数是否属于冻结的层
            is_frozen_param = False
            if hasattr(self, 'encoder_layers') and self.freeze_transformer_layers:
                for idx, layer in enumerate(self.encoder_layers):
                    if idx in self.frozen_layers:
                        layer_prefix = f'encoder_layers.{idx}'
                        if key.startswith(layer_prefix):
                            is_frozen_param = True
                            break

            if is_frozen_param:
                skipped_frozen_weights.append(key)
                continue

            ## current state and saved state both on single GPU or both on multi GPUs
            if key in saved_state_dict:
                model_state_dict[key] = saved_state_dict[key]
                used_weights.append(key)

            ## current state on single GPU and saved state on multi GPUs
            elif 'module.'+key in saved_state_dict:
                model_state_dict[key] = saved_state_dict['module.'+key]
                used_weights.append('module.'+key)

            else:
                unchanged_weights.append(key)

        unused_weights = []
        for key in saved_state_dict:
            if key not in used_weights and 'module.'+key not in used_weights:
                unused_weights.append(key)

        print(f"Loaded {len(used_weights)} parameters from pretrained weights")
        print(f"Skipped {len(skipped_frozen_weights)} frozen parameters")
        print(f"Unchanged {len(unchanged_weights)} parameters")
        print(f"Unused {len(unused_weights)} parameters from checkpoint")

        self.load_state_dict(model_state_dict)

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
        elif self.arch == 'trans_mamba':
            x = torch.cat([time_emb, text_emb, cont_emb, x], dim=1)
            x = self.positional_encoder(x.permute(1, 0, 2)).permute(1, 0, 2)

            x_mask = None
            if self.mask_motion:
                x_mask = torch.cat([time_mask, text_mask, cont_mask, kwargs['x_mask']], dim=1)
            for layer in self.encoder_layers:
                if isinstance(layer, nn.TransformerEncoderLayer):
                    x = layer(x, src_key_padding_mask=x_mask)
                else:
                    x = layer(x, padding_mask=x_mask)

            non_motion_token = time_mask.shape[1] + text_mask.shape[1] + cont_mask.shape[1]
            x = x[:, non_motion_token:, :]
        elif self.arch == 'trans_double_mamba':
            # 与 trans_mamba 类似的处理流程，但使用双向 Mamba
            x = torch.cat([time_emb, text_emb, cont_emb, x], dim=1)
            x = self.positional_encoder(x.permute(1, 0, 2)).permute(1, 0, 2)

            x_mask = None
            if self.mask_motion:
                x_mask = torch.cat([time_mask, text_mask, cont_mask, kwargs['x_mask']], dim=1)

            for layer in self.encoder_layers:
                if isinstance(layer, nn.TransformerEncoderLayer):
                    x = layer(x, src_key_padding_mask=x_mask)
                else:  # BidirectionalMambaBlock
                    x = layer(x, padding_mask=x_mask)

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
        else:
            raise NotImplementedError

        x = self.motion_layer(x)
        return x

