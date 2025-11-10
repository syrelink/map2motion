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


# 使用 @Model.register() 装饰器，这是一种常见的设计模式，用于将该模型类注册到一个全局的模型注册表中，
# 这样就可以通过配置文件中的字符串名称来方便地实例化该模型。
@Model.register()
class CMDM(nn.Module):

    def __init__(self, cfg: DictConfig, *args, **kwargs):

        super().__init__()
        # 获取设备信息，默认为 'cpu'
        self.device = kwargs['device'] if 'device' in kwargs else 'cpu'

        # --- 运动数据相关配置 ---
        self.motion_type = cfg.data_repr  # 运动数据的表示方式 (e.g., 'humanml')
        self.motion_dim = cfg.input_feats  # 输入运动特征的维度
        self.latent_dim = cfg.latent_dim  # 模型内部的隐状态维度
        self.mask_motion = cfg.mask_motion  # 是否对运动序列的填充部分进行掩码

        # --- 核心架构选择 ---
        self.arch = cfg.arch  # 'trans_enc' 或 'trans_dec'

        # --- 时间步长嵌入 (Timestep Embedding) ---
        # 扩散模型需要将当前的时间步 t 编码成一个向量，作为模型的条件输入
        self.time_emb_dim = cfg.time_emb_dim
        self.timestep_embedder = TimestepEmbedder(self.latent_dim, self.time_emb_dim, max_len=1000)

        # --- 场景接触信息编码器 (Contact Encoder) ---
        self.contact_type = cfg.contact_model.contact_type
        self.contact_dim = compute_repr_dimesion(self.contact_type)  # 计算接触特征的维度
        self.planes = cfg.contact_model.planes  # PointNet++ 中各层的输出维度

        # 根据选择的架构，决定使用哪种场景编码器

        if self.arch == 'trans_enc':
            # Transformer Encoder 架构：场景编码器只输出最终的特征
            SceneMapModule = SceneMapEncoder
            # 添加一个线性层，将场景特征的维度映射到模型的隐状态维度
            self.contact_adapter = nn.Linear(self.planes[-1], self.latent_dim, bias=True)
        elif self.arch == 'trans_mamba':
            SceneMapModule = SceneMapEncoder
            self.contact_adapter = nn.Linear(self.planes[-1], self.latent_dim, bias=True)
        elif self.arch == 'trans_rwkv':
            SceneMapModule = SceneMapEncoder
            self.contact_adapter = nn.Linear(self.planes[-1], self.latent_dim, bias=True)
        elif self.arch == 'trans_dec':
            # Transformer Decoder 架构：场景编码器输出多层次的特征，用于交叉注意力
            SceneMapModule = SceneMapEncoderDecoder
        elif self.arch == 'trans_mambaTrans':
            # Transformer Decoder 架构：场景编码器输出多层次的特征，用于交叉注意力
            SceneMapModule = SceneMapEncoderDecoder
        else:
            raise NotImplementedError(f"不支持的架构: {self.arch}")

        # 实例化场景编码器 (基于 PointNet++)
        self.contact_encoder = SceneMapModule(
            point_feat_dim=self.contact_dim,
            planes=self.planes,
            blocks=cfg.contact_model.blocks,
            num_points=cfg.contact_model.num_points,
        )

        # --- 文本编码器 (Text Encoder) ---
        self.text_model_name = cfg.text_model.version  # 文本模型的名称 (e.g., 'clip-ViT-B/32' 或 'bert-base-uncased')
        self.text_max_length = cfg.text_model.max_length  # 文本序列的最大长度
        # 获取预训练语言模型的特征维度和类型 ('clip' 或 'bert')
        self.text_feat_dim, self.text_feat_type = get_lang_feat_dim_type(self.text_model_name)

        # 加载并冻结预训练的语言模型，我们只用它来提取特征，不进行训练
        if self.text_feat_type == 'clip':
            self.text_model = load_and_freeze_clip_model(self.text_model_name)
        elif self.text_feat_type == 'bert':
            self.tokenizer, self.text_model = load_and_freeze_bert_model(self.text_model_name)
        else:
            raise NotImplementedError

        # 添加一个线性层（适配器），将文本特征的维度映射到模型的隐状态维度
        self.language_adapter = nn.Linear(self.text_feat_dim, self.latent_dim, bias=True)

        # --- 核心 Transformer 架构 ---
        # 输入运动数据的适配器，将其维度映射到模型的隐状态维度
        self.motion_adapter = nn.Linear(self.motion_dim, self.latent_dim, bias=True)
        # 位置编码器，为序列中的每个 token 添加位置信息
        self.positional_encoder = PositionalEncoding(self.latent_dim, dropout=0.1, max_len=5000)

        self.num_layers = cfg.num_layers  # Transformer 层的数量
        if self.arch == 'trans_mamba':
            self.self_attn_layer = MambaTransBackbone(
                    num_layers=sum(cfg.num_layers),
                    latent_dim=self.latent_dim,
                    num_heads=cfg.num_heads,
                    ff_size=cfg.dim_feedforward,
                    dropout=0.1,
                    drop_path_rate=0.1
                )

        elif self.arch == 'trans_enc':
            # 对于 Encoder 架构，使用一个标准的 nn.TransformerEncoder
            # 所有输入（时间、文本、接触、运动）被拼接成一个长序列进行处理
            self.self_attn_layer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.latent_dim,
                    nhead=cfg.num_heads,
                    dim_feedforward=cfg.dim_feedforward,
                    dropout=cfg.dropout,
                    activation='gelu',
                    batch_first=True,  # 输入/输出张量的形状为 [batch, seq, feature]
                ),
                enable_nested_tensor=False,  # 推荐设置为 False 以支持 padding mask
                num_layers=sum(cfg.num_layers),
            )
        elif self.arch == 'trans_rwkv':
            # 对于 Encoder 架构，使用一个标准的 nn.TransformerEncoder
            # 所有输入（时间、文本、接触、运动）被拼接成一个长序列进行处理
            # self.self_attn_layer = nn.TransformerEncoder(
            #     nn.TransformerEncoderLayer(
            #         d_model=self.latent_dim,
            #         nhead=cfg.num_heads,
            #         dim_feedforward=cfg.dim_feedforward,
            #         dropout=cfg.dropout,
            #         activation='gelu',
            #         batch_first=True,  # 输入/输出张量的形状为 [batch, seq, feature]
            #     ),
            #     enable_nested_tensor=False,  # 推荐设置为 False 以支持 padding mask
            #     num_layers=sum(cfg.num_layers),
            # )
            self.self_attn_layer = nn.Sequential(
                    *[Block_time(
                        n_embd=self.latent_dim,
                        n_layer=sum(self.num_layers), # 总层数，用于fancy init
                        layer_id=1,
                        hidden_rate=4, # FFN的隐藏层倍率，可设为超参数
                        # drop_path, init_values 等参数也可以根据需要添加
                    ) ]
                )
        elif self.arch == 'trans_dec':
            # 对于 Decoder 架构，模型包含自注意力和交叉注意力层
            self.self_attn_layers = nn.ModuleList()
            self.kv_mappling_layers = nn.ModuleList()  # 将接触特征映射为 key 和 value
            self.cross_attn_layers = nn.ModuleList()
            # 这里的num_layers = [1,1,1,1,1]
            for i, n in enumerate(self.num_layers):
                # 1. 自注意力层 (Self-Attention)
                # 处理 [时间, 文本, 运动] 序列
                self.self_attn_layers.append(
                    nn.TransformerEncoder(
                        nn.TransformerEncoderLayer(
                            d_model=self.latent_dim, nhead=cfg.num_heads,
                            dim_feedforward=cfg.dim_feedforward, dropout=cfg.dropout,
                            activation='gelu', batch_first=True,
                        ),
                        num_layers=n,
                    )
                )

                # 2. 交叉注意力层 (Cross-Attention)
                # 在每个自注意力块之后（除了最后一个），插入交叉注意力块
                if i != len(self.num_layers) - 1:
                    # 接触特征适配器，用于生成交叉注意力的 key 和 value
                    self.kv_mappling_layers.append(
                        nn.Sequential(
                            nn.Linear(self.planes[-1 - i], self.latent_dim, bias=True),
                            nn.LayerNorm(self.latent_dim),
                        )
                    )
                    # 交叉注意力层，其中 query 来自运动序列，key/value 来自接触信息
                    self.cross_attn_layers.append(
                        nn.TransformerDecoderLayer(
                            d_model=self.latent_dim, nhead=cfg.num_heads,
                            dim_feedforward=cfg.dim_feedforward, dropout=cfg.dropout,
                            activation='gelu', batch_first=True,
                        )
                    )
        elif self.arch == 'trans_mambaTrans':
            # 1. 创建 *两个* 模块列表
            self.self_mamba_block = nn.ModuleList()  # 仅用于 i = 0
            self.self_attn_layers = nn.ModuleList()  # 用于 i > 0

            self.kv_mappling_layers = nn.ModuleList()
            self.cross_attn_layers = nn.ModuleList()

            # 这里的num_layers = [1,1,1,1,1]
            total_layers = len(self.num_layers)
            dpr = [x.item() for x in torch.linspace(0, 0.1, total_layers)]  # Drop path rates

            for i, n in enumerate(self.num_layers):

                # 2. 条件化创建模块
                if i == 0:
                    # --- 当 i == 0 时: 创建 Mamba 块 ---
                    # (这是我们之前讨论过的 ResidualHybridBlock)
                    self.self_mamba_block.append(
                        MambaTransBackbone(
                            num_layers=1,
                            latent_dim=self.latent_dim,
                            num_heads=4,
                            ff_size=1024,
                            dropout=0.1,
                            drop_path_rate=0.1
                        )
                    )
                else:
                    # --- 当 i > 0 时: 创建 Transformer 块 ---
                    # (这是您原始的 TransformerEncoder)
                    self.self_attn_layers.append(
                        nn.TransformerEncoder(
                            nn.TransformerEncoderLayer(
                                d_model=self.latent_dim, nhead=cfg.num_heads,
                                dim_feedforward=cfg.dim_feedforward, dropout=cfg.dropout,
                                activation='gelu', batch_first=True,
                            ),
                            num_layers=n,
                        )
                    )

                # 3. 交叉注意力层 (逻辑不变)
                # 在每个自注意力块之后（除了最后一个），插入交叉注意力块
                if i != len(self.num_layers) - 1:  # i = 0, 1, 2, 3
                    # 接触特征适配器
                    self.kv_mappling_layers.append(
                        nn.Sequential(
                            nn.Linear(self.planes[-1 - i], self.latent_dim, bias=True),
                            nn.LayerNorm(self.latent_dim),
                        )
                    )
                    # 交叉注意力层
                    self.cross_attn_layers.append(
                        nn.TransformerDecoderLayer(
                            d_model=self.latent_dim, nhead=cfg.num_heads,
                            dim_feedforward=cfg.dim_feedforward, dropout=cfg.dropout,
                            activation='gelu', batch_first=True,
                        )
                    )
        else:
            raise NotImplementedError

        # --- 输出层 ---
        # 将 Transformer 处理后的隐状态映射回原始的运动特征维度
        self.motion_layer = nn.Linear(self.latent_dim, self.motion_dim, bias=True)

    def forward(self, x, timesteps, **kwargs):
        """
        模型的前向传播函数。

        Args:
            x: 带噪声的输入运动序列，形状为 [bs, seq_len, motion_dim]
            timesteps: 当前的扩散时间步，形状为 [bs]
            kwargs: 其他条件输入，例如:
                - 'c_text': 文本描述列表
                - 'c_pc_xyz': 场景点云坐标 [bs, num_points, 3]
                - 'c_pc_contact': 场景点云接触标签 [bs, num_points, contact_dim]
                - 'x_mask': 运动序列的填充掩码 [bs, seq_len]
                - ... (其他用于 classifier-free guidance 的掩码)

        Return:
            预测的噪声（或去噪后的运动），形状为 [bs, seq_len, motion_dim]
        """
        # --- 1. 准备条件嵌入 ---

        # 时间嵌入
        time_emb = self.timestep_embedder(timesteps)  # [bs, 1, latent_dim]
        time_mask = torch.zeros((x.shape[0], 1), dtype=torch.bool, device=self.device)  # 时间 token 不需要掩码

        # 文本嵌入
        if self.text_feat_type == 'clip':
            text_emb = encode_text_clip(self.text_model, kwargs['c_text'], max_length=self.text_max_length,
                                        device=self.device)
            text_emb = text_emb.unsqueeze(1).detach().float()  # [bs, 1, text_feat_dim]
            text_mask = torch.zeros((x.shape[0], 1), dtype=torch.bool, device=self.device)  # CLIP 输出单个特征向量，不需要掩码
        elif self.text_feat_type == 'bert':
            text_emb, text_mask = encode_text_bert(self.tokenizer, self.text_model, kwargs['c_text'],
                                                   max_length=self.text_max_length, device=self.device)
            text_mask = ~(text_mask.to(torch.bool))  # BERT 的 attention_mask 是 1 表示有效，Transformer 需要 1 表示无效，所以取反
        else:
            raise NotImplementedError

        # (可选) 应用额外的文本掩码，用于 classifier-free guidance
        if 'c_text_mask' in kwargs:
            text_mask = torch.logical_or(text_mask, kwargs['c_text_mask'].repeat(1, text_mask.shape[1]))
        if 'c_text_erase' in kwargs:  # 将某些样本的文本条件置零
            text_emb = text_emb * (1. - kwargs['c_text_erase'].unsqueeze(-1).float())

        text_emb = self.language_adapter(text_emb)  # [bs, text_seq_len, latent_dim]

        # 场景接触信息嵌入
        cont_emb = self.contact_encoder(kwargs['c_pc_xyz'], kwargs['c_pc_contact'])
        if hasattr(self, 'contact_adapter'):  # 仅 trans_enc 架构有此适配器
            cont_mask = torch.zeros((x.shape[0], cont_emb.shape[1]), dtype=torch.bool, device=self.device)
            # (可选) 应用额外的场景掩码
            if 'c_pc_mask' in kwargs:
                cont_mask = torch.logical_or(cont_mask, kwargs['c_pc_mask'].repeat(1, cont_mask.shape[1]))
            if 'c_pc_erase' in kwargs:
                cont_emb = cont_emb * (1. - kwargs['c_pc_erase'].unsqueeze(-1).float())
            cont_emb = self.contact_adapter(cont_emb)  # [bs, num_groups, latent_dim]

        # --- 2. 准备运动序列输入 ---
        x = self.motion_adapter(x)  # [bs, seq_len, latent_dim]

        # --- 3. 根据架构进行 Transformer 计算 ---
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

        elif self.arch == 'trans_enc':
            # 将所有序列拼接成一个长序列：[时间, 文本, 接触, 运动]
            x = torch.cat([time_emb, text_emb, cont_emb, x],
                          dim=1)  # [bs, 1 + text_len + num_groups + seq_len, latent_dim]
            # 添加位置编码
            x = self.positional_encoder(x.permute(1, 0, 2)).permute(1, 0, 2)

            x_mask = None
            if self.mask_motion:
                # 拼接对应的掩码
                x_mask = torch.cat([time_mask, text_mask, cont_mask, kwargs['x_mask']], dim=1)

            # 通过 Transformer Encoder
            x = self.self_attn_layer(x, src_key_padding_mask=x_mask)

            # 从输出序列中提取出属于运动的部分
            non_motion_token = time_mask.shape[1] + text_mask.shape[1] + cont_mask.shape[1]
            x = x[:, non_motion_token:, :]

        elif self.arch == 'trans_dec':
            # 拼接序列：[时间, 文本, 运动]
            x = torch.cat([time_emb, text_emb, x], dim=1)  # [bs, 1 + text_len + seq_len, latent_dim]
            # 添加位置编码
            x = self.positional_encoder(x.permute(1, 0, 2)).permute(1, 0, 2)

            x_mask = None
            if self.mask_motion:
                # 拼接对应的掩码
                x_mask = torch.cat([time_mask, text_mask, kwargs['x_mask']], dim=1)

            # 依次通过自注意力和交叉注意力层
            for i in range(len(self.num_layers)):
                # 自注意力
                x = self.self_attn_layers[i](x, src_key_padding_mask=x_mask)

                # 交叉注意力 (除了最后一层)
                if i != len(self.num_layers) - 1:
                    mem = cont_emb[i]  # 获取当前层次的接触特征作为 memory
                    mem_mask = torch.zeros((x.shape[0], mem.shape[1]), dtype=torch.bool, device=self.device)
                    # (可选) 应用额外的场景掩码
                    if 'c_pc_mask' in kwargs:
                        mem_mask = torch.logical_or(mem_mask, kwargs['c_pc_mask'].repeat(1, mem_mask.shape[1]))
                    if 'c_pc_erase' in kwargs:
                        mem = mem * (1. - kwargs['c_pc_erase'].unsqueeze(-1).float())

                    # 将 memory 映射到隐空间维度
                    mem = self.kv_mappling_layers[i](mem)
                    # 执行交叉注意力，x 是 query, mem 是 key 和 value
                    x = self.cross_attn_layers[i](x, mem, tgt_key_padding_mask=x_mask, memory_key_padding_mask=mem_mask)

            # 从输出序列中提取出属于运动的部分
            non_motion_token = time_mask.shape[1] + text_mask.shape[1]
            x = x[:, non_motion_token:, :]

        # 该分支 'trans_mambaTrans' 的逻辑与 'trans_dec' 相同，可能是为未来扩展保留的
        elif self.arch == 'trans_mambaTrans':
            """
            这是为 'trans_mambaTrans' 架构修改后的 forward 函数。

            参数:
            time_emb, text_emb, x: 要拼接的序列
            time_mask, text_mask: 对应的掩码
            cont_emb: 来自 U-Net 的接触特征列表 [mem_0, mem_1, mem_2, mem_3]
            kwargs: 包含 'x_mask', 'c_pc_mask' 等
            """

            # 拼接序列：[时间, 文本, 运动]
            x = torch.cat([time_emb, text_emb, x], dim=1)  # [bs, 1 + text_len + seq_len, latent_dim]
            # 添加位置编码
            x = self.positional_encoder(x.permute(1, 0, 2)).permute(1, 0, 2)

            x_mask = None
            if self.mask_motion:
                # 拼接对应的掩码
                x_mask = torch.cat([time_mask, text_mask, kwargs['x_mask']], dim=1)

            # --- 2. 核心循环 (修改点) ---

            # 依次通过自处理和交叉注意力层
            for i in range(len(self.num_layers)):  # 循环 5 次 (i = 0, 1, 2, 3, 4)

                # --- 2a. 自处理层 (Mamba 或 Transformer) ---

                if i == 0:
                    # --- 第一层 (i=0): 调用 MambaTransBackbone ---

                    # !! 关键假设 !!
                    # 这里的代码 *必须* 依赖您对 MambaTransBackbone.forward 的修改：
                    # 1. 它的 forward 接受 (B, S, C) batch_first 输入
                    # 2. 它的 forward 接受 src_key_padding_mask
                    # 3. 它的 forward 在 *内部* 手动应用掩码 (masked_fill)

                    x = self.self_mamba_block[0](x, src_key_padding_mask=x_mask)

                else:
                    # --- 其他层 (i > 0): 调用 TransformerEncoder ---
                    # nn.TransformerEncoder (batch_first=True) 可以直接接收 src_key_padding_mask
                    # self.self_attn_layers 的索引是从 0 开始的 (对应 i=1, 2, 3, 4)
                    # 所以我们用索引 [i-1]
                    x = self.self_attn_layers[i - 1](x, src_key_padding_mask=x_mask)

                # --- 2b. 交叉注意力层 (与 'trans_dec' 相同) ---

                # 在每个自处理块之后（除了最后一个），插入交叉注意力块
                if i != len(self.num_layers) - 1:
                    mem = cont_emb[i]  # 获取当前层次的接触特征作为 memory
                    mem_mask = torch.zeros((x.shape[0], mem.shape[1]), dtype=torch.bool, device=self.device)
                    # (可选) 应用额外的场景掩码
                    if 'c_pc_mask' in kwargs:
                        mem_mask = torch.logical_or(mem_mask, kwargs['c_pc_mask'].repeat(1, mem_mask.shape[1]))
                    if 'c_pc_erase' in kwargs:
                        mem = mem * (1. - kwargs['c_pc_erase'].unsqueeze(-1).float())

                    # 将 memory 映射到隐空间维度
                    mem = self.kv_mappling_layers[i](mem)
                    # 执行交叉注意力，x 是 query, mem 是 key 和 value
                    x = self.cross_attn_layers[i](x, mem, tgt_key_padding_mask=x_mask, memory_key_padding_mask=mem_mask)

            # --- 3. 输出处理 (与 'trans_dec' 相同) ---

            # 从输出序列中提取出属于运动的部分
            non_motion_token = time_mask.shape[1] + text_mask.shape[1]
            x = x[:, non_motion_token:, :]
        else:
            raise NotImplementedError

        # --- 4. 输出预测结果 ---
        # 将最终的隐状态映射回运动维度
        x = self.motion_layer(x)
        return x