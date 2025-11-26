# Trans-Mamba 微调指南

## 🚀 快速开始

### 1. 确保你有预训练的 encoder checkpoint

你需要一个训练好的 CMDM 模型 checkpoint，用于提供预训练的权重。通常这个checkpoint应该位于：
```
/path/to/your/pretrained/model.pt
```

### 2. 运行微调训练

```bash
# 全新训练
bash scripts/t2m_contact_motion/train_trans_mamba_finetune.sh \
    "CMDM-TransMamba-Finetune" \
    "/path/to/your/pretrained/encoder/checkpoint.pt" \
    29500

# 从断点恢复训练（添加第三个参数为resume checkpoint路径）
bash scripts/t2m_contact_motion/train_trans_mamba_finetune.sh \
    "CMDM-TransMamba-Finetune" \
    "/path/to/your/pretrained/encoder/checkpoint.pt" \
    "/path/to/resume/checkpoint.pt" \
    29500
```

### 3. 参数说明

- `EXP_NAME`: 实验名称，用于创建输出目录
- `PRETRAINED_CKPT_PATH`: 预训练checkpoint的完整路径（必需）
- `RESUME_CKPT_PATH`: 从断点恢复的checkpoint路径（可选，用于恢复训练）
- `PORT`: 多GPU训练的端口号（可选，默认29500）

## 🔧 架构说明

- **总层数**: 5层 (num_layers: [1,1,1,1,1])
- **架构**: 前4层使用 Transformer，最后1层使用 Mamba
- **冻结策略**: 冻结前4层（索引0-3），只训练最后一层 Mamba（索引4）
- **优势**: 大幅减少训练参数，提升训练效率

## 📊 预期效果

- **训练效率**: 只训练最后一层参数，训练速度提升约80%
- **性能提升**: 利用预训练的特征表示，专注于学习序列建模
- **目标**: 降低FID，提升top1/2/3准确率

## ⚠️ 注意事项

1. **Checkpoint路径**: 确保预训练checkpoint路径正确且文件存在
2. **CUDA环境**: 确保有足够的GPU内存（脚本默认使用2张GPU）
3. **训练时长**: 微调通常需要比全量训练更少的步数（设置为100k步）

## 🔍 验证训练状态

训练开始时，你应该看到：
```
=== Trans-Mamba Finetuning Configuration ===
EXP_NAME: CMDM-TransMamba-Finetune
PRETRAINED_CKPT: /path/to/checkpoint.pt
PORT: 29500
===========================================
Frozen layer 0: TransformerEncoderLayer
Frozen layer 1: TransformerEncoderLayer
Frozen layer 2: TransformerEncoderLayer
Frozen layer 3: TransformerEncoderLayer
Loaded X parameters from pretrained weights
Skipped 4 frozen parameters
```

## 🆘 故障排除

### 错误："PRETRAINED_CKPT is required"
- 解决方案：提供正确的预训练checkpoint路径

### 错误："Pretrained checkpoint not found"
- 解决方案：检查文件路径是否正确，文件是否存在

### Hydra配置错误
- 解决方案：检查配置文件语法，特别是数组格式 `[0,1,2,3]`

### CUDA内存不足
- 解决方案：减少batch_size或使用单GPU训练
