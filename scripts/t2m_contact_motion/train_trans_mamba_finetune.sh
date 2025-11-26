EXP_NAME=$1
PRETRAINED_CKPT=$2
RESUME_CKPT=$3
PORT=$4

if [ -z "$EXP_NAME" ]
then
    echo "Usage: $0 <EXP_NAME> <PRETRAINED_CKPT_PATH> [RESUME_CKPT] [PORT]"
    echo "Example (fresh training): $0 CMDM-TransMamba-Finetune /path/to/pretrained/model.pt"
    echo "Example (resume training): $0 CMDM-TransMamba-Finetune /path/to/pretrained/model.pt /path/to/resume/model.pt 29500"
    exit 1
fi

if [ -z "$PRETRAINED_CKPT" ]
then
    echo "Error: PRETRAINED_CKPT is required. Please provide the path to pretrained encoder checkpoint."
    exit 1
fi

if [ ! -f "$PRETRAINED_CKPT" ]
then
    echo "Error: Pretrained checkpoint not found at $PRETRAINED_CKPT"
    exit 1
fi

# 检查是否有第4个参数（PORT），如果没有则设置默认值
if [ -z "$4" ]
then
    PORT=29500
else
    PORT=$4
fi

# 如果提供了RESUME_CKPT，检查它是否存在
if [ -n "$RESUME_CKPT" ] && [ ! -f "$RESUME_CKPT" ]
then
    echo "Error: Resume checkpoint not found at $RESUME_CKPT"
    exit 1
fi

echo "=== Trans-Mamba Finetuning Configuration ==="
echo "EXP_NAME: $EXP_NAME"
echo "PRETRAINED_CKPT: $PRETRAINED_CKPT"
if [ -n "$RESUME_CKPT" ]
then
    echo "RESUME_CKPT: $RESUME_CKPT"
    RESUME_PARAM="task.train.resume_ckpt='$RESUME_CKPT'"
else
    echo "RESUME_CKPT: None (fresh training)"
    RESUME_PARAM="task.train.resume_ckpt=null"
fi
echo "PORT: $PORT"
echo "============================================"

CUDA_VISIBLE_DEVICES=0,1 /home/zq/anaconda3/envs/afford/bin/torchrun --nnodes=1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:${PORT} train_trans_mamba_finetune.py \
            hydra/job_logging=none hydra/hydra_logging=none \
            exp_name=${EXP_NAME} \
            output_dir=outputs \
            platform=TensorBoard \
            diffusion.steps=1000 \
            task=text_to_motion_contact_motion_gen \
            task.dataset.sigma=0.8 \
            task.train.batch_size=64 \
            task.train.max_steps=300000 \
            task.train.save_every_step=20000 \
            task.dataset.train_transforms=['RandomEraseLang','RandomEraseContact','NumpyToTensor'] \
            model=cmdm \
            model.arch='trans_mamba' \
            model.data_repr='h3d' \
            model.text_model.max_length=20 \
            model.freeze_transformer_layers=true \
            model.frozen_layers="[0,1,2,3]" \
            model.mamba_layers=1 \
            pretrained_ckpt=${PRETRAINED_CKPT} \
            ${RESUME_PARAM}

'''
# 全新训练示例
bash scripts/t2m_contact_motion/train_trans_mamba_finetune.sh \
    "CMDM-TransMamba-Finetune" \
    "outputs/2025-09-15_15-05-55_RTX4090-real/ckpt/model400000.pt" \
    29500

# 从断点恢复训练示例
bash scripts/t2m_contact_motion/train_trans_mamba_finetune.sh \
    "CMDM-TransMamba-Finetune" \
    "outputs/2025-09-15_15-05-55_RTX4090-real/ckpt/model400000.pt" \
    "outputs/2025-11-25_21-40-47_CMDM-TransMamba-Finetune/ckpt/model100000.pt" \
    29500
'''            

