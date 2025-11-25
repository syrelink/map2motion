EXP_NAME=$1
PRETRAINED_CKPT=$2
PORT=$3

if [ -z "$EXP_NAME" ]
then
    echo "Usage: $0 <EXP_NAME> <PRETRAINED_CKPT_PATH> [PORT]"
    echo "Example: $0 CMDM-TransMamba-Finetune /path/to/pretrained/model.pt 29500"
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

if [ -z "$PORT" ]
then
    PORT=29500
fi

echo "=== Trans-Mamba Finetuning Configuration ==="
echo "EXP_NAME: $EXP_NAME"
echo "PRETRAINED_CKPT: $PRETRAINED_CKPT"
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
            task.train.max_steps=100000 \
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
            task.train.resume_ckpt=null

'''
bash scripts/t2m_contact_motion/train_trans_mamba_finetune.sh \
    "CMDM-TransMamba-Finetune" \
    "outputs/2025-09-15_15-05-55_RTX4090-real/ckpt/model400000.pt" \
    29500
'''            