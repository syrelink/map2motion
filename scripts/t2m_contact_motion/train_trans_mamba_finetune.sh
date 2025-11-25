EXP_NAME=$1
PRETRAINED_CKPT=$2
PORT=$3

if [ -z "$PORT" ]
then
    PORT=29500
fi

CUDA_VISIBLE_DEVICES=0,1 /home/zq/anaconda3/envs/afford/bin/torchrun --nnodes=1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:${PORT} train_trans_mamba_finetune.py \
            hydra/job_logging=none hydra/hydra_logging=none \
            exp_name=${EXP_NAME} \
            output_dir=outputs \
            platform=TensorBoard \
            diffusion.steps=1000 \
            task=text_to_motion_contact_motion_gen \
            task.dataset.sigma=0.8 \
            task.train.batch_size=64 \
            task.train.max_steps=100000 \  # 微调使用较少的步数
            task.train.save_every_step=20000 \
            task.dataset.train_transforms=['RandomEraseLang','RandomEraseContact','NumpyToTensor'] \
            model=cmdm \
            model.arch='trans_mamba' \
            model.data_repr='h3d' \
            model.text_model.max_length=20 \
            model.freeze_transformer_layers=true \
            model.frozen_layers=[0,1,2,3] \
            model.mamba_layers=1 \
            pretrained_ckpt=${PRETRAINED_CKPT} \
            task.train.resume_ckpt=null

