#!/usr/bin/env bash
#SBATCH --partition=long
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=80GB
#SBATCH --time=5:00:00
#SBATCH --cpus-per-gpu=1
#SBATCH --output=sbatch_out/test_logicrl.%A.out
#SBATCH --job-name=test_logicrl

module load anaconda/3
module load cuda/12.6.0
conda activate logic
cd YOUR_DIR

export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
MODEL_PATH="Qwen/Qwen2.5-1.5B-Instruct"
set -x
export VLLM_ATTENTION_BACKEND=XFORMERS
SavePath=YOUR_DIR
torchrun --nnodes=1 --nproc-per-node=2 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=reinforce_plus_plus \
    data.train_files=[./data/kk/instruct/3ppl/train.parquet,./data/kk/instruct/4ppl/train.parquet,./data/kk/instruct/5ppl/train.parquet,./data/kk/instruct/6ppl/train.parquet,./data/kk/instruct/7ppl/train.parquet] \
    data.val_files=data/kk/instruct/5ppl/test.parquet \
    data.train_batch_size=8 \
    data.val_batch_size=8 \
    data.max_prompt_length=400 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=3e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=160 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=160 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    trainer.project_name='Re++_logic_KK' \
    trainer.experiment_name='Qwen-1.5B' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.default_local_dir=$SavePath \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=5 $@ 2>&1 | tee grpo.log

