# =====================
# Hyperparam
set -x
MODEL_PATH=Qwen/Qwen2.5-0.5B
export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_API_KEY=aed6eff902037c69a53ae9a27392b6a997e6ec20
SavePath=./saved_model_debug/r++/

# =====================
# Call trainer
# Param choice explained as follows:
#   algorithm.adv_estimator : algorithm choice
#   algorithm.kl_ctrl.kl_coef : KL divergence coefficient for keeping the new policy close to the reference policy
#   
#   data.max_prompt_length : max number of tokens in the input prompt
#   data.max_response_length : max number of tokens model can generate in response
#
#   actor_rollout_ref.actor.optim.lr : learning rate for the actor model
#   actor_rollout_ref.model.enable_gradient_checkpointing : saves memory by recomputing intermdeiate activation during backward pass
#
#   actor_rollout_ref.actor.ppo_mini_batch_size : mini-batch size used in PPO
#   actor_rollout_ref.actor.ppo_micro_batch_size : micro batch size for gradient accumulation
#   
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=[./data/kk/instruct/3ppl/train.parquet,./data/kk/instruct/4ppl/train.parquet,./data/kk/instruct/5ppl/train.parquet,./data/kk/instruct/6ppl/train.parquet,./data/kk/instruct/7ppl/train.parquet] \
    data.val_files=data/kk/instruct/5ppl/test.parquet \
    data.train_batch_size=8 \
    data.val_batch_size=8 \
    data.max_prompt_length=400 \
    data.max_response_length=768 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    +actor_rollout_ref.model.torch_dtype=fp16 \
    actor_rollout_ref.actor.optim.lr=3e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    trainer.project_name='Re++_logic_KK' \
    trainer.experiment_name='GRPO - Qwen-0.5B - debug' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.default_local_dir=$SavePath \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=5 $@ 2>&1 | tee debug.log

    # Remove GPU memory cache
    # Try to run for 1 node, read meaningful errors: where termination is happening. 
