gpt-2 pretrain smoke test:
  source .venv/bin/activate
  export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

  automodel examples/llm_pretrain/nanogpt_pretrain.yaml \
    --nproc-per-node 8 \
    --dataset.file_pattern /tmp/automodel-nanogpt-smoke/fineweb_max_tokens_1M/dataset.bin \
    --step_scheduler.max_steps 10 \
    --checkpoint.checkpoint_dir /tmp/automodel-nanogpt-smoke/checkpoints