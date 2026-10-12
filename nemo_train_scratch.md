things:
- megakernels
- ttt-d
- quantization
- cute rewriting
- finish nsys profiling


https://ui.perfetto.dev/#!/?url=https://huggingface.co/buckets/byt-duck/nemotron-triton-peft-traces/resolve/detailed-nemo_triton_peft_hellaswag-ft.json


hf buckets cp ./file.txt hf://buckets/username/my-bucket-name/

gpt-2 pretrain smoke test:
  source .venv/bin/activate
  export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

  automodel examples/llm_pretrain/nanogpt_pretrain.yaml \
    --nproc-per-node 8 \
    --dataset.file_pattern /tmp/automodel-nanogpt-smoke/fineweb_max_tokens_1M/dataset.bin \
    --step_scheduler.max_steps 10 \
    --checkpoint.checkpoint_dir /tmp/automodel-nanogpt-smoke/checkpoints

grouped gemm: pip install git+https://github.com/fanshiqing/grouped_gemm@v1.1.4


nemotron nano-v3 finetune test:
  source /home/palshraya/Automodel.venv/bin/activate

  export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cudnn/lib:$VIRTUAL_ENV/lib/python3.12/site-packages/
  nvidia/nccl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

  uv run automodel examples/llm_finetune/nemotron/nemotron_nano_v3_hellaswag.yaml \
    --nproc-per-node 8


nemotron nano-v3 lora:
uv run automodel examples/llm_finetune/nemotron/nemotron_nano_v3_hellaswag_peft.yaml \ --nproc-per-node 8


== profiling ==

torch.profiler -> zoom level: pytorch operators / modules
	- which pytorch operations are expensive? 

nsys / nsight systems -> zoom level: cpu<->gpu system timeline
	- why is gpu idle? what are the communication, computation gaps?

ncu -> zoom level: individual kernel
	- why is this kernel slow?

nvidia-smi -> node monitoring
	- is this gpu even being used at all?


with nsight:

nsys profile \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --trace=cuda,nvtx,osrt \
    --sample=none \
    --cpuctxsw=none \
    --output=profile_output \
    uv run automodel examples/llm_finetune/nemotron/nemotron_nano_v3_hellaswag_peft.yaml \
        --nproc-per-node 8

export PATH="$HOME/tools/nsight-systems/nsight_systems-linux-x86_64-2024.2.3.38-archive/nsight-systems/2024.2.3/target-linux-x64:$PATH"

/home/palshraya/tools/nsight-systems/nsight_systems-linux-x86_64-2024.2.3.38-archive/nsight-systems/2024.2.3/target-linux-x64

======== oops =======
prev: PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION = ""
prev protobuf: 3.20.3


======== parallelism strategy ========

This recipe uses FSDP2 with 4-way expert parallelism.
distributed:
  strategy: fsdp2
  dp_size: none  # inferred automatically
  tp_size: 1
  cp_size: 1
  ep_size: 4
  sequence_parallel: false
With the suggested 8-GPU launch:
- Data parallel / FSDP size: 8 / (TP 1 × PP 1 × CP 1) = 8
- Expert parallelism: Experts are distributed across groups of 4 GPUs.
- MoE mesh: (PP=1, EP-shard=2, EP=4)
- Tensor parallelism: Disabled.
- Pipeline parallelism: Disabled; omitted means pp_size: 1.
- Context parallelism: Disabled.
- Sequence parallelism: Disabled.
- FSDP2: Shards model state rather than keeping a complete model replica on every GPU.
- LoRA/PEFT: Only the configured low-rank adapters are trained; *.out_proj modules are excluded.
With global_batch_size: 8 and local_batch_size: 1, an 8-GPU run processes one local sample per rank and reaches the global batch without gradient accumulation.
One important constraint: ep_size must divide dp_size × cp_size. Here that means the GPU count must be divisible by 4. So 4 or 8 GPUs work without changing the topology, but simply setting --nproc-per-node 2 would not work unless you reduce ep_size. Neither DDP nor MegatronFSDP can replace FSDP2 here because they do not support this expert-parallel setup.


======== misc =====

       """Forward pass with optional loss computation.

        Supports both BSHD format (``input_ids`` shape ``[B, S]``) and THD format
        (``input_ids`` shape ``[T]`` after ``squeeze_input_for_thd``). When
        ``kwargs["qkv_format"] == "thd"`` AND the attention backend is TE,
        inputs are squeezed to THD before the base-model forward and logits
        are unsqueezed back to ``[1, T, V]`` on exit. SDPA / flex stay in BSHD.

        Pipeline-parallel awareness: when run as a PP stage, ``input_ids`` is
        the upstream stage's hidden-state tensor on non-first stages, and
        ``*mtp_embed_inputs`` carries ``num_nextn_predict_layers`` future-token
        embeddings produced by the first stage. See the Returns section below
        for the per-stage tuple contract. The single-rank (no-PP) path returns
        :class:`NemotronHCausalLMOutputWithPast` unchanged.

        Args:
            input_ids: Input token IDs. BSHD: ``[B, S]``; THD: ``[1, T]``
                (squeezed internally). On non-first PP stages this slot
                instead carries the upstream stage's hidden-state tensor.
            *mtp_embed_inputs: Pre-computed future-token embeddings produced
                by the first PP stage and forwarded between stages as
                positional args. Empty on the single-rank (no-PP) path.
            attention_mask: 2D padding mask ``[B, S]``.
            causal_mask_mapping: Dict with precomputed 4D causal masks
                (key ``"full_attention"`` is consumed).
            inputs_embeds: Pre-computed input embeddings (optional).
            labels: Token IDs for loss computation ``[B, S]`` (optional;
                under PP, loss is computed by ``PipelineCausalLMLoss``).
            past_key_values: Optional ``NemotronHybridCache`` for incremental decoding.
            use_cache: Whether to return ``past_key_values`` for subsequent steps.
            cache_position: Token position indices for cache updates.
            position_ids: Position IDs (forwarded into MTP sublayer kwargs).
            mtp_per_depth_input_ids: Optional globally shifted future-token IDs,
                one tensor per MTP depth. Each tensor has shape [batch,
                sequence], or [tokens] for THD, matching the local CP layout.
                When supplied, MTP embeds these tensors directly instead of
                rolling rank-local ``input_ids``.
            mtp_per_depth_position_ids: Optional pre-computed future-token
                position IDs, one tensor per MTP depth. Each tensor has shape
                [batch, sequence], or [tokens] for THD, matching the local CP
                layout of ``position_ids``. When supplied, MTP consumes these
                tensors directly instead of rolling rank-local positions.
            padding_mask: Padding mask ``[B, S]`` used by the THD squeeze helper
                and as the MoE / mamba 2D mask source.
            logits_to_keep: If > 0, only compute logits for the last
                ``logits_to_keep`` token positions.
            output_hidden_states: Whether to return hidden states.
            return_dict: Accepted for API compatibility (always returns a
                ``NemotronHCausalLMOutputWithPast`` off-PP).
            compute_logits: Whether to materialize logits. Fused-linear loss callers
                may disable this off-PP when they consume final hidden states directly.
            **kwargs: Additional arguments forwarded to the base model
                (e.g. ``qkv_format``, ``cu_seqlens``, ``cu_seqlens_padded``,
                ``max_seqlen``, ``seq_idx``, ``cp_rank``, ``cp_size``,
                ``_packed_seq_ids``).

        Returns:
            Off-PP: :class:`NemotronHCausalLMOutputWithPast` with ``logits``,
            optional ``loss``, ``past_key_values``, ``hidden_states``, and the
            MTP per-depth hidden states / loss-scaling factor when MTP is on.

            Under PP, returns a positional tuple instead:
              * mid stages: ``(hidden_states, *mtp_embed_inputs)`` arity ``1 + D``.
              * last stage: ``(logits, *mtp_per_depth_h, seq_idx)`` arity
                ``1 + D + 1`` when MTP is enabled, else ``logits`` alone.
        """


======== understanding a training iter =======

training step -> `self._run_train_optim_step` [executes a single training step]

[COME BACK TO THIS] preprocessing -> `total num batch tokens` + `total num label tokens` [insiginificant (right now?) compared to giant fwd/bwd pass]

`fwd bwd step` -> `self._forward_backward_step`

[COME BACK TO THIS] there is ~833us gap between `fwd bwd step` and `model fwd pass` -> why? (isn't super signification)

`model fwd pass` -> `out = model(**batch)`
- note we are using SPDA attention -> so we stay in BSHD mode *[TODO]*: could be useful for modifying AI?

(mtp_depth): for predicting tokens multiple timesteps in the future, rather than just the first...can be used to support speculative decoding - https://docs.nvidia.com/nemo/megatron-bridge/0.4.1/training/multi-token-prediction.html
- in this training config, mtp is not active

why separate out this root fwd from the logits matrix production matmul?
- *[TODO]* answer this -> for efficiency reasons, ofc...perhaps also to do with the parallelism...



i think in FSDP the actual gathering of the weights is probably the thing taking a long time------> lets run this profiling again....

training step: 1s -> 534 ms
- cudasStreamSynch: 16ms -> 812ms (very roughly)
  - in prev run: there was someone taking 100% of GPU 4 -> perhaps that GPU was taking especially long to finish its ops due to SMT w/ other process -> everyone else was waiting on it

 atten::
 (https://factory.com/open-source-wikis/pytorch?page=systems%2Faten.md)
 - ATen-level dispath in pytorch -> ATen ("A Tensor Library") is the C++ tensor lib that exposes everything torch.* does at the math level
 - dispatcher checks the types -> (e.g. CUDA -> will queue add kernel op on cuda stream) -> GPU exec kernel


no PP and no TP -> load checkpoints first, then apply FSDP
- with TP>1 -> we shared first and load after so all ranks stay in sync... [not in this case]

tensor: param
device mesh: device_mesh
placements: [Shard(0)]


ok sharding infra:
- /home/palshraya/Automodel/nemo_automodel/recipes/llm/train_ft.py -> instantiante_infrastructure()
    - /home/palshraya/Automodel/nemo_automodel/_transformers/infrastructure.py -> parallelize_model
        - /home/palshraya/Automodel/nemo_automodel/components/moe/parallelizer.py -> apply_fsdp
            - /home/palshraya/Automodel/nemo_automodel/components/moe/parallelizer.py -> parallelizer_utils.fully_shard_by_dtype
                - /home/palshraya/Automodel/nemo_automodel/components/distributed/parallelizer_utils.py -> _call_fully_shard
                    - /home/palshraya/Automodel/nemo_automodel/components/distributed/parallelizer_utils.py -> fully_shard (torch.distributed.fsdp.fully_shard)
                        - before forward and backward passes, pre-forward/backward hooks are responsible for all-gethering parameters and converting model.parameters() from DTensor to plain torch.Tensor ... (read entire thing in documentation)


the only strategy_kwargs are sequence_parallel = False
what is our FSDP2 config?
- (in strategy_config) why is enable_fsdp2_prefetch = False?
    - also what is this prefetch depth?

parallelism -> dictionary of keys, values of defaults + overrides
our values of parallelism sizes:

class ParallelismSizes:
    """Build-time requested parallelism sizes.

    This is durable user intent, not runtime topology. ``MeshContext`` derives
    its size properties from live ``DeviceMesh`` objects after build.
    """

    dp_size: 8 -> all 8 ranks are in FSDP data parallel (we all-gather weights of layer on *all gpus* (bruh this is what all-gather means...) and run independent forward passes...)
    dp_replicate_size: int | None = None -> 1
    tp_size: int = 1
    pp_size: int = 1
    cp_size: int = 1
    ep_size: int = 4

    dp_shard_size = 1

moe_dict -> None

moe_parallel_config: this is our MoEParallelizerConfig:
    ignore_router_for_ac: bool = True
    reshard_after_forward: bool = False
    lm_head_precision: Union[str, torch.dtype] | None = None
    wrap_outer_model: bool = True
    mp_policy: MixedPrecisionPolicy | None = None

strategy_condfig: default FSDP2 class

_create_fsdp2_device_mesh

.build() of DistribuedSetup
- mesh_context -> set here

what is ranks vs world size?
- process IDs of all jobs -> here ranks is None so will default to everything

DeviceMesh
- inconsistent mesh across ranks -> will lead to silent hangs
- shape of device mesh + axes:
shape = (1, 1, 8, 1, 1)

  axes = (
    pp,
    dp_replicate,
    dp_shard,
    cp,
    tp,
  )

device_mesh shape = 1 × 1 × 8 × 1 × 1
-> 8 is the dp-shard dimension and it contains all 8 global process ranks...

moe_mesh
-  shape = 2 × 4

  axes = (ep_shard, ep)

  rank array =
  [[0, 1, 2, 3],
   [4, 5, 6, 7]]

non_pp_size = dp_size * cp_size * tp_size
- ep_shard_size = non_pp_size // ep_size -> 8 // 4
    - ep_size -> split the collection of experts across 4 positions
    - ep_shard_size -> for each such expert position, split that position's expert weight tensors over 2 ranks (GPus/processes)

total: 128 routed experts

# EP out of FSDP ownership; i think.........

chat (TODO:verify): The outer decoder-block FSDP unit does not own the expert parameters. A separate nested expert FSDP unit does own their weight-dimension sharding.

# support for explicit mamba -> attention prefetch first...

moe_parallel_config in `instantiate_infrastructure`

# support support for explicit MOE prefetch...

return type of parallelize_module in apply_ep is just nn.Module type....

MOE sharding:
- the MOE gate / router is sharded over fsdp_mesh
- MOE experts are sharded over ep_shard_mesh

all MOE parameters: are ignored params:
ignored_params (set[nn.Parameter] | None) – Optional(Set[nn.Parameter]): The set of parameters to be ignored by FSDP. They will not be sharded, nor moved to the device during init, nor have their gradients reduced in backward.

# inspired by VeoOmni prefetch code....

[TODO]: understand how this works a bit more

set_modules_to_forward_prefetch -> just selects FSDP modules for which this FSDP module should explcitly prefetch all-gathers...
- so we all-gather *all* weights, no matter which experts are being routed to

so when mamba does a prefetch -> will all-gather the router and each expert group weight dims; experts will still be sharded across expert groups -> which is what we want...


ok so we basically just want to register that the inner FSDP module must also be prefetched........



Nemotronv3Blocks:
- norm + mixer + residual add
    - mixer \in mamba, attention, mlp, moe
-   The NemotronV3Block.forward() then performs the norm, calls that internal mixer, and applies the residual connection:

  hidden_states = self.norm(hidden_states)
  hidden_states = self.mixer(hidden_states)
  hidden_states = residual + hidden_states


thus the outer FSDP units are all NemotronV3Blocks -> the forward loop invokes the entire object 


ahh okok i see the issue...calling naive FSDP explicit prefectch will just prefetch the sharded decoder block weights...if the module is an MOE module, won't actually prefetch the innermost sharded weights -> each rank will have to wait on all-gather for this nested FSDP shard before moving forward.......

ok so how does the Veomni codebase fix this.....
- this registers all the FSDP modules [even the nested ones] for explicit manual prefetch

## handling different dtypes?
- fully_shard_by_dtype -> can create multiple nested FSDP modules bc of the dtype difference... hopefully this is not an issue here....plspsplpspls



# backwards prefetching perhaps later
if fsdp2_backward_prefetch_depth > 0:
    for block_index in range(1, len(model_fsdp_modules)):
        block_fsdp_modules = model_fsdp_modules[block_index]
        if not block_fsdp_modules:
            continue
        prefetch_targets = [
            fsdp_module
            for previous_block_modules in reversed(
                model_fsdp_modules[max(0, block_index - fsdp2_backward_prefetch_depth) : block_index]
            )
            for fsdp_module in reversed(previous_block_modules)
        ]
        if prefetch_targets:
            block_fsdp_modules[-1].set_modules_to_backward_prefetch(prefetch_targets)

========= QUESTIONS ========
what is torch.compile?
	- normally pytorch executes in eager mode, where ~each math operation is a separate kernel launch
	- this command JIT compiles your pytorch operation to be one fused kernel (by looking at computation graph, etc.)

BSHD v. THD sequence format? https://developer.nvidia.com/blog/scale-biology-transformer-models-with-pytorch-and-nvidia-bionemo-recipes/
  - BSHD: every seq in batch padded to max seq length
  - THD: packed inputs without padding tokens -> index vectors to denote boundaries between sequences -> (squeeze BSHD -> THD)
