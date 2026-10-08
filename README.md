# LLM Infrastructure and Scaling Lab

Source-backed memory estimates, phase profiling and fail-closed distributed smoke runners. Free-only budget: no GPU rental or account creation.

## Run

Install PyTorch for your hardware and `pip install -r requirements.txt`.

```
python -m infra.memory --parameters 8000000000 --world-size 2
python -m infra.profile --device cpu --steps 5
python -m infra.profile --device cuda --steps 5
# Linux/multi-GPU environment, not a GPU result on a single Colab T4:
torchrun --standalone --nproc-per-node=2 -m infra.distributed --mode fsdp
torchrun --standalone --nproc-per-node=2 -m infra.distributed --mode zero3
# CPU collective smoke only:
torchrun --standalone --nproc-per-node=2 -m infra.distributed --mode ddp
```

FSDP runner uses FULL_SHARD. SHARD_GRAD_OP retains unsharded parameters after forward until backward; HYBRID_SHARD shards within groups and replicates across groups. This repository does not benchmark those strategies. ZeRO-1 partitions optimizer state; ZeRO-2 also gradients; ZeRO-3 also parameters. Offload moves storage to CPU/NVMe but adds transfer cost; it does not delete required storage.

Memory math explicitly distinguishes decimal GB from binary GiB. FP16 weight-only cost is 2 bytes/parameter; 4-bit ideal is 0.5 plus quantization metadata and unquantized tensors. Adam state assumptions are explicit: 2-byte weights + 2-byte gradients + 8-byte moments (+ optional 4-byte master weights). These are persistent states, not total training VRAM. Activations, gathered layers, collective buffers and allocator overhead can cause OOM even if the estimate fits. KV cache uses KV heads (GQA/MQA), batch, actual cache length and dtype; attention query head count is not interchangeable.

Profiler traces include real labelled data transfer, forward, backward and optimizer phases. CPU mode never fabricates CUDA memory. Synchronization and profiler overhead are included; traces are diagnostic, not serving latency measurements. `chrome://tracing` or a compatible trace viewer can inspect trace.json.

Distributed runner trains a small randomly initialized MLP with all parameters; its original CPU evidence does not prove 8B full fine-tuning or 70B OOM remediation. The separate Transformer dual-T4 result below now supplies actual DDP/FSDP GPU logs. Do not load third-party pickle checkpoints.

Sources: [PyTorch Profiler](https://docs.pytorch.org/docs/stable/profiler.html), [FSDP](https://docs.pytorch.org/docs/stable/fsdp.html), [DeepSpeed ZeRO](https://www.deepspeed.ai/tutorials/zero/).

## Actual local results and prepared LLM route

`results/cpu-profile/` contains an actual five-step PyTorch CPU profile/trace. `results/cpu-ddp/` contains two actual rank receipts: five DDP steps, equal final rank parameter checksums. Windows TCP rendezvous initially failed because its PyTorch build lacked libuv; temporary FileStore fixed it. These results do not prove FSDP or GPU scaling.

Prepared full-model ZeRO3 entry point (not run here):

```
torchrun --standalone --nproc-per-node=2 -m infra.hf_zero3 --model YOUR_AUTHORIZED_MODEL --revision PINNED_COMMIT --data synthetic-text.jsonl --steps 20
```

JSONL requires a `text` field. Full trainable model, no LoRA; sharded initialization configuration is created before model loading. This route and FSDP/DeepSpeed require matching Linux/CUDA/DeepSpeed dependencies and must be validated on available GPUs; no paid rental is authorized. An 8B full Adam training state alone can approach 128 decimal GB with master weights, before activations and temporary gathered layers. Two 24GB cards are not automatically sufficient. ZeRO3 does not promise every model fits. CPU-offload is explained but not silently enabled.

## Free Kaggle dual-T4 experiment

Upload `notebooks/kaggle_dual_t4.ipynb` to Kaggle, choose GPU T4 x2 and enable Internet, then Run all. GPU access depends on your account/quota. The runner refuses fewer than two real visible CUDA devices and records their UUIDs.

The same random-initialized eight-block causal Transformer is full-parameter trained for 30 steps with DDP, then FSDP FULL_SHARD. DDP stores replicated parameters; FSDP reports actual per-rank shard element counts. Both validate matching rank outputs on identical input and export allocator peaks per step. Mixed FP16 computations, FP32 AdamW states and gradient scaling; no pretrained Llama/Mistral fine-tuning or 70B claim. Transformer causality and full gradient/update flow are checked locally; real two-GPU execution has now been verified from exported Kaggle receipts: [report, graph and raw rank logs](results/dual-t4/README.md).

CPU DDP uses Gloo; CUDA collectives use NCCL. CPU multiprocess tests are not physical GPU simulations or evidence of FSDP GPU memory savings. Full 8B Adam states alone can exceed 32 GB, so two T4s do not imply a fitting 7B/8B full fine-tuning workload. Measure the prepared manageable Transformer first.

Sources: [Kaggle notebooks](https://www.kaggle.com/docs/notebooks), [PyTorch distributed backends](https://docs.pytorch.org/docs/stable/distributed.html), [FSDP](https://docs.pytorch.org/docs/stable/fsdp.html).

Verified dual-T4 result: FSDP halved resident parameters per rank and reduced measured peak allocated memory by 57.66% in the 29.48M synthetic Transformer workload. Warm-step median was 48.94 ms DDP versus 53.39 ms FSDP (one sequential run each); no universal speedup claim.
