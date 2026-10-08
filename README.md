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

Distributed runner trains a small randomly initialized MLP with all parameters; no claim of 8B full fine-tuning, multi-GPU completion or 70B OOM remediation without actual logs. Do not load third-party pickle checkpoints.

Sources: [PyTorch Profiler](https://docs.pytorch.org/docs/stable/profiler.html), [FSDP](https://docs.pytorch.org/docs/stable/fsdp.html), [DeepSpeed ZeRO](https://www.deepspeed.ai/tutorials/zero/).

## Actual local results and prepared LLM route

`results/cpu-profile/` contains an actual five-step PyTorch CPU profile/trace. `results/cpu-ddp/` contains two actual rank receipts: five DDP steps, equal final rank parameter checksums. Windows TCP rendezvous initially failed because its PyTorch build lacked libuv; temporary FileStore fixed it. These results do not prove FSDP or GPU scaling.

Prepared full-model ZeRO3 entry point (not run here):

```
torchrun --standalone --nproc-per-node=2 -m infra.hf_zero3 --model YOUR_AUTHORIZED_MODEL --revision PINNED_COMMIT --data synthetic-text.jsonl --steps 20
```

JSONL requires a `text` field. Full trainable model, no LoRA; sharded initialization configuration is created before model loading. This route and FSDP/DeepSpeed require matching Linux/CUDA/DeepSpeed dependencies and must be validated on available GPUs; no paid rental is authorized. An 8B full Adam training state alone can approach 128 decimal GB with master weights, before activations and temporary gathered layers. Two 24GB cards are not automatically sufficient. ZeRO3 does not promise every model fits. CPU-offload is explained but not silently enabled.
