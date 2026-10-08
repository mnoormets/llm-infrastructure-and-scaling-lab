"""Causal Transformer full-parameter DDP/FSDP experiment; random initialization."""
import argparse,json,os,subprocess,time
from pathlib import Path
import torch
from torch import nn
import torch.distributed as dist

class Block(nn.Module):
    def __init__(self,width,heads):
        super().__init__();self.width=width;self.heads=heads
        self.norm1=nn.LayerNorm(width);self.qkv=nn.Linear(width,3*width);self.proj=nn.Linear(width,width)
        self.norm2=nn.LayerNorm(width);self.mlp=nn.Sequential(nn.Linear(width,4*width),nn.GELU(),nn.Linear(4*width,width))
    def forward(self,x):
        batch,length,width=x.shape
        q,k,v=self.qkv(self.norm1(x)).chunk(3,dim=-1)
        q,k,v=[t.view(batch,length,self.heads,width//self.heads).transpose(1,2) for t in (q,k,v)]
        attention=nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True)
        x=x+self.proj(attention.transpose(1,2).contiguous().view(batch,length,width))
        return x+self.mlp(self.norm2(x))

class LanguageModel(nn.Module):
    def __init__(self,width=512,layers=8,heads=8,vocab=4096):
        super().__init__();self.embedding=nn.Embedding(vocab,width);self.positions=nn.Embedding(128,width);self.blocks=nn.ModuleList(Block(width,heads) for _ in range(layers))
        self.norm=nn.LayerNorm(width);self.head=nn.Linear(width,vocab,bias=False)
    def forward(self,tokens):
        x=self.embedding(tokens)+self.positions(torch.arange(tokens.shape[1],device=tokens.device))
        for block in self.blocks:x=block(x)
        return self.head(self.norm(x))

def main():
    p=argparse.ArgumentParser();p.add_argument("--mode",choices=["ddp","fsdp"],required=True)
    p.add_argument("--steps",type=int,default=30);p.add_argument("--output",default="runs/kaggle")
    a=p.parse_args();world=int(os.environ.get("WORLD_SIZE","1"));rank=int(os.environ.get("RANK","0"));local=int(os.environ.get("LOCAL_RANK","0"))
    if world!=2 or not torch.cuda.is_available() or torch.cuda.device_count()<2:raise RuntimeError("Exactly two visible CUDA GPUs required; no simulated result")
    if not 1<=a.steps<=30:raise ValueError("Steps 1..30")
    torch.cuda.set_device(local);device=torch.device("cuda",local)
    dist.init_process_group("nccl",device_id=device)
    try:
        torch.manual_seed(73);model=LanguageModel().to(device);full=sum(p.numel() for p in model.parameters())
        if a.mode=="fsdp":
            from functools import partial
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP,ShardingStrategy,MixedPrecision
            from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
            model=FSDP(model,sharding_strategy=ShardingStrategy.FULL_SHARD,
                       auto_wrap_policy=partial(transformer_auto_wrap_policy,transformer_layer_cls={Block}),
                       mixed_precision=MixedPrecision(param_dtype=torch.float16,reduce_dtype=torch.float32,buffer_dtype=torch.float16),device_id=device)
            scaler=ShardedGradScaler(enabled=True)
        else:
            model=nn.parallel.DistributedDataParallel(model,device_ids=[local]);scaler=torch.amp.GradScaler("cuda")
        local_count=sum(p.numel() for p in model.parameters())
        counts=[None]*world;dist.all_gather_object(counts,local_count)
        if a.mode=="fsdp" and not (max(counts)<full and sum(counts)>=full):raise AssertionError("Expected actual parameter shards")
        if a.mode=="ddp" and counts!=[full]*world:raise AssertionError("DDP should replicate parameters")
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4);rows=[]
        for step in range(a.steps):
            torch.manual_seed(10000+step*world+rank)
            tokens=torch.randint(0,4096,(2,129),device=device)
            optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            with torch.autocast("cuda",dtype=torch.float16):
                logits=model(tokens[:,:-1]);loss=nn.functional.cross_entropy(logits.reshape(-1,4096),tokens[:,1:].reshape(-1))
            if not torch.isfinite(loss):raise RuntimeError("Nonfinite loss")
            scaler.scale(loss).backward();scaler.step(optimizer);scaler.update();torch.cuda.synchronize()
            rows.append({"step":step+1,"loss":float(loss.detach()),"seconds":time.perf_counter()-start,
                         "allocated_bytes":torch.cuda.memory_allocated(),"reserved_bytes":torch.cuda.memory_reserved(),
                         "peak_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_reserved_bytes":torch.cuda.max_memory_reserved(),
                         "grad_scale":scaler.get_scale()})
        model.eval();torch.manual_seed(77);fixed=torch.randint(0,4096,(1,32),device=device)
        with torch.no_grad(),torch.autocast("cuda",dtype=torch.float16):checksum=model(fixed).float().sum()
        checks=[torch.zeros_like(checksum) for _ in range(world)];dist.all_gather(checks,checksum)
        matched=all(torch.allclose(checks[0],v,rtol=1e-4,atol=.05) for v in checks)
        if not matched:raise AssertionError("Ranks disagree on the same validation input")
        report={"status":"completed","mode":a.mode,"rank":rank,"world_size":world,"local_device":local,
                "gpu":torch.cuda.get_device_name(local),"torch_version":torch.__version__,"nccl_version":torch.cuda.nccl.version(),
                "nvidia_smi":subprocess.check_output(["nvidia-smi","--query-gpu=index,name,uuid,memory.total","--format=csv,noheader"],text=True),
                "source_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
                "full_parameter_count":full,"local_parameter_elements":local_count,"all_rank_parameter_elements":counts,
                "rank_outputs_match":matched,"full_parameter_training":True,"initialization":"random; not pretrained Llama/Mistral",
                "configuration":{"width":512,"layers":8,"heads":8,"vocab":4096,"batch_per_rank":2,"sequence":128,"lr":1e-4,"mixed_precision":"fp16","optimizer":"AdamW"},
                "steps":rows,"scope":"Two real GPUs and full-parameter synthetic causal Transformer; not 7B/70B training proof"}
        out=Path(a.output)/a.mode;out.mkdir(parents=True,exist_ok=True)
        (out/f"rank-{rank}.json").write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!="steps"}),flush=True)
    finally:dist.destroy_process_group()

if __name__=="__main__":main()
