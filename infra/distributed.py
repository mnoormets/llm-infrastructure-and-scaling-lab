"""Full-parameter distributed smoke run; fail closed for multi-GPU modes without CUDA."""
import argparse,os,json
from pathlib import Path

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['ddp','fsdp','zero3'],default='ddp');parser.add_argument('--steps',type=int,default=5);parser.add_argument('--output',default='runs/distributed');a=parser.parse_args()
    import torch
    import torch.distributed as dist
    torch.set_num_threads(1)
    world=int(os.environ.get('WORLD_SIZE','1'));rank=int(os.environ.get('RANK','0'));local=int(os.environ.get('LOCAL_RANK','0'))
    if world<2:raise RuntimeError('Launch with torchrun --nproc-per-node=2; one rank is not a distributed result')
    gpu=a.mode!='ddp'
    if gpu and (not torch.cuda.is_available() or torch.cuda.device_count()<world):raise RuntimeError('FSDP/ZeRO3 require multiple visible CUDA devices; not run')
    if not 1<=a.steps<=50:raise ValueError('Bound steps to 1..50')
    if gpu:torch.cuda.set_device(local)
    init_file=os.environ.get('LAB_INIT_FILE')
    if init_file:
        store=dist.FileStore(init_file,world)
        dist.init_process_group('nccl' if gpu else 'gloo',store=store,rank=rank,world_size=world)
    else:dist.init_process_group('nccl' if gpu else 'gloo')
    try:
        torch.manual_seed(73);device=f'cuda:{local}' if gpu else 'cpu'
        model=torch.nn.Sequential(torch.nn.Linear(64,128),torch.nn.GELU(),torch.nn.Linear(128,64)).to(device)
        parameter_count=sum(p.numel() for p in model.parameters());assert all(p.requires_grad for p in model.parameters())
        if a.mode=='fsdp':
            from torch.distributed.fsdp import FullyShardedDataParallel,ShardingStrategy
            model=FullyShardedDataParallel(model,sharding_strategy=ShardingStrategy.FULL_SHARD,device_id=local,use_orig_params=True)
        elif a.mode=='ddp':model=torch.nn.parallel.DistributedDataParallel(model)
        if a.mode=='zero3':
            import deepspeed
            model,optimizer,_,_=deepspeed.initialize(model=model,model_parameters=model.parameters(),config={'train_micro_batch_size_per_gpu':1,'gradient_accumulation_steps':1,'train_batch_size':world,'zero_optimization':{'stage':3},'optimizer':{'type':'Adam','params':{'lr':2e-4}}})
        else:optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4)
        losses=[]
        for step in range(a.steps):
            torch.manual_seed(1000+rank*100+step);x=torch.randn(1,64,device=device);y=torch.randn(1,64,device=device)
            if a.mode!='zero3':optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.mse_loss(model(x),y)
            if a.mode=='zero3':model.backward(loss);model.step()
            else:loss.backward();optimizer.step()
            assert torch.isfinite(loss);losses.append(float(loss.detach().cpu()))
        checksum=sum(p.detach().double().sum() for p in model.parameters()) if a.mode=='ddp' else None
        matched=None
        if checksum is not None:
            sums=[torch.zeros_like(checksum) for _ in range(world)];dist.all_gather(sums,checksum);matched=all(torch.allclose(sums[0],s,rtol=0,atol=1e-7) for s in sums);assert matched
        out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
        report={'status':'completed','mode':a.mode,'device':device,'world_size':world,'rank':rank,'steps':a.steps,'full_trainable_parameters_before_wrapping':parameter_count,'losses':losses,'ddp_rank_weights_match':matched,'scope':'Small full-parameter MLP smoke run; not pretrained LLM fine-tuning or 70B scaling'}
        (out/f'rank-{rank}.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
    finally:dist.destroy_process_group()
if __name__=='__main__':main()
