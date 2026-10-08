"""Bounded real PyTorch training/profile run. CPU mode never labels bytes as CUDA VRAM."""
import argparse,json,time
from pathlib import Path

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',choices=['cpu','cuda'],default='cpu');parser.add_argument('--steps',type=int,default=5);parser.add_argument('--output',default='runs/profile');a=parser.parse_args()
    if not 2<=a.steps<=50:parser.error('2..50 steps required')
    import torch
    torch.set_num_threads(2)
    if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('No CUDA GPU; no GPU result claimed')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);torch.manual_seed(73)
    model=torch.nn.Sequential(torch.nn.Linear(256,512),torch.nn.GELU(),torch.nn.Linear(512,256)).to(a.device);optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4)
    phases=[]
    def sample(step,phase):
        row={'step':step,'phase':phase}
        if a.device=='cuda':
            torch.cuda.synchronize();row.update(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
        else:row.update(cuda_memory_available=False)
        phases.append(row)
    activity=[torch.profiler.ProfilerActivity.CPU]
    if a.device=='cuda':activity.append(torch.profiler.ProfilerActivity.CUDA)
    started=time.perf_counter();losses=[]
    with torch.profiler.profile(activities=activity,record_shapes=True,profile_memory=True) as prof:
        for step in range(a.steps):
            if a.device=='cuda':torch.cuda.reset_peak_memory_stats()
            with torch.profiler.record_function('data_transfer'):
                inputs=torch.randn(16,256).to(a.device);target=torch.randn(16,256).to(a.device)
            sample(step,'after_data_transfer');optimizer.zero_grad(set_to_none=True)
            with torch.profiler.record_function('forward'):loss=torch.nn.functional.mse_loss(model(inputs),target)
            sample(step,'after_forward')
            with torch.profiler.record_function('backward'):loss.backward()
            sample(step,'after_backward')
            with torch.profiler.record_function('optimizer'):optimizer.step()
            sample(step,'after_optimizer');losses.append(float(loss.detach().cpu()));prof.step()
    prof.export_chrome_trace(str(out/'trace.json'))
    (out/'operators.txt').write_text(prof.key_averages().table(sort_by='self_cpu_time_total',row_limit=40),encoding='utf-8')
    report={'status':'completed','scope':'Small MLP full training profile, not a 7B/70B benchmark','device':a.device,'torch':torch.__version__,'steps':a.steps,'elapsed_seconds':time.perf_counter()-started,'profiler_and_sync_overhead_included':True,'losses':losses,'memory_phases':phases}
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ('memory_phases','losses')},indent=2))
if __name__=='__main__':main()
