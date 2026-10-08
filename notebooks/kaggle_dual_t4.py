"""Run within a Kaggle notebook with Internet on and GPU T4 x2 selected."""
import json,os,shutil,subprocess,sys,time
from pathlib import Path
import torch
if not Path("/kaggle/working").exists():raise RuntimeError("Run in Kaggle")
if not torch.cuda.is_available() or torch.cuda.device_count()!=2:raise RuntimeError("Select GPU T4 x2; exactly two GPUs required")
ROOT=Path("/kaggle/working/llm-infrastructure-and-scaling-lab")
if not ROOT.exists():subprocess.run(["git","clone","https://github.com/mnoormets/llm-infrastructure-and-scaling-lab.git",str(ROOT)],check=True)
else:subprocess.run(["git","-C",str(ROOT),"pull","--ff-only"],check=True)
os.chdir(ROOT)
output=ROOT/"runs"/time.strftime("dual-t4-%Y%m%d-%H%M%S");output.mkdir(parents=True)
status={"status":"running","physical_gpu_count":2,"modes":[],"scope":"Synthetic full-parameter Transformer; no pretrained 7B training"}
try:
    for mode in ["ddp","fsdp"]:
        with (output/f"{mode}.log").open("w") as log:
            result=subprocess.run([sys.executable,"-u","-m","torch.distributed.run","--standalone","--nproc_per_node=2","-m","infra.sharded_transformer","--mode",mode,"--steps","30","--output",str(output)],stdout=log,stderr=subprocess.STDOUT,timeout=1200)
        if result.returncode:raise RuntimeError(f"{mode} exited {result.returncode}; inspect {mode}.log")
        reports=[json.loads((output/mode/f"rank-{rank}.json").read_text()) for rank in range(2)]
        assert all(r["status"]=="completed" and len(r["steps"])==30 and r["rank_outputs_match"] for r in reports)
        status["modes"].append({"mode":mode,"full_parameter_count":reports[0]["full_parameter_count"],"all_rank_parameter_elements":reports[0]["all_rank_parameter_elements"],
            "max_allocated_gib_per_rank":[max(s["peak_allocated_bytes"] for s in r["steps"])/2**30 for r in reports],
            "scope":"DDP replicated versus FSDP FULL_SHARD parameter counts; memory includes activation and communication overhead"})
    status["status"]="completed"
except Exception as error:
    status.update(status="failed",error=str(error));raise
finally:
    (output/"summary.json").write_text(json.dumps(status,indent=2))
    archive=shutil.make_archive("/kaggle/working/dual-t4-experiment","zip",root_dir=output)
    print("Result ZIP:",archive)
    from IPython.display import display,FileLink
    display(FileLink(archive))
