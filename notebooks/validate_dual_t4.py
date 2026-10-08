"""Validate exported JSON/log evidence, never execute archive contents."""
import argparse,hashlib,json,math,re,statistics,subprocess,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def validate(archive):
    archive=Path(archive)
    with zipfile.ZipFile(archive) as z:
        names=["summary.json","ddp/rank-0.json","ddp/rank-1.json","fsdp/rank-0.json","fsdp/rank-1.json","ddp.log","fsdp.log"]
        if any(z.getinfo(name).file_size>2_000_000 for name in names):raise ValueError("Oversized evidence file")
        summary=json.loads(z.read("summary.json"))
        assert summary["status"]=="completed" and summary["physical_gpu_count"]==2
        reports={mode:[json.loads(z.read(f"{mode}/rank-{rank}.json")) for rank in range(2)] for mode in ["ddp","fsdp"]}
        all_reports=sum(reports.values(),[])
        commits={r["source_commit"] for r in all_reports};assert len(commits)==1
        commit=commits.pop()
        # Check reported provenance refers to an actual local repository commit.
        subprocess.run(["git","cat-file","-e",commit+"^{commit}"],cwd=ROOT,check=True)
        config=all_reports[0]["configuration"]
        assert all(r["configuration"]==config for r in all_reports)
        from infra.sharded_transformer import LanguageModel
        model=LanguageModel(width=config["width"],layers=config["layers"],heads=config["heads"],vocab=config["vocab"])
        parameters=sum(p.numel() for p in model.parameters());del model
        assert all(r["full_parameter_count"]==parameters for r in all_reports)
        device_lines=all_reports[0]["nvidia_smi"].strip().splitlines()
        uuids=[re.search(r"GPU-[a-f0-9-]+",line).group() for line in device_lines]
        assert len(device_lines)==2 and len(set(uuids))==2
        assert all(r["nvidia_smi"]==all_reports[0]["nvidia_smi"] for r in all_reports)
        metrics={}
        for mode,rank_reports in reports.items():
            counts=[r["local_parameter_elements"] for r in rank_reports]
            expected=[parameters]*2 if mode=="ddp" else [parameters//2]*2
            assert counts==expected
            for rank,r in enumerate(rank_reports):
                assert r["rank"]==rank and r["local_device"]==rank and r["world_size"]==2
                assert r["status"]=="completed" and r["mode"]==mode and r["rank_outputs_match"]
                assert r["all_rank_parameter_elements"]==counts
                assert [s["step"] for s in r["steps"]]==list(range(1,31))
                for s in r["steps"]:
                    assert math.isfinite(s["loss"]) and math.isfinite(s["seconds"]) and s["seconds"]>0
                    assert 0<=s["allocated_bytes"]<=s["reserved_bytes"]
                    assert s["peak_allocated_bytes"]>=s["allocated_bytes"]
                    assert s["peak_reserved_bytes"]>=s["reserved_bytes"]
                    assert s["grad_scale"]==65536.0
            slow_rank_step_seconds=[max(r["steps"][i]["seconds"] for r in rank_reports) for i in range(30)]
            peaks=[max(s["peak_allocated_bytes"] for s in r["steps"])/2**30 for r in rank_reports]
            metrics[mode]={"parameter_elements_per_rank":counts,"peak_allocated_gib_per_rank":peaks,
                           "peak_reserved_gib_per_rank":[max(s["peak_reserved_bytes"] for s in r["steps"])/2**30 for r in rank_reports],
                           "median_slow_rank_step_seconds_after_5_warmup":statistics.median(slow_rank_step_seconds[5:]),
                           "warm_step_definition":"Steps 6..30, maximum of rank durations per step; one sequential run per method"}
            recorded=next(m for m in summary["modes"] if m["mode"]==mode)
            assert recorded["all_rank_parameter_elements"]==counts
            assert recorded["max_allocated_gib_per_rank"]==peaks
        difference=max(abs(reports["ddp"][rank]["steps"][i]["loss"]-reports["fsdp"][rank]["steps"][i]["loss"]) for rank in range(2) for i in range(30))
        assert difference<.001
        results=ROOT/"results/dual-t4";results.mkdir(parents=True,exist_ok=True)
        for name in names:
            target=results/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(name))
    result={"archive_sha256":hashlib.sha256(archive.read_bytes()).hexdigest(),"source_commit":commit,"status":"validated",
            "physical_gpu_count":2,"distinct_gpu_uuids_present":True,"full_parameters":parameters,
            "steps_per_mode_per_rank":30,"all_steps_finite":True,"all_recorded_gradient_scales_unchanged":True,
            "rank_output_agreement":"Reported assertion passed in runner; raw output/checksum tensors not exported for independent replay",
            "maximum_ddp_fsdp_loss_difference":difference,"metrics":metrics,
            "fsdp_peak_allocated_reduction_fraction":1-max(metrics["fsdp"]["peak_allocated_gib_per_rank"])/max(metrics["ddp"]["peak_allocated_gib_per_rank"]),
            "scope":"Random-initialized 29.48M causal Transformer, full-parameter synthetic training, two Tesla T4 GPUs",
            "limitations":["No pretrained 7B/70B full fine-tuning","No local GPU replay","One sequential run per method, not a broad speed benchmark","Allocator memory excludes CUDA context and some external allocations","Random synthetic next-token task is not a language quality evaluation"]}
    (results/"validation.json").write_text(json.dumps(result,indent=2)+"\n")
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("archive");a=p.parse_args();print(json.dumps(validate(a.archive),indent=2))
