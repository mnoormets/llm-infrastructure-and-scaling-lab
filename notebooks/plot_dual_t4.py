"""Reproduce memory and step-time charts from exported JSON only."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
out=Path(__file__).resolve().parents[1]/"results/dual-t4"
fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
for mode in ["ddp","fsdp"]:
    rows=json.loads((out/mode/"rank-0.json").read_text())["steps"]
    axes[0].plot([s["step"] for s in rows],[s["peak_allocated_bytes"]/2**30 for s in rows],label=mode.upper())
    axes[1].plot([s["step"] for s in rows[5:]],[s["seconds"]*1000 for s in rows[5:]],label=mode.upper())
axes[0].set(xlabel="Training step",ylabel="Rank 0 peak allocated (GiB)")
axes[1].set(xlabel="Training step (first 5 omitted)",ylabel="Rank 0 step time (ms)")
for ax in axes:ax.legend();ax.grid(alpha=.2)
fig.suptitle("29.48M random Transformer · 2× Tesla T4 · one run per method")
fig.savefig(out/"memory-step-time.png",dpi=150);plt.close(fig)
