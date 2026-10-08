"""Two CPU rank smoke using a temporary FileStore; avoids Windows libuv rendezvous."""
import os,subprocess,sys,tempfile
from pathlib import Path

def main():
    with tempfile.TemporaryDirectory() as folder:
        store=str(Path(folder)/'store')
        processes=[]
        for rank in range(2):
            env=os.environ.copy();env.update(WORLD_SIZE='2',RANK=str(rank),LOCAL_RANK=str(rank),LAB_INIT_FILE=store,OMP_NUM_THREADS='1')
            processes.append(subprocess.Popen([sys.executable,'-m','infra.distributed','--mode','ddp','--output','results/cpu-ddp'],env=env))
        try:
            codes=[p.wait(timeout=90) for p in processes]
            if any(codes):raise RuntimeError(f'CPU distributed run failed: {codes}')
        finally:
            for p in processes:
                if p.poll() is None:p.kill();p.wait()
if __name__=='__main__':main()
