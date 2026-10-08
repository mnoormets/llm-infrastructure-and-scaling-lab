"""Prepared, unexecuted multi-GPU full pretrained-LM fine-tuning route. No model download before GPU gate."""
import argparse,os,json,hashlib
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--revision',required=True);p.add_argument('--data',required=True);p.add_argument('--output',default='runs/hf-zero3');p.add_argument('--steps',type=int,default=20);a=p.parse_args()
    import torch
    world=int(os.environ.get('WORLD_SIZE','1'));local=int(os.environ.get('LOCAL_RANK','0'))
    if world<2 or not torch.cuda.is_available() or torch.cuda.device_count()<world:raise RuntimeError('Requires at least two visible GPUs under torchrun; free single-T4 Colab is insufficient. No model weights downloaded.')
    if not 1<=a.steps<=100:raise ValueError('Bound training to 1..100 steps')
    torch.cuda.set_device(local)
    from transformers import AutoTokenizer,AutoModelForCausalLM,TrainingArguments,Trainer,default_data_collator
    data=Path(a.data);raw=data.read_bytes();texts=[json.loads(line)['text'] for line in raw.decode().splitlines() if line.strip()]
    if not texts or len(texts)>5000 or any(not isinstance(t,str) or len(t)>10000 for t in texts):raise ValueError('Expected bounded JSONL text corpus')
    config=Path(__file__).resolve().parents[1]/'configs/zero3.json'
    native_bf16=torch.cuda.is_bf16_supported(including_emulation=False)
    args=TrainingArguments(output_dir=a.output,max_steps=a.steps,per_device_train_batch_size=1,gradient_accumulation_steps=1,learning_rate=1e-5,warmup_steps=max(1,a.steps//10),bf16=native_bf16,fp16=not native_bf16,deepspeed=str(config),gradient_checkpointing=True,save_strategy='no',logging_steps=1,report_to='none',seed=73)
    # Keep args alive: its HF DeepSpeed configuration is initialized BEFORE from_pretrained,
    # enabling sharded initialization rather than replicating the full model on each GPU.
    tokenizer=AutoTokenizer.from_pretrained(a.model,revision=a.revision,trust_remote_code=False);tokenizer.pad_token=tokenizer.eos_token
    model=AutoModelForCausalLM.from_pretrained(a.model,revision=a.revision,trust_remote_code=False,use_safetensors=True,dtype=torch.bfloat16 if native_bf16 else torch.float16)
    model.config.use_cache=False
    if not all(p.requires_grad for p in model.parameters()):raise RuntimeError('Full fine-tuning requires all parameters trainable')
    rows=[]
    for text in texts:
        item=tokenizer(text,max_length=512,padding='max_length',truncation=True)
        item['labels']=[token if mask else -100 for token,mask in zip(item['input_ids'],item['attention_mask'])];rows.append(item)
    result=Trainer(model=model,args=args,train_dataset=rows,data_collator=default_data_collator).train()
    if int(os.environ.get('RANK','0'))==0:
        out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
        (out/'report.json').write_text(json.dumps({'status':'completed','mode':'zero3-full-finetune','model':a.model,'revision':a.revision,'world_size':world,'steps':a.steps,'data_sha256':hashlib.sha256(raw).hexdigest(),'metrics':result.metrics,'scope':'Bounded full pretrained-LM training smoke, not production quality or serving benchmark'},indent=2))
if __name__=='__main__':main()
