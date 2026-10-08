"""Analytical tensor storage estimates; excludes activations, workspace and allocator overhead."""
import argparse,json,math

def weights(parameters,bits=16,quantized_fraction=1.0,metadata_bytes_per_quantized_parameter=0.0):
    if parameters<=0 or bits not in (2,4,8,16,32) or not 0<=quantized_fraction<=1 or metadata_bytes_per_quantized_parameter<0:raise ValueError('Invalid weight assumptions')
    return math.ceil(parameters*(quantized_fraction*(bits/8+metadata_bytes_per_quantized_parameter)+(1-quantized_fraction)*2))

def kv_cache(layers,kv_heads,head_dim,sequence,batch=1,element_bytes=2):
    if any(x<=0 for x in (layers,kv_heads,head_dim,sequence,batch,element_bytes)):raise ValueError('Positive dimensions required')
    return 2*layers*kv_heads*head_dim*sequence*batch*element_bytes

def training_states(parameters,world_size=1,stage=0,master_weights=True):
    if parameters<=0 or world_size<1 or stage not in (0,1,2,3):raise ValueError('Invalid training assumptions')
    # Explicit assumption: 16-bit weights/gradients, FP32 Adam moments, optional FP32 master weights.
    shard=lambda b: b/world_size
    w=parameters*2;g=parameters*2;o=parameters*(8+(4 if master_weights else 0))
    return {'weights':shard(w) if stage==3 else w,'gradients':shard(g) if stage>=2 else g,'optimizer_and_master':shard(o) if stage>=1 else o}

def describe(value):return {'bytes':value,'decimal_GB':value/1e9,'binary_GiB':value/2**30}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--parameters',type=int,default=8_000_000_000);parser.add_argument('--world-size',type=int,default=2);a=parser.parse_args()
    result={'scope':'Analytical estimates, not measured VRAM or proof of fit. Excludes activations, communication/all-gather buffers, KV cache, quantization workspaces and fragmentation. Shards are idealized averages; offload changes placement, not total state size.', 'parameters':a.parameters,'weights_fp16':describe(weights(a.parameters)),'weights_4bit_ideal':describe(weights(a.parameters,4)),'kv_example':{'assumptions':'32 layers, 8 KV heads, head_dim 128, 8192 tokens, batch 1, 2-byte KV dtype','storage':describe(kv_cache(32,8,128,8192))},'zero':{str(s):{k:describe(v) for k,v in training_states(a.parameters,a.world_size,s).items()} for s in range(4)}}
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
