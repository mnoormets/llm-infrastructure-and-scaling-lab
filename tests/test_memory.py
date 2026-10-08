import pytest
from infra.memory import weights,kv_cache,training_states

def test_units_and_4bit_metadata():
    assert weights(1_000_000_000)==2_000_000_000
    assert weights(1_000_000_000,4)==500_000_000
    assert weights(1000,4,.9,.125)==763

def test_gqa_kv_and_batch():
    expected=2*32*8*128*8192*2
    assert kv_cache(32,8,128,8192)==expected
    assert kv_cache(32,8,128,8192,batch=100)==100*expected

def test_zero_partition_stages_and_master():
    assert sum(training_states(100,1,0).values())==1600
    assert sum(training_states(100,1,0,False).values())==1200
    states=[training_states(100,2,s) for s in range(4)]
    assert [sum(s.values()) for s in states]==[1600,1000,900,800]

@pytest.mark.parametrize('function,args',[(weights,(0,)),(weights,(100,3)),(kv_cache,(1,0,1,1)),(training_states,(100,0)),(training_states,(100,2,4))])
def test_bad_assumptions(function,args):
    with pytest.raises(ValueError):function(*args)
