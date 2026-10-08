import torch
from infra.sharded_transformer import LanguageModel

def test_causal_mask_blocks_future_information():
    torch.manual_seed(73)
    model=LanguageModel(width=32,layers=2,heads=4,vocab=64).eval()
    a=torch.tensor([[1,2,3,4,5]])
    b=torch.tensor([[1,2,3,8,9]])
    with torch.no_grad():first=model(a);second=model(b)
    assert first.shape==(1,5,64)
    assert torch.allclose(first[:,:3],second[:,:3],atol=1e-6)
    assert not torch.allclose(first[:,3:],second[:,3:])

def test_all_parameters_have_finite_gradients_and_optimizer_updates():
    torch.manual_seed(73)
    model=LanguageModel(width=32,layers=2,heads=4,vocab=64)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
    tokens=torch.randint(0,64,(2,9))
    old=model.head.weight.detach().clone()
    logits=model(tokens[:,:-1])
    loss=torch.nn.functional.cross_entropy(logits.reshape(-1,64),tokens[:,1:].reshape(-1))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    assert not torch.equal(old,model.head.weight)
