import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

inputs = tok("The capital of France is", return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits[0, -1]

probs = torch.softmax(logits, dim=-1)

top = torch.topk(probs, 5)

for p, idx in zip(top.values, top.indices):
    print(f"{tok.decode(idx)!r:>12} {p.item():.1%}")