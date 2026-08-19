"""v9 unit check: solution._quantize_weighted gain on Gaussian + timing."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "example", "solution"))
import torch
import solution as S

torch.manual_seed(123)
R, C = 4096, 4096
x = torch.randn(R, C)
ones = torch.ones(1, C)


def deq(p):
    return (p["sign"] * p["mant"] * p["scale_lv3"] * p["scale_lv2"]
            * p["scale_factor"]).flatten(-4, -1)


t0 = time.time()
p = S._quantize_weighted(x, ones)
t1 = time.time()
e_new = ((deq(p) - x) ** 2).mean().item()
alg1 = 6.922078e-03
print(f"v9 search: gain over alg1 {100*(1-e_new/alg1):.2f}%   (target ~11.9)")
print(f"_quantize_weighted 4096x4096: {t1-t0:.2f}s")

# timing on smaller dyn-like tensors
for (r, c) in [(512, 2048), (1024, 4096), (2048, 2048)]:
    y = torch.randn(r, c)
    torch.cuda.synchronize if False else None
    t0 = time.time()
    S._quantize_weighted(y, torch.ones(1, c))
    t1 = time.time()
    t0b = time.time()
    S._quantize_weighted(y, torch.ones(1, c))
    t1b = time.time()
    print(f"  {r}x{c}: {min(t1-t0, t1b-t0b)*1000:.0f} ms")
