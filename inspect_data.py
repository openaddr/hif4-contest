"""Inspect mini_sample linear.pt / attn.pt: shapes, dtypes, value distributions."""
import torch

for name in ("linear", "attn"):
    data = torch.load(f"example/mini_sample/{name}.pt", weights_only=True, map_location="cpu")
    print(f"===== {name}.pt: {len(data)} group(s) =====")
    for gi, g in enumerate(data):
        print(f"--- group {gi} keys: {list(g.keys())}")
        if name == "linear":
            wq, ws = g["weight"]
            print(f"weight_quant {tuple(wq.shape)} {wq.dtype}, weight_scale {tuple(ws.shape)} {ws.dtype}")
            print("  wq stats: min %.4g max %.4g  unique(%d) %s" % (
                wq.min(), wq.max(), wq.unique().numel(),
                wq.unique()[:8].tolist()))
            print("  ws stats: min %.4g max %.4g mean %.4g" % (ws.min(), ws.max(), ws.float().mean()))
            for tag in ("calib", "test"):
                for i, (aq, as_) in enumerate(g[f"{tag}_activation_list"]):
                    print(f"  {tag}[{i}] act {tuple(aq.shape)} {aq.dtype} scale {tuple(as_.shape)}  aq[min=%.3g,max=%.3g] as[min=%.3g,max=%.3g]" % (
                        aq.min(), aq.max(), as_.min(), as_.max()))
        else:
            print(f"q_num_heads={g['q_num_heads']} kv_num_heads={g['kv_num_heads']} head_dim={g['head_dim']}")
            for tag in ("calib", "test"):
                for i, s in enumerate(g[tag]):
                    q, k, v = s["q"], s["k"], s["v"]
                    print(f"  {tag}[{i}] q{tuple(q[0].shape)} k{tuple(k[0].shape)} v{tuple(v[0].shape)}")
                    if i == 0:
                        for role, (qq, ss) in (("q", q), ("k", k), ("v", v)):
                            print(f"    {role}: quant[min=%.3g,max=%.3g] uniq=%d  scale[min=%.3g,max=%.3g] mean=%.3g" % (
                                qq.min(), qq.max(), qq.unique().numel(), ss.min(), ss.max(), ss.float().mean()))
