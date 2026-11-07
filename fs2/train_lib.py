import csv, json, math, shutil, time
from pathlib import Path
import torch, numpy as np
from torch.utils.data import DataLoader
from .data import DS, collate, BucketSampler
from .model import FastSpeech2, fs2_loss

KEYS = ["total", "mel", "postnet", "pitch", "energy", "duration"]

def plot_losses(csv_path, out_dir):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rows = list(csv.DictReader(open(csv_path)))
    for split, fn in (("train", "training_loss.png"), ("val", "validation_loss.png")):
        r = [x for x in rows if x["split"] == split]
        if not r: continue
        fig, ax = plt.subplots(1, 2, figsize=(12, 4)); st = [int(x["step"]) for x in r]
        ax[0].plot(st, [float(x["total"]) for x in r], lw=2); ax[0].set_title(f"{split} total loss"); ax[0].set_xlabel("step"); ax[0].set_ylabel("loss"); ax[0].grid(alpha=.3)
        for k in KEYS[1:]: ax[1].plot(st, [float(x[k]) for x in r], label=k)
        ax[1].set_title(f"{split} loss components"); ax[1].set_xlabel("step"); ax[1].legend(); ax[1].set_yscale("log"); ax[1].grid(alpha=.3)
        plt.tight_layout(); Path(out_dir).mkdir(parents=True, exist_ok=True); plt.savefig(Path(out_dir) / fn, dpi=150); plt.close()

def train(cfg, lang_dir, results_dir):
    t = cfg["train"]; dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(t["seed"]); np.random.seed(t["seed"])
    work, ck = Path(cfg["paths"]["work_dir"]), Path(cfg["paths"]["ckpt_dir"]); ck.mkdir(parents=True, exist_ok=True)
    vocab = json.load(open(work / "phones.json", encoding="utf-8")); vmap = {p: i for i, p in enumerate(vocab)}
    stats = json.load(open(work / "stats.json")); rd = lambda n: (work / n).read_text().split()
    tr, va = DS(work / "features", rd("train.txt"), vmap, stats["speakers"]), DS(work / "features", rd("val.txt"), vmap, stats["speakers"])
    assert not ({x["id"] for x in tr.items} & {x["id"] for x in va.items}), "train/val overlap"
    fl = lambda d: [len(x["mel"]) for x in d.items]
    dl = DataLoader(tr, batch_sampler=BucketSampler(fl(tr), t["batch_frames"], True, t.get("max_bs", 32), t["seed"]), collate_fn=collate, num_workers=t.get("num_workers", 2))
    vdl = DataLoader(va, batch_sampler=BucketSampler(fl(va), t["batch_frames"], False, t.get("max_bs", 32)), collate_fn=collate)
    model = FastSpeech2(len(vocab), cfg["model"], stats, len(stats["speakers"]), cfg["audio"]["n_mels"]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=t["lr"], betas=(.9, .98), eps=1e-9)
    print(f"train={len(tr)} val={len(va)} params={sum(p.numel() for p in model.parameters())/1e6:.1f}M device={dev}")
    step, best, log = 0, 1e9, ck / "loss_log.csv"
    if (ck / "last.pt").exists():
        s = torch.load(ck / "last.pt", map_location=dev, weights_only=False)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); step, best = s["step"], s["best"]; print("resumed from step", step)
    else:
        with open(log, "w", newline="") as f: csv.writer(f).writerow(["split", "step"] + KEYS)
    def mv(b): return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
    def run(b, train_mode):
        o = model(b["src"], b["spk"], max_mel=b["mel"].size(1), p_t=b["p"], e_t=b["e"], d_t=b["d"])
        return fs2_loss(o, b["mel"], b["p"], b["e"], b["d"])
    def save(name):
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step, best=best, vocab=vocab, stats=stats,
                        model_cfg=cfg["model"], audio=cfg["audio"], n_spk=len(stats["speakers"])), ck / name)
    acc, n_acc, t0 = {k: 0.0 for k in KEYS}, 0, time.time()
    while step < t["total_steps"]:
        for b in dl:
            step += 1; b = mv(b); model.train()
            for g in opt.param_groups: g["lr"] = t["lr"] * min(step / t["warmup"], math.sqrt(t["warmup"] / step))
            L = run(b, True); opt.zero_grad(); L["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"]); opt.step()
            for k in KEYS: acc[k] += L[k].item()
            n_acc += 1
            if step % t["log_every"] == 0:
                with open(log, "a", newline="") as f: csv.writer(f).writerow(["train", step] + [acc[k] / n_acc for k in KEYS])
                print(f"step {step} total {acc['total']/n_acc:.4f} ({time.time()-t0:.0f}s)"); acc, n_acc = {k: 0.0 for k in KEYS}, 0
            if step % t["val_every"] == 0 or step == t["total_steps"]:
                model.eval(); vs = {k: 0.0 for k in KEYS}; n = 0
                with torch.no_grad():
                    for vb in vdl:
                        vb = mv(vb); L = run(vb, False); nb = len(vb["ids"])
                        for k in KEYS: vs[k] += L[k].item() * nb
                        n += nb
                vs = {k: v / n for k, v in vs.items()}
                with open(log, "a", newline="") as f: csv.writer(f).writerow(["val", step] + [vs[k] for k in KEYS])
                print(f"  [val] step {step} total {vs['total']:.4f}")
                if vs["total"] < best: best = vs["total"]; save("best.pt")
                plot_losses(log, results_dir)
            if step % t["save_every"] == 0 or step == t["total_steps"]: save("last.pt")
            if step >= t["total_steps"]: break
    plot_losses(log, results_dir)
    dst = Path(lang_dir) / "checkpoints"; dst.mkdir(exist_ok=True)
    if (ck / "best.pt").exists(): shutil.copy(ck / "best.pt", dst / "best.pt")
    shutil.copy(log, Path(results_dir) / "loss_log.csv"); print("done; best val", best)
