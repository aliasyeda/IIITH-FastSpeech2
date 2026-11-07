import random, numpy as np, torch
from torch.utils.data import Dataset

class DS(Dataset):
    def __init__(s, feat_dir, ids, vocab, spk_map):
        s.items = []
        for i in ids:
            z = np.load(f"{feat_dir}/{i}.npz")
            s.items.append(dict(id=i, text=np.array([vocab[p] for p in z["phones"]]), mel=z["mel"], p=z["pitch"], e=z["energy"],
                                d=z["dur"], spk=spk_map.get(str(z["spk"]), 0)))
    def __len__(s): return len(s.items)
    def __getitem__(s, i): return s.items[i]

class BucketSampler:
    """Length-bucketed batches with a frame budget (long Telugu utterances -> few per batch, short ones -> many)."""
    def __init__(s, lens, budget, shuffle, max_bs=32, seed=0):
        s.lens, s.budget, s.shuffle, s.max_bs, s.rng = lens, budget, shuffle, max_bs, random.Random(seed)
    def _batches(s):
        key = (lambda i: s.lens[i] * (1 + 0.15 * s.rng.random())) if s.shuffle else (lambda i: s.lens[i])
        cur, mx, out = [], 0, []
        for i in sorted(range(len(s.lens)), key=key):
            m2 = max(mx, s.lens[i])
            if cur and (m2 * (len(cur) + 1) > s.budget or len(cur) >= s.max_bs): out.append(cur); cur, m2 = [], s.lens[i]
            cur.append(i); mx = m2
        if cur: out.append(cur)
        if s.shuffle: s.rng.shuffle(out)
        return out
    def __iter__(s): return iter(s._batches())
    def __len__(s): return len(s._batches())

def _pad(seqs, dt):
    n = max(len(x) for x in seqs); out = np.zeros((len(seqs), n) + seqs[0].shape[1:], dt)
    for i, x in enumerate(seqs): out[i, :len(x)] = x
    return torch.from_numpy(out)

def collate(b):
    return dict(src=_pad([x["text"] for x in b], np.int64), mel=_pad([x["mel"] for x in b], np.float32),
                p=_pad([x["p"] for x in b], np.float32), e=_pad([x["e"] for x in b], np.float32),
                d=_pad([x["d"] for x in b], np.int64), spk=torch.tensor([x["spk"] for x in b]), ids=[x["id"] for x in b])
