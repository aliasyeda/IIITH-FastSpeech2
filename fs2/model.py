"""FastSpeech 2 (Ren et al., 2021): FFT encoder -> variance adaptor (duration/pitch/energy) -> FFT decoder -> mel + postnet."""
import torch, torch.nn as nn, torch.nn.functional as F

def sinusoid(n, d):
    pos = torch.arange(n).unsqueeze(1).float(); i = torch.arange(d // 2).unsqueeze(0).float()
    ang = pos / torch.pow(10000, 2 * i / d); t = torch.zeros(n, d)
    t[:, 0::2], t[:, 1::2] = torch.sin(ang), torch.cos(ang); return t

class FFT(nn.Module):
    def __init__(s, d, h, ff, k, p):
        super().__init__()
        s.att = nn.MultiheadAttention(d, h, dropout=p, batch_first=True)
        s.c1, s.c2 = nn.Conv1d(d, ff, k, padding=k // 2), nn.Conv1d(ff, d, 1)
        s.n1, s.n2, s.do = nn.LayerNorm(d), nn.LayerNorm(d), nn.Dropout(p)
    def forward(s, x, pad):
        m = pad.unsqueeze(-1)
        a, _ = s.att(x, x, x, key_padding_mask=pad, need_weights=False)
        x = s.n1(x + s.do(a)).masked_fill(m, 0)
        y = s.c2(s.do(F.relu(s.c1(x.transpose(1, 2))))).transpose(1, 2)
        return s.n2(x + s.do(y)).masked_fill(m, 0)

class Stack(nn.Module):
    def __init__(s, n, d, h, ff, k, p):
        super().__init__(); s.layers = nn.ModuleList(FFT(d, h, ff, k, p) for _ in range(n))
        s.register_buffer("pe", sinusoid(4096, d), persistent=False)
    def forward(s, x, pad):
        x = x + s.pe[: x.size(1)].unsqueeze(0)
        for l in s.layers: x = l(x, pad)
        return x

class VarPred(nn.Module):
    def __init__(s, d, f, k, p):
        super().__init__()
        s.c1, s.c2 = nn.Conv1d(d, f, k, padding=k // 2), nn.Conv1d(f, f, k, padding=k // 2)
        s.n1, s.n2, s.do, s.out = nn.LayerNorm(f), nn.LayerNorm(f), nn.Dropout(p), nn.Linear(f, 1)
    def forward(s, x, pad):
        y = s.do(s.n1(F.relu(s.c1(x.transpose(1, 2))).transpose(1, 2)))
        y = s.do(s.n2(F.relu(s.c2(y.transpose(1, 2))).transpose(1, 2)))
        return s.out(y).squeeze(-1).masked_fill(pad, 0.0)

class Postnet(nn.Module):
    def __init__(s, n_mels, ch=512, k=5, n=5):
        super().__init__(); L = []
        for i in range(n):
            ci, co = (n_mels if i == 0 else ch), (n_mels if i == n - 1 else ch)
            L += [nn.Conv1d(ci, co, k, padding=k // 2), nn.BatchNorm1d(co)] + ([nn.Tanh()] if i < n - 1 else [])
        s.net = nn.Sequential(*L)
    def forward(s, x): return s.net(x.transpose(1, 2)).transpose(1, 2)

def length_regulate(x, d, max_len=None):
    out = [torch.repeat_interleave(x[b], d[b], dim=0) for b in range(x.size(0))]
    T = max_len or max(1, max(o.size(0) for o in out)); T = min(T, 4096)
    lens = torch.tensor([min(o.size(0), T) for o in out], device=x.device)
    out = [F.pad(o[:T], (0, 0, 0, T - min(o.size(0), T))) for o in out]
    return torch.stack(out), lens

class FastSpeech2(nn.Module):
    def __init__(s, n_vocab, m, stats, n_spk=1, n_mels=80):
        super().__init__(); d = m["d"]
        s.emb = nn.Embedding(n_vocab, d, padding_idx=0)
        s.enc = Stack(m["enc_layers"], d, m["heads"], m["ff"], m["kernel"], m["dropout"])
        s.dec = Stack(m["dec_layers"], d, m["heads"], m["ff"], m["kernel"], m["dropout"])
        s.dur, s.pit, s.ene = [VarPred(d, m["var_filter"], m["var_kernel"], m["dropout"]) for _ in range(3)]
        nb = m["n_bins"]
        s.register_buffer("pb", torch.linspace(stats["p_min"], stats["p_max"], nb - 1))
        s.register_buffer("eb", torch.linspace(stats["e_min"], stats["e_max"], nb - 1))
        s.pemb, s.eemb = nn.Embedding(nb, d), nn.Embedding(nb, d)
        s.spk = nn.Embedding(n_spk, d) if n_spk > 1 else None
        s.mel, s.post = nn.Linear(d, n_mels), Postnet(n_mels)

    def forward(s, src, spk=None, max_mel=None, p_t=None, e_t=None, d_t=None, pc=1.0, ec=1.0, dc=1.0):
        pad = src == 0
        x = s.enc(s.emb(src), pad)
        if s.spk is not None: x = x + s.spk(spk).unsqueeze(1)
        logd = s.dur(x, pad)
        p = s.pit(x, pad); x = x + s.pemb(torch.bucketize(p_t if p_t is not None else p * pc, s.pb))
        e = s.ene(x, pad); x = x + s.eemb(torch.bucketize(e_t if e_t is not None else e * ec, s.eb))
        d = d_t if d_t is not None else torch.clamp(torch.round((torch.exp(logd) - 1) * dc), min=1).long().masked_fill(pad, 0)
        x, mel_len = length_regulate(x, d, max_mel)
        mpad = torch.arange(x.size(1), device=x.device)[None] >= mel_len[:, None]
        y = s.dec(x, mpad); mel = s.mel(y).masked_fill(mpad.unsqueeze(-1), 0)
        post = (mel + s.post(mel)).masked_fill(mpad.unsqueeze(-1), 0)
        return dict(mel=mel, post=post, logd=logd, p=p, e=e, d=d, mel_len=mel_len, pad=pad, mpad=mpad)

def fs2_loss(o, mel_t, p_t, e_t, d_t):
    mv, sv = ~o["mpad"], ~o["pad"]
    l1 = lambda a: (F.l1_loss(a, mel_t, reduction="none").mean(-1) * mv).sum() / mv.sum()
    mse = lambda a, b: (((a - b) ** 2) * sv).sum() / sv.sum()
    L = dict(mel=l1(o["mel"]), postnet=l1(o["post"]), pitch=mse(o["p"], p_t), energy=mse(o["e"], e_t),
             duration=mse(o["logd"], torch.log(d_t.float() + 1)))
    L["total"] = sum(L.values()); return L
