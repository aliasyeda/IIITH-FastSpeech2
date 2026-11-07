import json, sys
from pathlib import Path
import numpy as np, torch, soundfile as sf
from .model import FastSpeech2

def load_vocoder(cfg, dev):
    v = cfg.get("vocoder", {})
    try:
        sys.path.insert(0, v["hifigan_repo"]); import hifigan
        g = hifigan.Generator(hifigan.AttrDict(json.load(open(v["config"]))))
        g.load_state_dict(torch.load(v["ckpt"], map_location=dev)["generator"]); g.eval(); g.remove_weight_norm(); return g.to(dev)
    except Exception as e:
        print("[WARN] HiFi-GAN not loaded (%s) -> falling back to Griffin-Lim (lower quality)" % e); return None

def vocode(mel, voc, a, dev):
    if voc is not None:
        with torch.no_grad(): w = voc(mel.T.unsqueeze(0).to(dev)).squeeze().cpu().numpy()
        return np.clip(w, -1, 1)
    import librosa
    return librosa.feature.inverse.mel_to_audio(np.exp(mel.numpy().T), sr=a["sr"], n_fft=a["n_fft"], hop_length=a["hop"], win_length=a["win"],
                                                power=1.0, fmin=a["fmin"], fmax=a["fmax"], n_iter=80)

def _font(lang):
    """Telugu tick labels need a Telugu-capable font; if none is installed, labels fall back to token indices (mapping in JSON)."""
    if lang != "telugu": return None
    from matplotlib import font_manager as fm
    for f in fm.findSystemFonts():
        if any(k in f.lower() for k in ("telugu", "gautami", "pothana", "vani", "mandali", "lohit-te", "akshar")): return fm.FontProperties(fname=f)
    return None

def _plots(sid, ph, d_ms, hz, en, out, lang="english"):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fp = _font(lang); labels = ph if (lang != "telugu" or fp is not None) else [str(i) for i in range(len(ph))]
    x = np.arange(len(ph)); w = max(7, 0.38 * len(ph))
    for sub, y, lab, kind in (("duration", d_ms, "Predicted duration (ms)", "bar"), ("pitch", hz, "Predicted F0 (Hz)", "step"), ("energy", en, "Predicted energy", "bar")):
        fig, ax = plt.subplots(figsize=(w, 3.6))
        (ax.bar(x, y, color="tab:blue") if kind == "bar" else (ax.step(x, y, where="mid", color="tab:red"), ax.plot(x, y, "o", ms=3, color="tab:red")))
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=90, fontsize=8, **({"fontproperties": fp} if fp else {})); ax.set_xlabel("token" if labels is ph else "token index (see inference_summary.json)"); ax.set_ylabel(lab); ax.set_title(f"{sid}: {lab}"); ax.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(Path(out) / "variance_adaptor_plots" / sub / f"{sid}.png", dpi=150); plt.close()

def infer(cfg, lang, ckpt, sentences, out):
    from .pipeline import get_lang
    dev = "cuda" if torch.cuda.is_available() else "cpu"; s = torch.load(ckpt, map_location=dev, weights_only=False)
    st, vocab, a = s["stats"], s["vocab"], s["audio"]; vmap = {p: i for i, p in enumerate(vocab)}
    m = FastSpeech2(len(vocab), s["model_cfg"], st, s["n_spk"], a["n_mels"]).to(dev); m.load_state_dict(s["model"]); m.eval()
    L = get_lang(cfg); voc = load_vocoder(cfg, dev); out = Path(out)
    for sub in ("test_generated_audios", "variance_adaptor_plots/duration", "variance_adaptor_plots/pitch", "variance_adaptor_plots/energy"): (out / sub).mkdir(parents=True, exist_ok=True)
    spk = str(cfg["inference"].get("speaker") or st["default_speaker"]); sid_ = st["speakers"].get(spk, 0); ps = st["per_speaker"].get(spk, st["glob"])
    print(f"[infer] speaker={spk} (id {sid_})")
    summary = {}
    for sid, text in sentences.items():
        norm = L.normalize(text); ph = L.phones(norm); unk = [p for p in ph if p not in vmap]
        if unk: print(f"[WARN] {sid}: tokens unseen in training vocab, dropped: {sorted(set(unk))}"); ph = [p for p in ph if p in vmap]
        if getattr(L, 'unknown', None): print(f"[WARN] {sid}: characters not in alignment dictionary, skipped: {sorted(L.unknown)}")
        src = torch.tensor([[vmap[p] for p in ph]], device=dev)
        with torch.no_grad(): o = m(src, torch.tensor([sid_], device=dev))
        mel = o["post"][0, : int(o["mel_len"][0])].cpu(); wav = vocode(mel, voc, a, dev)
        sf.write(out / "test_generated_audios" / f"{sid}.wav", wav, a["sr"])
        d = o["d"][0].cpu().numpy(); hz = np.exp(o["p"][0].cpu().numpy() * ps["p_std"] + ps["p_mean"]); en = o["e"][0].cpu().numpy() * ps["e_std"] + ps["e_mean"]
        _plots(sid, ph, d * a["hop"] / a["sr"] * 1000, hz, en, out, cfg["language"])
        summary[sid] = dict(input_text=text, normalized_text=norm, phones=ph, duration_frames=d.tolist(), pitch_hz=hz.round(2).tolist(),
                            energy=en.round(3).tolist(), total_seconds=round(len(wav) / a["sr"], 3))
        print(f"{sid}: {len(ph)} phones, {summary[sid]['total_seconds']}s")
    json.dump(summary, open(out / "inference_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
