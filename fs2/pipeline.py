"""Shared preprocessing: prepare (manifests + audit) -> [align: English only] -> features (segments, durations, mel, F0, energy)."""
import os, json, shutil, subprocess, sys, re
from pathlib import Path
from collections import Counter
import numpy as np, yaml
from . import audio, textgrid

SIL = {"", "sil", "sp", "<eps>"}

def load_cfg(path): return yaml.safe_load(open(path, encoding="utf-8"))
def get_lang(cfg):
    if cfg["language"] == "english":
        from .text_en import English; return English(cfg)
    from .text_te import Telugu; return Telugu(cfg)

# ------------------------------------------------------------------ prepare
def prepare(cfg):
    return prepare_telugu(cfg) if cfg["language"] == "telugu" else prepare_english(cfg)

def prepare_english(cfg):
    L = get_lang(cfg); work = Path(cfg["paths"]["work_dir"]); corpus = work / "corpus"; work.mkdir(parents=True, exist_ok=True)
    tr, va = L.select(cfg); words, keep = set(), {"train": [], "val": []}
    for name, items in (("train", tr), ("val", va)):
        for it in items:
            norm = L.normalize(it["text"]); toks = [w for w in norm.split() if w not in L.PUNCT]
            if not toks or not L.ok(norm): continue
            d = corpus / it["spk"]; d.mkdir(parents=True, exist_ok=True); dst = d / f"{it['id']}.wav"
            if not dst.exists():
                try: os.symlink(os.path.abspath(it["wav"]), dst)
                except OSError: shutil.copy(it["wav"], dst)
            (d / f"{it['id']}.lab").write_text(" ".join(toks), encoding="utf-8")
            words.update(toks)
            keep[name].append(dict(it, norm=norm, textgrid=str(work / "align" / it["spk"] / f"{it['id']}.TextGrid")))
    assert not ({i["id"] for i in keep["train"]} & {i["id"] for i in keep["val"]}), "train/val overlap"
    dic = L.build_dictionary(words)
    with open(work / "corpus.dict", "w", encoding="utf-8") as f:
        for w, ph in dic.items():
            if ph: f.write(f"{w}\t{' '.join(ph)}\n")
    for n, it in keep.items():
        json.dump(it, open(work / f"{n}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"[prepare] {n}: {len(it)} utts, {sum(i['dur'] for i in it)/3600:.2f} h")

def prepare_telugu(cfg):
    """Reads the EXISTING manifests/corpus/TextGrids (read-only), audits alignment coverage, writes work/{train,val}.json."""
    import pandas as pd
    c = cfg["data"]; work = Path(cfg["paths"]["work_dir"]); work.mkdir(parents=True, exist_ok=True)
    corpus = Path(c["corpus_dir"]) / c.get("speaker_subdir", "speaker1"); out = Path(c["alignment_dir"])
    tgs = {p.stem: p for p in out.rglob("*.TextGrid")}; print(f"[audit] TextGrids found in {out}: {len(tgs)}")
    items = {}
    for split, csvp in (("train", c["train_manifest"]), ("val", c["val_manifest"])):
        L = []
        for _, r in pd.read_csv(csvp).iterrows():
            sid = str(int(float(r["original_index"])))
            L.append(dict(id=sid, spk=str(r["gender"]), wav=str(corpus / f"{sid}.wav"), text=str(r["text"]), dur=float(r["duration_seconds"]),
                          textgrid=str(tgs[sid]) if sid in tgs else None))
        items[split] = L
    ids = {k: {i["id"] for i in v} for k, v in items.items()}
    assert not (ids["train"] & ids["val"]), "train/val overlap!"
    rep = dict(manifest_counts={k: len(v) for k, v in items.items()}, textgrids_total=len(tgs),
               textgrids_not_in_manifests=sorted(set(tgs) - ids["train"] - ids["val"])[:20], missing=[], bad_textgrid=[])
    logs = [p for ext in ("*.csv", "*.log", "*.txt") for p in list(out.glob(ext)) + list((Path.home() / "Documents/MFA").rglob(ext))
            if p.is_file() and p.stat().st_size < 60e6]
    print("[audit] MFA logs/csv searched:", [str(p) for p in logs][:8])
    for split, L in items.items():
        for it in L:
            prob = []
            if it["textgrid"] is None: prob.append("no_textgrid")
            if not Path(it["wav"]).exists(): prob.append("no_wav")
            if it["textgrid"] and not prob:
                try:
                    iv = textgrid.read_phones(it["textgrid"])
                    if not iv or abs(iv[-1][1] - it["dur"]) > 0.5: prob.append(f"textgrid_end={iv[-1][1]:.2f}s vs wav {it['dur']:.2f}s" if iv else "empty_phones_tier")
                    if any(l.strip() == "spn" for *_, l in iv): prob.append("contains_spn")
                except Exception as e: prob.append(f"parse_error:{e}")
            if prob:
                hits = []
                for p in logs:
                    try:
                        for ln in open(p, encoding="utf-8", errors="ignore"):
                            if re.search(rf"(?<!\d){it['id']}(?!\d)", ln): hits.append(f"{p.name}: {ln.strip()[:160]}"); break
                    except Exception: pass
                rep["missing" if "no_textgrid" in prob else "bad_textgrid"].append(
                    dict(id=it["id"], split=split, problems=prob, wav_seconds=round(it["dur"], 2), n_chars=len(it["text"]), log_hits=hits[:3]))
                it["textgrid"] = None if "no_textgrid" in prob or any(p.startswith(("empty", "parse")) for p in prob) else it["textgrid"]
    for k in ("missing", "bad_textgrid"):
        print(f"[audit] {k}: {len(rep[k])}")
        for m in rep[k]: print("   ", json.dumps(m, ensure_ascii=False))
    usable = {k: [i for i in v if i["textgrid"] and Path(i["wav"]).exists()] for k, v in items.items()}
    for k, v in usable.items():
        h0, h1 = sum(i["dur"] for i in items[k]) / 3600, sum(i["dur"] for i in v) / 3600
        rep[f"{k}_usable"] = dict(utts=len(v), hours=round(h1, 3), of_utts=len(items[k]), of_hours=round(h0, 3))
        print(f"[audit] {k}: usable {len(v)}/{len(items[k])} utts, {h1:.2f}/{h0:.2f} h")
        json.dump(v, open(work / f"{k}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(rep, open(work / "alignment_audit.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("[audit] report ->", work / "alignment_audit.json")

# ------------------------------------------------------------------ align (English; Telugu alignment already exists)
def align(cfg):
    if cfg["language"] == "telugu":
        sys.exit("Telugu alignment already completed (Telugu_Alignment_Output) - nothing to do.")
    m = cfg["mfa"]; work = Path(cfg["paths"]["work_dir"]); env = os.environ.copy()
    env["PATH"] = f"{m['env']}/bin:" + env["PATH"]; env["MPLBACKEND"] = "Agg"
    cmd = [f"{m['env']}/bin/mfa", "align", str(work / "corpus"), str(work / "corpus.dict"), m["acoustic_model"], str(work / "align"),
           "--clean", "-j", str(m.get("jobs", 2)), "--beam", "100", "--retry_beam", "400"]
    print(" ".join(cmd)); subprocess.run(cmd, env=env, check=True)

# ------------------------------------------------------------------ features
def clean_intervals(iv, sil_min):
    """silence -> '' ; merge adjacent silences; merge silences shorter than sil_min into the previous unit; trim edges."""
    out = []
    for s, e, l in iv:
        l = "" if l.strip() in SIL else l.strip()
        if l == "":
            if out and out[-1][2] == "": out[-1] = (out[-1][0], e, ""); continue
            if (e - s) < sil_min and out: out[-1] = (out[-1][0], e, out[-1][2]); continue
        out.append((s, e, l))
    while out and out[0][2] == "": out.pop(0)
    while out and out[-1][2] == "": out.pop()
    return out

def make_segments(iv, seg_max, split_sil, seg_min=2.0):
    """Split long utterances at long silences so each segment is <= seg_max seconds (alignment is reused, no re-alignment)."""
    segs, i0 = [], 0
    while True:
        t0 = iv[i0][0]
        if iv[-1][1] - t0 <= seg_max: segs.append(iv[i0:]); break
        cut = None
        for j in range(i0 + 1, len(iv) - 1):
            s, e, l = iv[j]
            if l == "" and e - s >= split_sil and seg_min <= s - t0 <= seg_max: cut = j
        if cut is None: segs.append(iv[i0:]); break
        segs.append(iv[i0:cut]); i0 = cut + 1
    return segs

def _process(args):
    it, a, d, out_dir = args
    sr, hop = a["sr"], a["hop"]
    res = dict(id=it["id"], status="ok", ids=[], phones=[], drops=Counter(), frames=0)

    if not it.get("textgrid") or not os.path.exists(it["textgrid"]):
        res["status"] = "no_textgrid"
        return res

    raw = textgrid.read_phones(it["textgrid"])
    if any(l.strip() == "spn" for *_, l in raw):
        res["status"] = "spn"
        return res

    iv = clean_intervals(raw, d.get("sil_min_sec", 0.08))
    if len(iv) < 2:
        res["status"] = "empty"
        return res

    segs = make_segments(iv, d.get("seg_max_sec", 15.0), d.get("split_sil_sec", 0.12))
    y = None

    for k, seg in enumerate(segs):
        t0, t1 = seg[0][0], seg[-1][1]
        ph = ["sp" if l == "" else l for _, _, l in seg]
        dur = np.array(
            [int(round(e * sr / hop)) - int(round(s * sr / hop))
             for s, e, _ in seg], np.int64
        )
        T = int(dur.sum())

        if T < 10 or len(ph) < 3:
            res["drops"]["too_short"] += 1
            continue
        if T > d.get("max_frames", 1500):
            res["drops"]["too_long_unsplittable"] += 1
            continue

        sid = it["id"] if len(segs) == 1 else f"{it['id']}_s{k}"
        fp = os.path.join(out_dir, f"{sid}.npz")

        # Reuse existing valid feature files before loading the audio.
        if os.path.exists(fp):
            try:
                with np.load(fp, allow_pickle=False) as z:
                    required = {"phones", "dur", "mel", "pitch", "energy", "spk", "nframes"}
                    if required.issubset(z.files):
                        res["ids"].append(sid)
                        res["phones"] += z["phones"].astype(str).tolist()
                        res["frames"] += int(z["nframes"])
                        continue
            except Exception:
                pass

        if y is None:
            y = audio.load_wav(it["wav"], sr)

        ys = y[int(t0 * sr):int(t1 * sr)]
        mel, en = audio.mel_energy(ys, a)
        f0 = audio.f0_track(ys, a)
        mel, en, f0 = audio.fit(mel, T), audio.fit(en, T), audio.fit(f0, T)

        np.savez(
            fp,
            phones=np.array(ph),
            dur=dur,
            mel=mel,
            pitch=audio.pitch_avg(f0, dur),
            energy=audio.mean_avg(en, dur),
            spk=it["spk"],
            nframes=T
        )
        res["ids"].append(sid)
        res["phones"] += ph
        res["frames"] += T

    if not res["ids"]:
        res["status"] = "no_usable_segment"

    return res


def features(cfg):
    from concurrent.futures import ProcessPoolExecutor
    a, d = cfg["audio"], cfg["data"]; work = Path(cfg["paths"]["work_dir"]); out = work / "features"; out.mkdir(exist_ok=True)
    sr, hop = a["sr"], a["hop"]; splits, inv, rep = {}, Counter(), {}
    for split in ("train", "val"):
        items = json.load(open(work / f"{split}.json", encoding="utf-8")); st, ids, drops, frames = Counter(), [], Counter(), 0
        with ProcessPoolExecutor(d.get("workers", 2)) as ex:
            for n, r in enumerate(ex.map(_process, [(it, a, d, str(out)) for it in items], chunksize=1)):
                st[r["status"]] += 1; ids += r["ids"]; drops.update(r["drops"]); frames += r["frames"]; inv.update(r["phones"])
                if n % 50 == 0: print(f"  [{split}] {n}/{len(items)}", flush=True)
        splits[split] = ids
        rep[split] = dict(input_utts=len(items), input_hours=round(sum(i["dur"] for i in items) / 3600, 3), segments=len(ids),
                          hours=round(frames * hop / sr / 3600, 3), utt_status=dict(st), segment_drops=dict(drops))
        print(f"[features] {split}: {rep[split]}")
    vocab = ["<pad>", "sp"] + sorted(set(inv) - {"sp"})
    # ---- per-speaker normalisation statistics (TRAIN only)
    P, E = {}, {}
    for i in splits["train"]:
        z = np.load(out / f"{i}.npz"); s = str(z["spk"]); P.setdefault(s, []).append(z["pitch"][z["pitch"] != 0]); E.setdefault(s, []).append(z["energy"])
    Pg, Eg = np.concatenate([x for v in P.values() for x in v]), np.concatenate([x for v in E.values() for x in v])
    mk = lambda p, e: dict(p_mean=float(p.mean()), p_std=float(p.std() + 1e-6), e_mean=float(e.mean()), e_std=float(e.std() + 1e-6))
    glob = mk(Pg, Eg); per = {s: (mk(np.concatenate(P[s]), np.concatenate(E[s])) if sum(map(len, P[s])) > 50 else glob) for s in P}
    pn, en_ = [], []
    for s in P:
        pn.append((np.concatenate(P[s]) - per[s]["p_mean"]) / per[s]["p_std"]); en_.append((np.concatenate(E[s]) - per[s]["e_mean"]) / per[s]["e_std"])
    pn, en_ = np.concatenate(pn), np.concatenate(en_)
    st = dict(glob=glob, per_speaker=per, p_min=float(np.percentile(pn, .5)), p_max=float(np.percentile(pn, 99.5)),
              e_min=float(np.percentile(en_, .5)), e_max=float(np.percentile(en_, 99.5)))
    spk_all = set()
    for i in splits["train"] + splits["val"]: spk_all.add(str(np.load(out / f"{i}.npz")["spk"]))
    hrs = {s: 0 for s in spk_all}
    for i in splits["train"]:
        z = np.load(out / f"{i}.npz"); hrs[str(z["spk"])] = hrs.get(str(z["spk"]), 0) + int(z["nframes"])
    for i in splits["train"] + splits["val"]:
        z = dict(np.load(out / f"{i}.npz")); sp = str(z["spk"]); ps = per.get(sp, glob); p = z["pitch"]
        z["pitch"] = np.where(p != 0, np.clip((p - ps["p_mean"]) / ps["p_std"], st["p_min"], st["p_max"]), 0).astype(np.float32)
        z["energy"] = np.clip((z["energy"] - ps["e_mean"]) / ps["e_std"], st["e_min"], st["e_max"]).astype(np.float32)
        np.savez(out / f"{i}.npz", **z)
    st.update(speakers={s: k for k, s in enumerate(sorted(spk_all))}, default_speaker=max(hrs, key=hrs.get), train_frames_per_speaker=hrs)
    rep["vocab_size"] = len(vocab); rep["phone_counts"] = dict(inv.most_common())
    json.dump(vocab, open(work / "phones.json", "w", encoding="utf-8"), ensure_ascii=False)
    json.dump(st, open(work / "stats.json", "w"), indent=1); json.dump(rep, open(work / "preprocess_report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for k, v in splits.items(): (work / f"{k}.txt").write_text("\n".join(v))
    print(f"[features] vocab={len(vocab)} speakers={st['speakers']} default={st['default_speaker']}\n[features] report -> {work/'preprocess_report.json'}")
