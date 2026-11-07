"""English text front-end: normalisation (numbers/times/years) + g2p_en (ARPAbet, stress stripped, 'sp' for punctuation)."""
import re, json, random
from pathlib import Path

HETERO = {"conduct", "record", "present", "object", "project", "permit", "content", "contract", "desert", "live", "read"}
SIB, VLS = {"S", "Z", "SH", "ZH", "CH", "JH"}, {"P", "T", "K", "F", "TH"}

class English:
    PUNCT = set(",.;:!?")
    def __init__(self, cfg):
        self._g = None; self.cache = {}
    @property
    def g2p(self):
        if self._g is None:
            import nltk
            for p in ("averaged_perceptron_tagger", "averaged_perceptron_tagger_eng", "cmudict", "punkt", "punkt_tab"):
                try: nltk.download(p, quiet=True)
                except Exception: pass
            from g2p_en import G2p
            self._g = G2p()
        return self._g

    def normalize(self, t):
        from num2words import num2words
        t = t.replace("’", "'").replace("‘", "'")
        t = re.sub(r"<[^>]*>|\[[^\]]*\]", " ", t)
        def tm(m):
            h, mi = int(m.group(1)), int(m.group(2)); ap = (m.group(3) or "").lower().replace(".", "")
            s = num2words(h)
            if mi == 0: s += "" if ap else " o'clock"
            elif mi < 10: s += " oh " + num2words(mi)
            else: s += " " + num2words(mi)
            if ap: s += " ay em" if ap == "am" else " pee em"
            return " " + s + " "
        t = re.sub(r"\b(\d{1,2}):(\d{2})\s*([AaPp]\.?[Mm]\.?)?", tm, t)
        t = re.sub(r"\b(1[1-9]\d\d|20\d\d)\b", lambda m: " " + num2words(int(m.group(1)), to="year") + " ", t)
        t = re.sub(r"\d+", lambda m: " " + num2words(int(m.group())) + " ", t)
        t = re.sub(r"[-–—]", " ", t.lower())
        t = re.sub(r"([,.;:!?])", r" \1 ", t)
        t = re.sub(r"[^a-z' ,.;:!?]", " ", t)
        toks = [w for w in t.split() if w in self.PUNCT or re.search("[a-z]", w)]
        return " ".join(toks)

    def ok(self, norm):
        return len([w for w in norm.split() if w not in self.PUNCT]) >= 2

    def word(self, w):
        if w in self.cache: return self.cache[w]
        if w.endswith("'s") and len(w) > 2:
            b = self.word(w[:-2]); l = b[-1] if b else ""
            ph = b + (["IH", "Z"] if l in SIB else ["S"] if l in VLS else ["Z"])
        else:
            w2 = w.strip("'")
            ph = [p for p in self.g2p(w2) if re.fullmatch(r"[A-Z]+\d?", p)] if w2 else []
        self.cache[w] = ph; return ph

    def phones(self, norm):
        toks = norm.split(); groups = None
        if any(t in HETERO for t in toks):   # sentence-level g2p so POS tagging can pick noun/verb variants
            ctx = [t[:-2] if t.endswith("'s") else t for t in toks]
            g = self.g2p(" ".join(ctx)); groups = [[]]
            for p in g:
                if p == " ": groups.append([])
                else: groups[-1].append(p)
            if len(groups) != len(toks): groups = None
        out = []
        for i, t in enumerate(toks):
            if t in self.PUNCT:
                if not out or out[-1] != "sp": out.append("sp")
            elif groups is not None and t in HETERO:
                out += [p for p in groups[i] if re.fullmatch(r"[A-Z]+\d?", p)]
            else: out += self.word(t)
        return out

    def build_dictionary(self, words):
        return {w: self.word(w) for w in sorted(words)}

    def select(self, cfg):
        import soundfile as sf
        c = cfg["data"]; txt = open(c["manifest"], encoding="utf-8").read().strip()
        rows = json.loads(txt) if txt.startswith("[") else [json.loads(l) for l in txt.splitlines() if l.strip()]
        items = []
        for i, r in enumerate(rows):
            text = r.get("text") or r.get("transcript") or r.get("sentence") or ""
            wav = r.get("audio_filepath") or r.get("path") or r.get("audio") or ""
            if not Path(wav).is_absolute(): wav = str(Path(c.get("audio_root", "")) / wav)
            if not Path(wav).exists(): continue
            dur = r.get("duration") or sf.info(wav).duration
            if not (c["min_dur"] <= dur <= c["max_dur"]) or not self.ok(self.normalize(text)): continue
            spk = re.sub(r"\W", "_", str(r.get("speaker_id", r.get("speaker", r.get("spk", "spk0")))))
            items.append(dict(id=f"{spk}_{i:06d}", spk=spk, wav=wav, text=text, dur=float(dur)))
        rng = random.Random(c["seed"]); rng.shuffle(items)
        va, tr, vs, ts = [], [], 0.0, 0.0
        for it in items:
            if vs < c["val_hours"] * 3600: va.append(it); vs += it["dur"]
            elif ts < c["train_hours"] * 3600: tr.append(it); ts += it["dur"]
            else: break
        print(f"[select] candidates={len(items)} train={len(tr)} ({ts/3600:.2f} h) val={len(va)} ({vs/3600:.2f} h)")
        return tr, va
