"""Telugu grapheme front-end, consistent with the AI4Bharat IndicMFA grapheme-to-grapheme dictionary used for alignment.
Alignment corpus convention: LAB = Unicode code points with whitespace and punctuation removed. Model tokens = the dictionary
pronunciation of each code point (== MFA 'phones' tier labels) + 'sp' for silences/pauses. No English phoneme processing."""
import re, unicodedata

ONES = ["సున్నా", "ఒకటి", "రెండు", "మూడు", "నాలుగు", "ఐదు", "ఆరు", "ఏడు", "ఎనిమిది", "తొమ్మిది", "పది", "పదకొండు", "పన్నెండు",
        "పదమూడు", "పద్నాలుగు", "పదిహేను", "పదహారు", "పదిహేడు", "పద్దెనిమిది", "పంతొమ్మిది"]
TENS = {2: "ఇరవై", 3: "ముప్పై", 4: "నలభై", 5: "యాభై", 6: "అరవై", 7: "డెబ్బై", 8: "ఎనభై", 9: "తొంభై"}
DIG = str.maketrans("౦౧౨౩౪౫౬౭౮౯", "0123456789")

def num_te(n):
    if n < 20: return ONES[n]
    if n < 100:
        t, o = divmod(n, 10); return TENS[t] + (" " + ONES[o] if o else "")
    if n < 1000:
        h, r = divmod(n, 100); s = "వంద" if h == 1 else ONES[h] + " వందల"; return s + (" " + num_te(r) if r else "")
    if n < 100000:
        k, r = divmod(n, 1000); s = "వెయ్యి" if k == 1 else num_te(k) + " వేల"; return s + (" " + num_te(r) if r else "")
    return " ".join(ONES[int(c)] for c in str(n))

def ordinal_te(n):
    w = num_te(n); return (w[:-1] if w.endswith("ు") else w) + "వ"

def load_g2g(path):
    d = {}
    for line in open(path, encoding="utf-8"):
        p = line.rstrip("\n").split()
        if len(p) < 2: continue
        k = 0
        while k < 4 and k < len(p) - 2 and re.fullmatch(r"[0-9.eE+-]+", p[1 + k]): k += 1
        d.setdefault(p[0], p[1 + k:])
    return d

class Telugu:
    PUNCT = set(",.;:!?")
    def __init__(self, cfg):
        self.dict = load_g2g(cfg["telugu"]["dictionary"]); self.unknown = set()
        print(f"[telugu] g2g dictionary: {len(self.dict)} entries")

    def normalize(self, t):
        """digits -> Telugu words, drop non-pause punctuation. Spacing/spelling are NOT altered (whitespace is ignored at the token level,
        exactly as in the alignment corpus)."""
        t = t.translate(DIG)
        t = re.sub(r"(\d{1,2}):(\d{2})", lambda m: f" {num_te(int(m[1]))} {'' if m[2] == '00' else num_te(int(m[2]))} ", t)
        t = re.sub(r"(\d+)వ", lambda m: f" {ordinal_te(int(m[1]))} ", t)
        t = re.sub(r"\d+", lambda m: f" {num_te(int(m.group()))} ", t)
        return "".join(c for c in t if c.isspace() or c in self.PUNCT or not unicodedata.category(c).startswith("P"))

    def phones(self, norm):
        out = []; self.unknown = set()
        for ch in norm:
            if ch.isspace(): continue
            if ch in self.PUNCT:
                if out and out[-1] != "sp": out.append("sp")
            elif ch in self.dict: out += self.dict[ch]
            else: self.unknown.add(ch)
        return out
