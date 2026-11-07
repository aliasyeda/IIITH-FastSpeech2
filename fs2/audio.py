import numpy as np
from functools import lru_cache

@lru_cache(maxsize=4)
def _mb(sr, n_fft, n_mels, fmin, fmax):
    import librosa
    return librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)

def load_wav(path, sr):
    import librosa
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y

def mel_energy(y, a):
    """ln-magnitude mel (HiFi-GAN compatible, clip 1e-5) -> (T,80); frame energy = L2 norm of STFT magnitude."""
    import librosa
    S = np.abs(librosa.stft(y, n_fft=a["n_fft"], hop_length=a["hop"], win_length=a["win"]))
    mel = np.log(np.clip(_mb(a["sr"], a["n_fft"], a["n_mels"], a["fmin"], a["fmax"]) @ S, 1e-5, None))
    return mel.T.astype(np.float32), np.linalg.norm(S, axis=0).astype(np.float32)

def f0_track(y, a):
    try:
        import pyworld as pw
    except ImportError:   # slow fallback
        import librosa
        f, vf, _ = librosa.pyin(y, fmin=60, fmax=600, sr=a["sr"], frame_length=2048, hop_length=a["hop"])
        return np.nan_to_num(np.where(vf, f, 0.0)).astype(np.float32)
    x = y.astype(np.float64)
    f, t = pw.dio(x, a["sr"], frame_period=a["hop"] / a["sr"] * 1000)
    return pw.stonemask(x, f, t, a["sr"]).astype(np.float32)

def fit(v, n):
    if len(v) >= n: return v[:n]
    if len(v) == 0: return np.zeros((n,) + v.shape[1:], v.dtype)
    return np.pad(v, [(0, n - len(v))] + [(0, 0)] * (v.ndim - 1), mode="edge")

def pitch_avg(f0, dur):
    """phoneme-level mean log-F0 over voiced frames; 0 = unvoiced phoneme."""
    out, pos = [], 0
    for d in dur:
        s = f0[pos:pos + d]; pos += d; s = s[s > 0]
        out.append(float(np.log(s).mean()) if len(s) else 0.0)
    return np.array(out, np.float32)

def mean_avg(v, dur):
    out, pos = [], 0
    for d in dur:
        s = v[pos:pos + d]; pos += d
        out.append(float(s.mean()) if len(s) else 0.0)
    return np.array(out, np.float32)
