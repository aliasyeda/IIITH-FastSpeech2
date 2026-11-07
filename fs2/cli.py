import argparse, json
from pathlib import Path
from . import pipeline

def _cfg(lang_dir, a):
    c = pipeline.load_cfg(a.config or Path(lang_dir) / "config.yaml"); return c

def preprocess_main(lang_dir):
    ap = argparse.ArgumentParser(); ap.add_argument("--config"); ap.add_argument("--stage", default="all", choices=["prepare", "align", "features", "all"])
    a = ap.parse_args(); cfg = _cfg(lang_dir, a)
    todo = ["prepare", "align", "features"] if a.stage == "all" else [a.stage]
    if cfg["language"] == "telugu": todo = [t for t in todo if t != "align"]   # Telugu alignment already done
    for st in todo: getattr(pipeline, st)(cfg)

def train_main(lang_dir):
    from .train_lib import train
    ap = argparse.ArgumentParser(); ap.add_argument("--config"); a = ap.parse_args(); cfg = _cfg(lang_dir, a)
    train(cfg, lang_dir, (Path(lang_dir) / cfg["paths"]["results_dir"]).resolve())

def inference_main(lang_dir):
    from .infer_lib import infer
    ap = argparse.ArgumentParser(); ap.add_argument("--config"); ap.add_argument("--ckpt"); a = ap.parse_args(); cfg = _cfg(lang_dir, a)
    sents = json.load(open(Path(lang_dir) / cfg["inference"]["sentences"], encoding="utf-8"))
    infer(cfg, cfg["language"], a.ckpt or Path(lang_dir) / "checkpoints" / "best.pt", sents, (Path(lang_dir) / cfg["paths"]["results_dir"]).resolve())
