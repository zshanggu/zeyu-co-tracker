"""Batch pipeline over a folder of demo_* folders: track A, track B, extract mask, compare.

Input layout (suffixes configurable):
    <root>/demo_XXX/demo_XXX_source.mp4      video A
    <root>/demo_XXX/demo_XXX_01.mp4          video B
    <root>/demo_XXX/demo_XXX_instance.mp4    instance render -> mask (selected color(s))

Output layout (mirrors the input):
    <out>/demo_XXX/demo_XXX_source/          tracking result of A (tracks.npy, tracks.mp4, ...)
    <out>/demo_XXX/demo_XXX_01/              tracking result of B
    <out>/demo_XXX/demo_XXX_instance_mask.npy
    <out>/demo_XXX/<first>_vs_<second>/      comparison (compare.mp4, per_frame.csv, ...)
    <out>/demo_XXX/*.log                     one log per step
    <out>/batch_status.json                  progress of every demo / step (read by the GUI)

All settings come from a JSON config (the GUI writes <out>/batch_config.json):
    python scripts/batch_pipeline.py --config outputs/gui/not_reviewed/batch_config.json
See DEFAULTS below for the keys.
"""

import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from extract_mask import extract_mask, parse_colors  # noqa: E402

DEFAULTS = {
    "root": "source_video/not_reviewed",
    "out": "outputs/gui/not_reviewed",
    "a_suffix": "_source",
    "b_suffix": "_01",
    "instance_suffix": "_instance",
    "skip_existing": False,
    "track": {"mode": "offline", "grid_size": "200x100", "gpus": [0], "chunk_size": 0,
              "segment_len": 10, "radius": 2, "frame_stride": 1, "max_frames": 0,
              "backward_tracking": False},
    "mask": {"colors": "167,18,135", "tolerance": 30, "cleanup": True, "open": 3, "close": 5},
    "compare": {"direction": "B vs A", "view": "pairs", "metric": "position", "time": "truncate",
                "radius": 2, "vmax": None, "mask_step": 5, "mask_rule": "both"},
}
STEPS = ("track_a", "track_b", "mask", "compare")


def merged(cfg):
    out = copy.deepcopy(DEFAULTS)
    for k, v in cfg.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        else:
            out[k] = v
    return out


def rel(p):
    try:
        return str(Path(p).resolve().relative_to(REPO))
    except ValueError:
        return str(Path(p).resolve())


class Status:
    """batch_status.json: {started, finished, current, demos: {name: {step: {...}}}}."""

    def __init__(self, path, demos):
        self.path = path
        self.data = {"started": time.time(), "finished": None, "current": None,
                     "demos": {d: {s: {"status": "pending"} for s in STEPS} for d in demos}}
        self.save()

    def set(self, demo, step, status, **extra):
        self.data["demos"][demo][step] = {"status": status, **extra}
        self.data["current"] = f"{demo}: {step}" if status == "running" else self.data["current"]
        self.save()

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1))
        tmp.replace(self.path)


def run(cmd, log_path):
    """Run a step's command; output goes to its own log and is echoed to the batch log."""
    with open(log_path, "w") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        rc = subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT).returncode
    tail = Path(log_path).read_text(errors="replace").replace("\r", "\n").splitlines()[-3:]
    return rc, "\n".join(tail)


def track_cmd(video, save_dir, t):
    cmd = [sys.executable, "scripts/track_video.py", "--video", rel(video), "--save_dir", rel(save_dir),
           "--mode", t["mode"], "--grid_size", str(t["grid_size"]), "--radius", str(t["radius"]),
           "--chunk_size", str(t["chunk_size"]), "--segment_len", str(t["segment_len"]),
           "--frame_stride", str(t["frame_stride"]), "--max_frames", str(t["max_frames"])]
    if t.get("gpus"):
        cmd += ["--gpus", ",".join(str(g) for g in t["gpus"])]
    if t.get("backward_tracking"):
        cmd.append("--backward_tracking")
    return cmd


def process(demo_dir, out_dir, cfg, status):
    name = demo_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    va = demo_dir / f"{name}{cfg['a_suffix']}.mp4"
    vb = demo_dir / f"{name}{cfg['b_suffix']}.mp4"
    vi = demo_dir / f"{name}{cfg['instance_suffix']}.mp4"
    skip = cfg["skip_existing"]
    results = {}

    for step, video in (("track_a", va), ("track_b", vb)):
        save = out_dir / video.stem
        results[step] = save
        if not video.is_file():
            status.set(name, step, "missing", note=f"no {video.name}")
            continue
        if skip and (save / "tracks.npy").is_file():
            status.set(name, step, "skipped", dir=rel(save), note="already done")
            continue
        status.set(name, step, "running")
        t0 = time.time()
        rc, tail = run(track_cmd(video, save, cfg["track"]), out_dir / f"{step}.log")
        ok = rc == 0 and (save / "tracks.npy").is_file()
        status.set(name, step, "done" if ok else "failed", dir=rel(save),
                   seconds=round(time.time() - t0, 1), **({} if ok else {"error": tail}))

    mask_path = out_dir / f"{vi.stem}_mask.npy"
    if not vi.is_file():
        status.set(name, "mask", "missing", note=f"no {vi.name}; compare runs without a mask")
        mask_path = None
    elif skip and mask_path.is_file():
        status.set(name, "mask", "skipped", path=rel(mask_path), note="already done")
    else:
        status.set(name, "mask", "running")
        t0 = time.time()
        m = cfg["mask"]
        try:
            stack = extract_mask(vi, parse_colors(m["colors"]), float(m["tolerance"]), bool(m["cleanup"]),
                                 int(m["open"]), int(m["close"]))
            np.save(mask_path, stack)
            found = int((stack.sum(axis=(1, 2)) > 0).sum())
            status.set(name, "mask", "done", path=rel(mask_path), seconds=round(time.time() - t0, 1),
                       note=f"{stack.shape}, non-empty frames {found}/{len(stack)}")
        except Exception as e:  # noqa: BLE001 - report and continue with the next step
            status.set(name, "mask", "failed", error=str(e))
            mask_path = None

    ta, tb = results["track_a"], results["track_b"]
    if not ((ta / "tracks.npy").is_file() and (tb / "tracks.npy").is_file()):
        status.set(name, "compare", "missing", note="needs both tracking results")
        return
    c = cfg["compare"]
    first, second = (tb, ta) if c["direction"] == "B vs A" else (ta, tb)
    cdir = out_dir / f"{first.name}_vs_{second.name}"
    if skip and (cdir / "summary.json").is_file():
        status.set(name, "compare", "skipped", dir=rel(cdir), note="already done")
        return
    cmd = [sys.executable, "scripts/compare_tracks.py", "--a", rel(first), "--b", rel(second),
           "--out", rel(cdir), "--view", c["view"], "--metric", c["metric"], "--time", c["time"],
           "--radius", str(c["radius"])]
    if c.get("vmax"):
        cmd += ["--vmax", str(c["vmax"])]
    if mask_path:
        cmd += ["--mask", rel(mask_path), "--mask_step", str(c["mask_step"]), "--mask_rule", c["mask_rule"]]
    status.set(name, "compare", "running")
    t0 = time.time()
    rc, tail = run(cmd, out_dir / "compare.log")
    ok = rc == 0 and (cdir / "compare.mp4").is_file()
    extra = {"error": tail} if not ok else {"note": "with mask" if mask_path else "no mask"}
    status.set(name, "compare", "done" if ok else "failed", dir=rel(cdir), names=[first.name, second.name],
               seconds=round(time.time() - t0, 1), **extra)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="JSON config (see DEFAULTS in this file)")
    args = p.parse_args()
    cfg = merged(json.load(open(args.config)))
    root, out = (REPO / cfg["root"]).resolve(), (REPO / cfg["out"]).resolve()
    if not root.is_dir():
        raise SystemExit(f"root folder not found: {root}")
    demos = sorted(d for d in root.iterdir() if d.is_dir() and d.name.startswith("demo_"))
    if not demos:
        raise SystemExit(f"no demo_* folders under {root}")
    out.mkdir(parents=True, exist_ok=True)
    status = Status(out / "batch_status.json", [d.name for d in demos])
    print(f"batch: {len(demos)} demos from {rel(root)} -> {rel(out)}", flush=True)
    for i, d in enumerate(demos, 1):
        print(f"[{i}/{len(demos)}] {d.name}", flush=True)
        process(d, out / d.name, cfg, status)
    status.data["finished"] = time.time()
    status.data["current"] = None
    status.save()
    counts = {}
    for steps in status.data["demos"].values():
        for st in steps.values():
            counts[st["status"]] = counts.get(st["status"], 0) + 1
    print("batch finished:", ", ".join(f"{v} {k}" for k, v in sorted(counts.items())), flush=True)


if __name__ == "__main__":
    main()
