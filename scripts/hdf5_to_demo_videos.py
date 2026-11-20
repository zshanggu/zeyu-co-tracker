"""Lay out a SoftMimicGen hdf5 (and optionally a folder of generated videos) as one folder per demo, the layout the
CoTracker GUI batch mode and scripts/vlm_check.py read.

For an hdf5 at <dir>/<name>.hdf5 it writes <dir>/<name>/demo_XXX/ (hdf5 demo_N -> demo_XXX with XXX = N, 3 digits):
  demo_XXX_source.mp4     room | wrist RGB (agentview_image | robot0_eye_in_hand_image), every step
  demo_XXX_instance.mp4   room | wrist instance ids, one fixed color per id (the SoftMimicGen colors, so the towel
                          is (167, 18, 135)), lossless
  demo_XXX_01.mp4         copied from --videos when given (any demo_XXX_<suffix>.mp4 found there, recursively)

Videos play at the control rate of the hdf5 (1 / (dt * decimation) from env_args, 20 fps when missing), one frame
per step. Existing files are skipped unless --overwrite, so the script can be re-run.

    python scripts/hdf5_to_demo_videos.py \
        --hdf5 /data/zeyu/PHD_LAB/Deformable/hdf5/simulation_from_eric/franka_towel_zeyu_n50_seeds23.hdf5 \
        --videos /data/zeyu/PHD_LAB/Deformable/hdf5/franka_towel_digital_twin_n50_seeds23_videos
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np

ROOM, WRIST = "agentview", "robot0_eye_in_hand"
CHUNK = 64  # steps read from the hdf5 at a time
SOURCE_PARAMS = dict(codec="libx264", quality=9, macro_block_size=1)
LOSSLESS_RGB = dict(codec="libx264rgb", pixelformat="rgb24", output_params=["-qp", "0"], macro_block_size=1)


def instance_to_rgb(ids: np.ndarray) -> np.ndarray:
    """(..., H, W, 1) int ids -> one fixed color per id, id 0 black. Same hash as SoftMimicGen
    experiments/policy_data/make_videos.py, so the colors match its instance videos."""
    x = ids[..., 0].astype(np.int64)
    rgb = np.stack([((x * k) >> 8) & 255 for k in (2654435761, 2246822519, 3266489917)], axis=-1).astype(np.uint8)
    rgb[x == 0] = 0
    return rgb


def control_fps(h5: h5py.File, fallback: float = 20.0) -> float:
    try:
        sim = json.loads(h5["data"].attrs["env_args"])["sim_args"]
        return 1.0 / (sim["dt"] * sim["decimation"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return fallback


def write_tiled(path: Path, left: h5py.Dataset, right: h5py.Dataset, fps: float, params: dict, color=None) -> None:
    """Write left | right frames of two (T, H, W, C) datasets, streamed in chunks; tmp file then rename."""
    tmp = path.with_name(f".{path.name}.tmp.mp4")
    with imageio.get_writer(tmp, fps=fps, **params) as w:
        for s in range(0, left.shape[0], CHUNK):
            a, b = left[s:s + CHUNK], right[s:s + CHUNK]
            if color is not None:
                a, b = color(a), color(b)
            for frame in np.concatenate([a, b], axis=2):
                w.append_data(frame)
    tmp.replace(path)


def find_videos(folder: Path, suffix: str) -> dict[int, Path]:
    """demo number -> demo_XXX_<suffix>.mp4 under folder (recursive)."""
    pattern = re.compile(rf"demo_(\d+)_{re.escape(suffix)}\.mp4")
    found = {}
    for p in sorted(folder.rglob(f"demo_*_{suffix}.mp4")):
        m = pattern.fullmatch(p.name)
        if m:
            idx = int(m.group(1))
            if idx in found:
                raise SystemExit(f"two videos for demo {idx}: {found[idx]} and {p}")
            found[idx] = p
    return found


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hdf5", type=Path, required=True)
    ap.add_argument("--videos", type=Path, help="folder with demo_XXX_<suffix>.mp4 to copy into the demo folders")
    ap.add_argument("--suffix", default="01", help="suffix of the --videos files (default 01)")
    ap.add_argument("--out", type=Path, help="output root (default: the hdf5 path without .hdf5)")
    ap.add_argument("--max_demos", type=int, help="only the first N demos (quick test)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_root = args.out or args.hdf5.with_suffix("")
    videos = find_videos(args.videos, args.suffix) if args.videos else {}
    if args.videos:
        print(f"{len(videos)} videos demo_*_{args.suffix}.mp4 in {args.videos}")

    with h5py.File(args.hdf5, "r") as h5:
        fps = control_fps(h5)
        demos = sorted(h5["data"], key=lambda k: int(k.split("_")[1]))[: args.max_demos]
        print(f"{len(demos)} demos in {args.hdf5} -> {out_root} ({fps:g} fps)")
        for key in demos:
            idx = int(key.split("_")[1])
            name = f"demo_{idx:03d}"
            d = out_root / name
            d.mkdir(parents=True, exist_ok=True)
            obs = h5["data"][key]["obs"]
            steps = obs[f"{ROOM}_image"].shape[0]
            done = []

            src = d / f"{name}_source.mp4"
            if args.overwrite or not src.exists():
                write_tiled(src, obs[f"{ROOM}_image"], obs[f"{WRIST}_image"], fps, SOURCE_PARAMS)
                done.append("source")

            inst = d / f"{name}_instance.mp4"
            if f"{ROOM}_instance_raw" not in obs:
                done.append("no instance ids in hdf5")
            elif args.overwrite or not inst.exists():
                write_tiled(inst, obs[f"{ROOM}_instance_raw"], obs[f"{WRIST}_instance_raw"], fps, LOSSLESS_RGB,
                            color=instance_to_rgb)
                done.append("instance")

            if args.videos:
                v = videos.get(idx)
                dst = d / f"{name}_{args.suffix}.mp4"
                if v is None:
                    done.append(f"no {name}_{args.suffix}.mp4")
                elif args.overwrite or not dst.exists():
                    shutil.copy2(v, dst)
                    n = imageio.get_reader(dst).count_frames()
                    done.append(f"{args.suffix}" + ("" if n == steps else f" ({n} frames vs {steps} steps!)"))

            print(f"{name}: {steps} steps, " + (", ".join(done) or "all present, skipped"), flush=True)

    extra = sorted(set(videos) - {int(k.split("_")[1]) for k in demos})
    if extra and not args.max_demos:
        print(f"videos with no hdf5 demo (not copied): {extra}")


if __name__ == "__main__":
    main()
