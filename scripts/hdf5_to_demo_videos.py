"""Lay out a SoftMimicGen hdf5 (and optionally a folder of generated videos) as one folder per demo, the layout the
CoTracker GUI batch mode and scripts/vlm_check.py read.

For an hdf5 at <dir>/<name>.hdf5 it writes <dir>/<name>/demo_XXX/ (hdf5 demo_N -> demo_XXX with XXX = N, 3 digits):
  demo_XXX_source.mp4     room | wrist RGB (agentview_image | robot0_eye_in_hand_image), every step
  demo_XXX_instance.mp4   room | wrist instance segmentation, one fixed color per object, lossless
  demo_XXX_instance_ids.json  this demo's raw hdf5 ids -> object, run id and color, per camera
  demo_XXX_01.mp4         copied from --videos when given (any demo_XXX_<suffix>.mp4 found there, recursively)
and <dir>/<name>/instance_colors.json, the run-wide table object -> id and color.

Isaac Lab numbers the instances differently in every demo (Jenga is id 16 in demo_0 and 21 in demo_7), so the raw
ids are mapped through each demo's "instance_ids" attribute (raw id -> prim path) to the object name (the prim path
without /World/envs/env_N/). Every object gets one id and one color for the whole run, so a single color selects the
same object in every demo. The colors come from a hash of the object name, so the same object also keeps its color
in other runs, and are kept at least MIN_COLOR_DIST apart (sum of |dR|+|dG|+|dB|). The deformable object
(Object/skin, the towel) stays (167, 18, 135), the color the mask and VLM defaults expect. Demos without the
attribute fall back to one color per raw id (the SoftMimicGen colors).

Videos play at the control rate of the hdf5 (1 / (dt * decimation) from env_args, 20 fps when missing), one frame
per step. Existing files are skipped unless --overwrite (everything) or --overwrite_instance (only the instance
videos), so the script can be re-run.

    python scripts/hdf5_to_demo_videos.py \
        --hdf5 /data/zeyu/PHD_LAB/Deformable/hdf5/simulation_from_eric/franka_towel_zeyu_n50_seeds23.hdf5 \
        --videos /data/zeyu/PHD_LAB/Deformable/hdf5/franka_towel_digital_twin_n50_seeds23_videos
"""

import argparse
import hashlib
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
CAMERAS = {ROOM: f"{ROOM}_image", WRIST: f"{WRIST}_image"}  # obs prefix -> key in the "instance_ids" attribute
PINNED_COLORS = {"Object/skin": (167, 18, 135)}  # the deformable object (towel) keeps its usual color
MIN_COLOR_DIST = 120  # min |dR|+|dG|+|dB| between two object colors (masks use a tolerance of about 30-40)
MIN_BRIGHTNESS = 150  # min R+G+B of an object color, so no object is confused with the black background
ENV_PREFIX = re.compile(r"^/World/envs/env_\d+/")


def object_name(prim_path: str) -> str:
    """Prim path -> object name shared by all demos: '/World/envs/env_0/Jenga/geometry/mesh' -> 'Jenga/geometry/mesh'."""
    return ENV_PREFIX.sub("", prim_path)


def demo_id_maps(group: h5py.Group) -> dict[str, dict[int, str]] | None:
    """camera obs prefix -> {raw id: object name} from the demo's "instance_ids" attribute (None when missing)."""
    if "instance_ids" not in group.attrs:
        return None
    raw = json.loads(group.attrs["instance_ids"])
    return {cam: {int(i): object_name(p) for i, p in raw.get(key, {}).items()} for cam, key in CAMERAS.items()}


def name_color(name: str, taken: list[tuple[int, int, int]]) -> tuple[int, int, int]:
    """A color from a hash of the name, rehashed with a salt until it is bright enough and far from `taken`."""
    for salt in range(10000):
        r, g, b = hashlib.md5(f"{name}#{salt}".encode()).digest()[:3]
        if r + g + b >= MIN_BRIGHTNESS and all(abs(r - t[0]) + abs(g - t[1]) + abs(b - t[2]) >= MIN_COLOR_DIST
                                               for t in taken):
            return (r, g, b)
    raise SystemExit(f"no free color for {name!r}; lower MIN_COLOR_DIST")


def run_table(names: set[str]) -> dict[str, dict]:
    """Object name -> {"id": 1.., "color": [r, g, b]} for the whole run. Ids follow the sorted names; colors are the
    pinned ones first, then name hashes in sorted order."""
    real = sorted(n for n in names if n != "INVALID")
    colors = {n: PINNED_COLORS[n] for n in real if n in PINNED_COLORS}
    for n in real:
        if n not in colors:
            colors[n] = name_color(n, list(colors.values()))
    return {n: {"id": i, "color": list(colors[n])} for i, n in enumerate(real, start=1)}


def lut_for(id_map: dict[int, str], table: dict[str, dict]) -> np.ndarray:
    """Raw id -> RGB lookup table (unknown, INVALID and id 0 stay black)."""
    lut = np.zeros((max(id_map, default=0) + 1, 3), np.uint8)
    for raw, name in id_map.items():
        if name in table:
            lut[raw] = table[name]["color"]
    return lut


def lut_color(lut: np.ndarray):
    """(..., H, W, 1) raw ids -> RGB through `lut`; ids outside the table become black."""
    def color(ids: np.ndarray) -> np.ndarray:
        x = ids[..., 0]
        return np.where((x >= 0)[..., None] & (x < len(lut))[..., None], lut[np.clip(x, 0, len(lut) - 1)], 0
                        ).astype(np.uint8)
    return color


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


def write_tiled(path: Path, left: h5py.Dataset, right: h5py.Dataset, fps: float, params: dict,
                color=None, color_right=None) -> None:
    """Write left | right frames of two (T, H, W, C) datasets, streamed in chunks; tmp file then rename.
    `color` maps the left frames (and the right ones unless `color_right` is given)."""
    tmp = path.with_name(f".{path.name}.tmp.mp4")
    color_right = color_right or color
    with imageio.get_writer(tmp, fps=fps, **params) as w:
        for s in range(0, left.shape[0], CHUNK):
            a, b = left[s:s + CHUNK], right[s:s + CHUNK]
            if color is not None:
                a, b = color(a), color_right(b)
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
    ap.add_argument("--overwrite", action="store_true", help="rewrite every file")
    ap.add_argument("--overwrite_instance", action="store_true", help="rewrite only the instance videos")
    args = ap.parse_args()

    out_root = args.out or args.hdf5.with_suffix("")
    videos = find_videos(args.videos, args.suffix) if args.videos else {}
    if args.videos:
        print(f"{len(videos)} videos demo_*_{args.suffix}.mp4 in {args.videos}")

    with h5py.File(args.hdf5, "r") as h5:
        fps = control_fps(h5)
        demos = sorted(h5["data"], key=lambda k: int(k.split("_")[1]))[: args.max_demos]
        print(f"{len(demos)} demos in {args.hdf5} -> {out_root} ({fps:g} fps)")

        # One id and color per object for the whole run, from every demo's raw id -> prim path map (all demos of the
        # file, also with --max_demos, so a quick test gets the same colors as the full run).
        maps = {key: demo_id_maps(h5["data"][key]) for key in h5["data"]}
        table = run_table({n for m in maps.values() if m for cm in m.values() for n in cm.values()})
        if table:
            out_root.mkdir(parents=True, exist_ok=True)
            (out_root / "instance_colors.json").write_text(json.dumps(
                {"note": "object -> run-wide id and RGB color in demo_XXX_instance.mp4 (black = background/invalid)",
                 "objects": table}, indent=1))
            print(f"{len(table)} objects, colors in {out_root / 'instance_colors.json'}:")
            for n, v in table.items():
                print(f"  {v['id']:>3} {tuple(v['color'])!s:17} {n}")
        missing = [k for k in demos if maps[k] is None]
        if missing:
            print(f"no instance_ids attribute in {len(missing)} demos (colored by raw id): {missing[:10]}")

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
            elif args.overwrite or args.overwrite_instance or not inst.exists():
                m = maps[key]
                if m is None:
                    write_tiled(inst, obs[f"{ROOM}_instance_raw"], obs[f"{WRIST}_instance_raw"], fps, LOSSLESS_RGB,
                                color=instance_to_rgb)
                    done.append("instance (raw id colors)")
                else:
                    write_tiled(inst, obs[f"{ROOM}_instance_raw"], obs[f"{WRIST}_instance_raw"], fps, LOSSLESS_RGB,
                                color=lut_color(lut_for(m[ROOM], table)),
                                color_right=lut_color(lut_for(m[WRIST], table)))
                    (d / f"{name}_instance_ids.json").write_text(json.dumps(
                        {CAMERAS[cam]: {str(raw): {"object": n, **table.get(n, {"id": 0, "color": [0, 0, 0]})}
                                        for raw, n in sorted(m[cam].items())} for cam in CAMERAS}, indent=1))
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
