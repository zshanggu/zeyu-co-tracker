"""Run CoTracker3 on a video and save tracks + a visualization.

Examples:
    # 20x20 grid on frame 0, offline model
    python scripts/track_video.py --video assets/apple.mp4 --grid_size 20

    # 100 columns x 50 rows, tiny dots
    python scripts/track_video.py --video assets/apple.mp4 --grid_size 100x50 --radius 1

    # Grid restricted to a segmentation mask, tracked both directions from frame 10
    python scripts/track_video.py --video v.mp4 --grid_size 30 --mask m.png \
        --grid_query_frame 10 --backward_tracking

    # Specific points: text file with one "t x y" per line (frame idx, pixel coords)
    python scripts/track_video.py --video v.mp4 --queries pts.txt

    # Long video: sliding-window online model (constant memory)
    python scripts/track_video.py --video long.mp4 --mode online --grid_size 20

    # OOM: split the points over 4 GPUs, at most 500 points per forward pass
    python scripts/track_video.py --video v.mp4 --grid_size 60 --gpus 0,1,2,3 --chunk_size 500

Outputs (in --out_dir/<video name>/):
    tracks.npy      float32 (T, N, 2)  pixel (x, y) per frame, original resolution
    visibility.npy  bool    (T, N)     whether each point is visible in each frame
    queries.npy     float32 (N, 3)     (t, x, y) query point of each track
    tracks.mp4      visualization
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image

from cotracker.predictor import CoTrackerOnlinePredictor, CoTrackerPredictor
from cotracker.utils.visualizer import Visualizer


def load_video(path, max_frames, stride):
    frames = iio.imread(path, plugin="FFMPEG")  # (T, H, W, 3) uint8
    frames = frames[::stride]
    if max_frames > 0:
        frames = frames[:max_frames]
    return frames


def parse_grid(spec):
    """'30' -> (30, 30); '100x50' -> (100 columns, 50 rows)."""
    parts = spec.lower().split("x")
    cols, rows = (int(parts[0]),) * 2 if len(parts) == 1 else map(int, parts)
    return cols, rows


def grid_points(cols, rows, H, W):
    """(rows*cols, 2) row-major (x, y) grid, same layout as CoTracker's get_points_on_a_grid:
    built at model resolution (384x512) with a W/64 margin, then rescaled to (H, W)."""
    mh, mw = 384, 512
    m = mw / 64
    ys = torch.linspace(m, mh - m, rows) if rows > 1 else torch.tensor([mh / 2])
    xs = torch.linspace(m, mw - m, cols) if cols > 1 else torch.tensor([mw / 2])
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    xy = torch.stack([gx, gy], dim=-1).reshape(-1, 2)
    return xy * torch.tensor([(W - 1) / (mw - 1), (H - 1) / (mh - 1)])


def build_queries(args, T, H, W):
    """All query points as a (N, 3) CPU tensor of (t, x, y) in original pixel coords."""
    if args.queries:
        q = np.loadtxt(args.queries, dtype=np.float32).reshape(-1, 3)
        return torch.from_numpy(q)
    xy = grid_points(*parse_grid(args.grid_size), H, W)
    if args.mask:
        m = np.array(Image.open(args.mask).convert("L").resize((W, H), Image.NEAREST)) > 0
        ix = xy[:, 0].round().long().clamp(0, W - 1)
        iy = xy[:, 1].round().long().clamp(0, H - 1)
        xy = xy[torch.from_numpy(m)[iy, ix]]
    t = torch.full((len(xy), 1), float(args.grid_query_frame))
    return torch.cat([t, xy], dim=1)


def run_offline(model, video, queries, args):
    return model(video, queries=queries, backward_tracking=args.backward_tracking)


def run_online(model, video, queries, args):
    # Same loop as the README: overlapping windows of 2*step frames, advancing by step.
    # The last (shorter) chunk is padded inside the model, so the tail is covered.
    T = video.shape[1]
    step = model.step
    model(video_chunk=video, is_first_step=True, queries=queries)
    tracks = vis = None
    for ind in range(0, T - step, step):
        tracks, vis = model(video_chunk=video[:, ind : ind + step * 2])
    return tracks[:, :T], vis[:, :T]


def gpu_worker(device, frames, chunks, ckpt, args):
    """Track each chunk of queries on one device; returns [(chunk_idx, tracks, vis)]."""
    if args.mode == "offline":
        model = CoTrackerPredictor(checkpoint=ckpt, offline=True, window_len=60).to(device)
        run = run_offline
    else:
        model = CoTrackerOnlinePredictor(checkpoint=ckpt, window_len=16).to(device)
        run = run_online
    video = torch.from_numpy(frames).to(device).permute(0, 3, 1, 2)[None].float()  # B T C H W
    results = []
    for idx, q in chunks:
        tracks, vis = run(model, video, q[None].to(device), args)
        results.append((idx, tracks[0].cpu(), vis[0].cpu()))
        print(f"  [{device}] chunk {idx}: {len(q)} points done", flush=True)
    del model, video
    torch.cuda.empty_cache()
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--mode", choices=["offline", "online"], default="offline")
    p.add_argument("--checkpoint", default=None, help="defaults to checkpoints/scaled_<mode>.pth")
    p.add_argument(
        "--grid_size", default="20", help='"N" for an NxN grid, or "COLSxROWS" e.g. 100x50'
    )
    p.add_argument("--grid_query_frame", type=int, default=0)
    p.add_argument("--queries", default=None, help='text file of "t x y" rows; overrides grid')
    p.add_argument("--mask", default=None, help="binary mask image; keeps grid points inside it")
    p.add_argument("--backward_tracking", action="store_true", help="offline only")
    p.add_argument("--max_frames", type=int, default=0, help="0 = all frames")
    p.add_argument("--frame_stride", type=int, default=1)
    p.add_argument("--gpus", default="all", help='"all" or comma list of visible GPU ids, e.g. 0,1')
    p.add_argument(
        "--chunk_size", type=int, default=0,
        help="max points per forward pass (0 = split points evenly over GPUs). Lower it on OOM.",
    )
    p.add_argument("--out_dir", default="./outputs")
    p.add_argument("--fps", type=int, default=10, help="visualization fps")
    p.add_argument("--radius", type=int, default=4, help="visualization dot radius in px (0 = 1 pixel)")
    p.add_argument("--no_vis", action="store_true")
    args = p.parse_args()

    if args.mode == "online" and args.backward_tracking:
        print("warning: --backward_tracking is ignored in online mode")
    ckpt = args.checkpoint or f"./checkpoints/scaled_{args.mode}.pth"
    if not torch.cuda.is_available():
        devices = ["cpu"]
    elif args.gpus == "all":
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    else:
        devices = [f"cuda:{i}" for i in args.gpus.split(",")]

    frames = load_video(args.video, args.max_frames, args.frame_stride)
    T, H, W, _ = frames.shape
    queries = build_queries(args, T, H, W)
    N = len(queries)
    chunk_size = args.chunk_size or -(-N // len(devices))
    chunks = list(enumerate(queries.split(chunk_size)))
    print(
        f"video: {T} frames, {W}x{H}, mode={args.mode}, {N} points in "
        f"{len(chunks)} chunk(s) of <= {chunk_size} on {devices}"
    )

    # Round-robin chunks over devices; one thread per device (CUDA work releases the GIL).
    per_device = [chunks[i :: len(devices)] for i in range(len(devices))]
    with ThreadPoolExecutor(len(devices)) as ex:
        futures = [
            ex.submit(gpu_worker, d, frames, c, ckpt, args)
            for d, c in zip(devices, per_device) if c
        ]
        results = sorted(r for f in futures for r in f.result())
    tracks = torch.cat([r[1] for r in results], dim=1)  # (T, N, 2)
    vis = torch.cat([r[2] for r in results], dim=1)  # (T, N)

    name = os.path.splitext(os.path.basename(args.video))[0]
    out = os.path.join(args.out_dir, name)
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "tracks.npy"), tracks.numpy().astype(np.float32))
    np.save(os.path.join(out, "visibility.npy"), vis.numpy().astype(bool))
    np.save(os.path.join(out, "queries.npy"), queries.numpy())
    print(f"tracks {tuple(tracks.shape)}, visible fraction {vis.float().mean():.2f} -> {out}")

    if not args.no_vis:
        query_frame = 0 if args.backward_tracking else args.grid_query_frame
        video = torch.from_numpy(frames).permute(0, 3, 1, 2)[None].float()
        Visualizer(save_dir=out, pad_value=0, point_radius=args.radius, fps=args.fps).visualize(
            video, tracks[None], vis[None], filename="tracks", query_frame=query_frame
        )


if __name__ == "__main__":
    main()
