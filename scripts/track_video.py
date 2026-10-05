"""Run CoTracker3 on a video and save tracks + a visualization.

Examples:
    # 20x20 grid on frame 0, offline model
    python scripts/track_video.py --video assets/apple.mp4 --grid_size 20

    # Grid restricted to a segmentation mask, tracked both directions from frame 10
    python scripts/track_video.py --video v.mp4 --grid_size 30 --mask m.png \
        --grid_query_frame 10 --backward_tracking

    # Specific points: text file with one "t x y" per line (frame idx, pixel coords)
    python scripts/track_video.py --video v.mp4 --queries pts.txt

    # Long video: sliding-window online model (constant memory)
    python scripts/track_video.py --video long.mp4 --mode online --grid_size 20

Outputs (in --out_dir/<video name>/):
    tracks.npy      float32 (T, N, 2)  pixel (x, y) per frame, original resolution
    visibility.npy  bool    (T, N)     whether each point is visible in each frame
    queries.npy     float32 (N, 3)     (t, x, y) queries used (only for --queries)
    tracks.mp4      visualization
"""

import argparse
import os

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


def run_offline(model, video, queries, mask, args):
    return model(
        video,
        queries=queries,
        segm_mask=mask,
        grid_size=args.grid_size if queries is None else 0,
        grid_query_frame=args.grid_query_frame,
        backward_tracking=args.backward_tracking,
    )


def run_online(model, video, queries, args):
    # Same loop as the README: overlapping windows of 2*step frames, advancing by step.
    # The last (shorter) chunk is padded inside the model, so the tail is covered.
    T = video.shape[1]
    step = model.step
    model(
        video_chunk=video,
        is_first_step=True,
        queries=queries,
        grid_size=args.grid_size,
        grid_query_frame=args.grid_query_frame,
    )
    tracks = vis = None
    for ind in range(0, T - step, step):
        tracks, vis = model(video_chunk=video[:, ind : ind + step * 2])
    return tracks[:, :T], vis[:, :T]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--mode", choices=["offline", "online"], default="offline")
    p.add_argument("--checkpoint", default=None, help="defaults to checkpoints/scaled_<mode>.pth")
    p.add_argument("--grid_size", type=int, default=20, help="N x N grid of query points")
    p.add_argument("--grid_query_frame", type=int, default=0)
    p.add_argument("--queries", default=None, help='text file of "t x y" rows; overrides grid')
    p.add_argument("--mask", default=None, help="binary mask image (offline + grid only)")
    p.add_argument("--backward_tracking", action="store_true", help="offline only")
    p.add_argument("--max_frames", type=int, default=0, help="0 = all frames")
    p.add_argument("--frame_stride", type=int, default=1)
    p.add_argument("--out_dir", default="./outputs")
    p.add_argument("--fps", type=int, default=10, help="visualization fps")
    p.add_argument("--no_vis", action="store_true")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = args.checkpoint or f"./checkpoints/scaled_{args.mode}.pth"

    frames = load_video(args.video, args.max_frames, args.frame_stride)
    T, H, W, _ = frames.shape
    print(f"video: {T} frames, {W}x{H}, device={device}, mode={args.mode}")
    video = torch.from_numpy(frames).permute(0, 3, 1, 2)[None].float().to(device)  # B T C H W

    queries = None
    if args.queries:
        q = np.loadtxt(args.queries, dtype=np.float32).reshape(-1, 3)
        queries = torch.from_numpy(q)[None].to(device)  # B N 3 (t, x, y)

    mask = None
    if args.mask:
        m = np.array(Image.open(args.mask).convert("L")) > 0
        mask = torch.from_numpy(m).float()[None, None].to(device)  # B 1 H W

    if args.mode == "offline":
        model = CoTrackerPredictor(checkpoint=ckpt, offline=True, window_len=60).to(device)
        tracks, vis = run_offline(model, video, queries, mask, args)
    else:
        if mask is not None or args.backward_tracking:
            print("warning: --mask/--backward_tracking are ignored in online mode")
        model = CoTrackerOnlinePredictor(checkpoint=ckpt, window_len=16).to(device)
        tracks, vis = run_online(model, video, queries, args)

    # tracks: (1, T, N, 2), vis: (1, T, N) bool
    name = os.path.splitext(os.path.basename(args.video))[0]
    out = os.path.join(args.out_dir, name)
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "tracks.npy"), tracks[0].cpu().numpy().astype(np.float32))
    np.save(os.path.join(out, "visibility.npy"), vis[0].cpu().numpy().astype(bool))
    if queries is not None:
        np.save(os.path.join(out, "queries.npy"), queries[0].cpu().numpy())
    print(f"tracks {tuple(tracks[0].shape)}, visible fraction {vis.float().mean():.2f} -> {out}")

    if not args.no_vis:
        query_frame = 0 if args.backward_tracking else args.grid_query_frame
        Visualizer(save_dir=out, pad_value=0, linewidth=2, fps=args.fps).visualize(
            video.cpu(), tracks.cpu(), vis.cpu(), filename="tracks", query_frame=query_frame
        )


if __name__ == "__main__":
    main()
