"""Run one VLM-only batch through the GUI server and wait for it, for a VLM question loop.

    python scripts/vlm_loop/run_batch.py --set_dir SET_DIR --name BATCH_NAME --questions QUESTIONS_JSON \
        [--gui http://localhost:7861] [--gpus 0] [--towel_color "R,G,B; R,G,B"] [--min_free_gb 18]

Run it on the machine of the GUI server (the server reads the files). It makes outputs/batch_inputs/BATCH_NAME
with links to the demo folders of SET_DIR, starts the batch like the Start button of the Batch dialog (all
CoTracker steps off, only "vlm" on), and waits. The results are in outputs/gui/BATCH_NAME and show in the GUI.

The batch is stopped and the script fails when the checker loads on the CPU instead of a GPU (no GPU of --gpus
has --min_free_gb free), when another batch is running, or when a video fails. A finished batch of the same
name and the same questions is not run again (the script only prints its folder).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def api(gui: str, path: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(gui.rstrip("/") + path, data=data, method="POST" if data is not None else "GET",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        sys.exit(f"GUI server {path}: HTTP {e.code} {e.read().decode(errors='replace')}")
    except urllib.error.URLError as e:
        sys.exit(f"GUI server {gui} not reachable: {e.reason}")


def link_root(set_dir: Path, root: Path):
    demos = sorted(d for d in set_dir.iterdir() if d.name.startswith("demo_"))
    if root.exists():
        have = sorted(p.name for p in root.iterdir())
        if have != [d.name for d in demos]:
            sys.exit(f"{root} exists with other demos than {set_dir}")
        return
    root.mkdir(parents=True)
    for d in demos:
        os.symlink(d.resolve(), root / d.name)


def same_finished(out: Path, questions: list, video_suffix: str) -> bool:
    """True when out has a finished batch whose results were made with these questions (as vlm_check
    records them in every <demo><video_suffix>_vlm.json)."""
    try:
        status = json.load(open(out / "batch_status.json"))
        results = sorted(out.glob(f"demo_*/demo_*{video_suffix}_vlm.json"))
        used = [json.load(open(p)).get("questions") for p in results]
    except (OSError, ValueError):
        return False
    return bool(status.get("finished")) and bool(used) and all(u == questions for u in used)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set_dir", type=Path, required=True)
    p.add_argument("--name", required=True, help="batch name: the input and output folder name")
    p.add_argument("--questions", required=True, help="questions JSON, relative to the repo")
    p.add_argument("--gui", default="http://localhost:7861")
    p.add_argument("--gpus", default="0", help="GPUs the checker may use, e.g. 0 or 2,3")
    p.add_argument("--towel_color", default="167,18,135")
    p.add_argument("--min_free_gb", type=float, default=18)
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--video_suffix", default="_01")
    p.add_argument("--instance_suffix", default="_instance_all")
    p.add_argument("--poll", type=float, default=10, help="seconds between status checks")
    args = p.parse_args()

    sys.path.insert(0, str(REPO / "scripts"))
    import vlm_check

    try:  # stop here on a bad questions file, before the GUI is asked
        questions = [q.to_json() for q in vlm_check.load_questions(REPO / args.questions)]
    except (RuntimeError, ValueError) as e:
        sys.exit(f"bad questions file {args.questions}: {e}")
    root = REPO / "outputs" / "batch_inputs" / args.name
    out = REPO / "outputs" / "gui" / args.name
    if same_finished(out, questions, args.video_suffix):
        print(f"already done: {out}")
        return
    link_root(args.set_dir, root)
    cur = api(args.gui, "/api/batch").get("batch")
    if cur and cur.get("job_status") == "running":
        sys.exit(f"another batch is running: {cur.get('out')}")
    body = {"root": str(root.relative_to(REPO)), "a_suffix": "_source_all", "b_suffix": args.video_suffix,
            "instance_suffix": args.instance_suffix,
            "steps": {"track_a": "off", "track_b": "off", "mask": "off", "compare": "off", "vlm": "run"},
            "track": {"gpus": [int(g) for g in args.gpus.split(",") if g != ""]},
            "vlm": {"questions": args.questions, "model": args.model, "towel_color": args.towel_color,
                    "min_free_gb": args.min_free_gb}}
    job = api(args.gui, "/api/batch/start", body).get("job")
    print(f"batch {args.name} started (job {job})", flush=True)
    t0, checked_device = time.time(), False
    while True:
        time.sleep(args.poll)
        cur = api(args.gui, "/api/batch").get("batch") or {}
        if cur.get("job") != job:
            sys.exit(f"the GUI now shows another batch ({cur.get('out')}); batch {args.name} lost")
        tail = cur.get("tail") or ""
        if not checked_device and "VLM check: loading" in tail:
            line = next(x for x in tail.splitlines() if "VLM check: loading" in x)
            print(line, flush=True)
            if " on CPU" in line:
                api(args.gui, "/api/batch/cancel", {})
                sys.exit(f"stopped: the checker would run on the CPU ({line.split(' on ', 1)[1]})")
            checked_device = True
        if cur.get("job_status") != "running":
            break
    demos = ((cur.get("progress") or {}).get("demos") or {})
    failed = [d for d, st in demos.items() if (st.get("vlm") or {}).get("status") not in ("done", "skipped")]
    if cur.get("job_status") != "done" or failed:
        sys.exit(f"batch {args.name} ended with {cur.get('job_status')}; not done: {', '.join(failed) or '-'}")
    print(f"done in {time.time() - t0:.0f} s: {out}")


if __name__ == "__main__":
    main()
