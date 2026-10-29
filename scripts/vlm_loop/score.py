"""Score one VLM batch against the labels of the set.

    python scripts/vlm_loop/score.py BATCH_OUT LABELS_CSV OUT_DIR [--calibrate] [--video_suffix _01]

BATCH_OUT is the GUI output folder of the batch (outputs/gui/<batch name>), with <demo>/<demo>_01_vlm.json.
A video is marked bad when any question is Yes (P(yes) > the question's threshold).

--calibrate sets every threshold from the P(yes) values of this batch instead of using the thresholds the
questions were run with: halfway between the highest P(yes) of a good video and the next higher P(yes) of a bad
video, so the question gives no good video a Yes and keeps the largest distance to the videos on both sides.
A question where no bad video is above every good video gets a threshold halfway between the highest good
P(yes) and 1 (it then catches nothing). The batch does not have to run again: only the comparison changes.

Writes:
    OUT_DIR/score.json        correct, bad_caught, good_passed, bad_passed (bad videos marked good),
                              good_flagged, the Yes list of every question, the margin (for every question
                              the smallest distance between its threshold and any video's P(yes), and the
                              smallest of these), and the questions with the thresholds used
    OUT_DIR/pyes.csv          one row per video: label, verdict and the P(yes) of every question
  with --calibrate also:
    OUT_DIR/score_as_run.json the same with the thresholds the questions were run with
    OUT_DIR/questions_calibrated.json  the questions with the set thresholds
Prints one summary line.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load(batch_out: Path, labels_csv: Path, video_suffix: str) -> tuple[dict, list[dict], dict]:
    """-> (labels {video: good|bad}, questions as run, P(yes) {video: {question id: p}})."""
    with open(labels_csv, newline="") as f:
        labels = {r["video"]: r["label"] for r in csv.DictReader(f)}
    pyes, questions = {}, None
    for v in sorted(labels):
        res = json.load(open(batch_out / v / f"{v}{video_suffix}_vlm.json"))
        pyes[v] = {q: a["p_yes"] for q, a in res["answers"].items()}
        questions = questions or res["questions"]
    return labels, questions, pyes


def calibrate(questions: list[dict], pyes: dict, labels: dict) -> list[dict]:
    """The questions with thresholds set from the P(yes) values (see --calibrate)."""
    out = []
    for q in questions:
        good_max = max(pyes[v][q["id"]] for v in labels if labels[v] == "good")
        above = [pyes[v][q["id"]] for v in labels if labels[v] == "bad" and pyes[v][q["id"]] > good_max]
        upper = min(above) if above else 1.0
        out.append(dict(q, threshold=(good_max + upper) / 2))
    return out


def score_with(questions: list[dict], pyes: dict, labels: dict) -> tuple[dict, list[dict]]:
    qids = [q["id"] for q in questions]
    thr = {q["id"]: q["threshold"] for q in questions}
    yes = {v: {q: pyes[v][q] > thr[q] for q in qids} for v in labels}
    verdict = {v: "bad" if any(yes[v].values()) else "good" for v in labels}
    bad = [v for v in labels if labels[v] == "bad"]
    good = [v for v in labels if labels[v] == "good"]
    margins = {q: round(min(abs(pyes[v][q] - thr[q]) for v in labels), 4) for q in qids}
    s = {
        "videos": len(labels),
        "correct": sum(verdict[v] == labels[v] for v in labels),
        "bad_caught": sum(verdict[v] == "bad" for v in bad), "bad_total": len(bad),
        "good_passed": sum(verdict[v] == "good" for v in good), "good_total": len(good),
        "bad_passed": [v for v in bad if verdict[v] == "good"],
        "good_flagged": [v for v in good if verdict[v] == "bad"],
        "yes_per_question": {q: {"bad": [v for v in bad if yes[v][q]], "good": [v for v in good if yes[v][q]]}
                             for q in qids},
        "margin_per_question": margins,
        "margin": min(margins.values()),
        "questions": questions,
    }
    rows = [dict(video=v, label=labels[v], verdict=verdict[v], **{f"{q}_p_yes": pyes[v][q] for q in qids})
            for v in sorted(labels)]
    return s, rows


def score(batch_out: Path, labels_csv: Path, video_suffix: str = "_01") -> tuple[dict, list[dict]]:
    """The batch with the thresholds it was run with."""
    labels, questions, pyes = load(batch_out, labels_csv, video_suffix)
    return score_with(questions, pyes, labels)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("batch_out", type=Path)
    p.add_argument("labels", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument("--calibrate", action="store_true", help="set the thresholds from this batch's P(yes)")
    p.add_argument("--video_suffix", default="_01")
    args = p.parse_args()
    labels, questions, pyes = load(args.batch_out, args.labels, args.video_suffix)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.calibrate:
        as_run, _ = score_with(questions, pyes, labels)
        json.dump(as_run, open(args.out_dir / "score_as_run.json", "w"), indent=1)
        questions = calibrate(questions, pyes, labels)
        json.dump(questions, open(args.out_dir / "questions_calibrated.json", "w"), indent=1)
    s, rows = score_with(questions, pyes, labels)
    json.dump(s, open(args.out_dir / "score.json", "w"), indent=1)
    with open(args.out_dir / "pyes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"{s['correct']}/{s['videos']} right, bad caught {s['bad_caught']}/{s['bad_total']}, "
          f"good passed {s['good_passed']}/{s['good_total']}, margin {s['margin']}"
          + (" (thresholds set from this batch)" if args.calibrate else ""))


if __name__ == "__main__":
    main()
