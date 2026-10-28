"""Score one VLM batch against the labels of the set.

    python scripts/vlm_loop/score.py BATCH_OUT LABELS_CSV OUT_DIR [--video_suffix _01]

BATCH_OUT is the GUI output folder of the batch (outputs/gui/<batch name>), with <demo>/<demo>_01_vlm.json.
A video is marked bad when any question is Y. Writes:
    OUT_DIR/score.json  correct, bad_caught, good_passed, bad_passed (bad videos marked good), good_flagged,
                        the Yes list of every question for good and bad videos, and the margin: for every
                        question the smallest distance between its threshold and the P(yes) of any video, and
                        the smallest of these over the questions (larger = the verdicts change less when
                        P(yes) moves a little between runs)
    OUT_DIR/pyes.csv    one row per video: label, verdict and the P(yes) of every question
Prints one summary line.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def score(batch_out: Path, labels_csv: Path, video_suffix: str = "_01") -> tuple[dict, list[dict]]:
    with open(labels_csv, newline="") as f:
        labels = {r["video"]: r["label"] for r in csv.DictReader(f)}
    answers, questions = {}, None
    for v in sorted(labels):
        res = json.load(open(batch_out / v / f"{v}{video_suffix}_vlm.json"))
        answers[v] = res["answers"]
        questions = questions or res["questions"]
    qids = [q["id"] for q in questions]
    verdict = {v: "bad" if any(a["answer"] == "Y" for a in answers[v].values()) else "good" for v in labels}
    bad = [v for v in labels if labels[v] == "bad"]
    good = [v for v in labels if labels[v] == "good"]
    margins = {q["id"]: round(min(abs(answers[v][q["id"]]["p_yes"] - q["threshold"]) for v in labels), 4)
               for q in questions}
    s = {
        "videos": len(labels),
        "correct": sum(verdict[v] == labels[v] for v in labels),
        "bad_caught": sum(verdict[v] == "bad" for v in bad), "bad_total": len(bad),
        "good_passed": sum(verdict[v] == "good" for v in good), "good_total": len(good),
        "bad_passed": [v for v in bad if verdict[v] == "good"],
        "good_flagged": [v for v in good if verdict[v] == "bad"],
        "yes_per_question": {q: {"bad": [v for v in bad if answers[v][q]["answer"] == "Y"],
                                 "good": [v for v in good if answers[v][q]["answer"] == "Y"]} for q in qids},
        "margin_per_question": margins,
        "margin": min(margins.values()),
        "questions": questions,
    }
    rows = [dict(video=v, label=labels[v], verdict=verdict[v],
                 **{f"{q}_p_yes": answers[v][q]["p_yes"] for q in qids}) for v in sorted(labels)]
    return s, rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("batch_out", type=Path)
    p.add_argument("labels", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument("--video_suffix", default="_01")
    args = p.parse_args()
    s, rows = score(args.batch_out, args.labels, args.video_suffix)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(s, open(args.out_dir / "score.json", "w"), indent=1)
    with open(args.out_dir / "pyes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"{s['correct']}/{s['videos']} right, bad caught {s['bad_caught']}/{s['bad_total']}, "
          f"good passed {s['good_passed']}/{s['good_total']}, margin {s['margin']}")


if __name__ == "__main__":
    main()
