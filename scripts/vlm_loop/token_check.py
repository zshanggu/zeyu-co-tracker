"""Check which tokens the VLM puts its probability on at the first answer token.

    python scripts/vlm_loop/token_check.py --share SHARE_DIR --sets SETS_CSV --set SET --questions QUESTIONS_JSON \
        --name NAME [--pyes PYES_CSV] [--first_good 10 --first_bad 10] [--towel_color "R,G,B; R,G,B"]
        [--device cuda:0]

The VLM check (scripts/vlm_check.py) scores a question with
    P(yes) = (P("Yes") + P("yes")) / (P("Yes") + P("yes") + P("No") + P("no")),
read at the first answer token. It reads only these four tokens. The tokenizer also has the same words with a
leading space (" Yes", " yes", " No", " no") as separate tokens. This script measures how much probability
the model puts on the four answer tokens, on the four leading-space tokens, and on all other tokens.

It asks the questions on the videos of one set, with the same images and the same prompt as the VLM check,
and writes one row per video and question to outputs/vlm_loop/token_check/NAME.csv:
    video, label, question, p_yes, p_yes_saved, p_yes_with_space, sum_answer, sum_space, sum_space_yes,
    sum_space_no, sum_other, top_token, top_prob
p_yes is computed here from the four answer tokens. p_yes_saved is the value of --pyes (a pyes CSV of
score.py, e.g. scripts/vlm_loop/results/vlm_v1.0/franka_pyes_vlm_v1.0.csv); the two must agree. Without
--pyes the column is empty. p_yes_with_space counts the leading-space tokens too:
    (P("Yes") + P("yes") + P(" Yes") + P(" yes")) / (the same + P("No") + P("no") + P(" No") + P(" no")).
sum_space_yes and sum_space_no are the two parts of sum_space. The summary (min, mean, max of every sum) is
printed.

To score both ways with score.py, two more files are written, in the format of score.py --pyes (one row per
video: video, then <question id>_p_yes for every question):
    outputs/vlm_loop/token_check/NAME_pyes.csv             p_yes
    outputs/vlm_loop/token_check/NAME_pyes_with_space.csv  p_yes_with_space

The videos are read straight from the share folder (as run_batch.py does). With --first_good and --first_bad
only the first N good and the first M bad videos of the set are used. The frames go to a temporary folder
that is removed at the end.
"""

from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import vlm_check  # noqa: E402
from extract_mask import parse_colors  # noqa: E402
from make_set import read_sets, share_demos  # noqa: E402

ANSWER_WORDS = ("Yes", "yes", "No", "no")
OUT_DIR = REPO / "outputs" / "vlm_loop" / "token_check"
COLUMNS = ["video", "label", "question", "p_yes", "p_yes_saved", "p_yes_with_space", "sum_answer", "sum_space",
           "sum_space_yes", "sum_space_no", "sum_other", "top_token", "top_prob"]


def single_token(tokenizer, word: str) -> int:
    """The id of a word that must be one token."""
    ids = tokenizer.encode(word, add_special_tokens=False)
    if len(ids) != 1:
        sys.exit(f"{word!r} is {len(ids)} tokens in this tokenizer, expected 1")
    return ids[0]


def first_token_probs(checker, images: list[Path], question: str):
    """The probability of every token at the first answer token. The input is built as in
    vlm_check.QwenChecker.p_yes."""
    from qwen_vl_utils import process_vision_info

    content = [{"type": "image", "image": p.resolve().as_uri()} for p in images]
    content.append({"type": "text", "text": f"{question} Answer Yes or No."})
    messages = [{"role": "user", "content": content}]
    text = checker.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(
        messages, image_patch_size=checker.processor.image_processor.patch_size)
    inputs = checker.processor(text=[text], images=image_inputs, videos=video_inputs,
                               return_tensors="pt").to(checker.device)
    with checker.torch.inference_mode():
        logits = checker.model(**inputs).logits[0, -1].float()
    return checker.torch.softmax(logits, -1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--share", type=Path, required=True, help="the share folder with the videos (share.py)")
    p.add_argument("--sets", type=Path, required=True, help="the sets CSV (video, label, set; make_set.py)")
    p.add_argument("--set", required=True, help="the set to use, e.g. test")
    p.add_argument("--questions", type=Path, required=True, help="questions JSON")
    p.add_argument("--name", required=True, help="the output is outputs/vlm_loop/token_check/NAME.csv")
    p.add_argument("--pyes", type=Path, help="a pyes CSV (score.py) with the saved P(yes) to compare with")
    p.add_argument("--first_good", type=int, help="use only the first N good videos (with --first_bad)")
    p.add_argument("--first_bad", type=int, help="use only the first M bad videos (with --first_good)")
    p.add_argument("--towel_color", default="167,18,135")
    p.add_argument("--model", default=vlm_check.DEFAULT_MODEL)
    p.add_argument("--device", default="cuda:0", help="cpu or cuda:N")
    p.add_argument("--video_suffix", default="_01")
    p.add_argument("--instance_suffix", default="_instance_all")
    args = p.parse_args()
    if (args.first_good is None) != (args.first_bad is None):
        p.error("--first_good and --first_bad go together")

    questions = vlm_check.load_questions(args.questions)
    rows = [r for r in read_sets(args.sets) if r["set"] == args.set]
    if not rows:
        sys.exit(f"no videos of set {args.set} in {args.sets}")
    if args.first_good is not None:
        rows = ([r for r in rows if r["label"] == "good"][:args.first_good]
                + [r for r in rows if r["label"] == "bad"][:args.first_bad])
    folders = share_demos(args.share)
    missing = [r["video"] for r in rows if r["video"] not in folders]
    if missing:
        sys.exit(f"{len(missing)} videos are not in {args.share / 'videos'}: {', '.join(missing[:5])}")
    saved = {}
    if args.pyes:
        with open(args.pyes, newline="") as f:
            saved = {r["video"]: r for r in csv.DictReader(f)}
    colors = parse_colors(args.towel_color)

    checker = vlm_check.QwenChecker(args.model, args.device)
    tok = checker.processor.tokenizer
    answer_ids = [single_token(tok, w) for w in ANSWER_WORDS]
    space_ids = [single_token(tok, " " + w) for w in ANSWER_WORDS]
    print(f"answer tokens {dict(zip(ANSWER_WORDS, answer_ids))}")
    print(f"leading-space tokens {dict(zip([' ' + w for w in ANSWER_WORDS], space_ids))}", flush=True)

    out = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, r in enumerate(rows, 1):
            name = r["video"]
            video = folders[name] / f"{name}{args.video_suffix}.mp4"
            instance = folders[name] / f"{name}{args.instance_suffix}.mp4"
            images = vlm_check.question_images(video, questions, Path(tmp),
                                               instance if instance.is_file() else None, colors)
            for q in questions:
                probs = first_token_probs(checker, images[q.id], q.text)
                yes, no = probs[answer_ids[:2]].sum().item(), probs[answer_ids[2:]].sum().item()
                space_yes, space_no = probs[space_ids[:2]].sum().item(), probs[space_ids[2:]].sum().item()
                sum_space = space_yes + space_no
                top_prob, top_id = probs.max(-1)
                old = saved.get(name, {}).get(f"{q.id}_p_yes", "")
                out.append({"video": name, "label": r["label"], "question": q.id, "p_yes": yes / (yes + no),
                            "p_yes_saved": old,
                            "p_yes_with_space": (yes + space_yes) / (yes + no + sum_space),
                            "sum_answer": yes + no, "sum_space": sum_space,
                            "sum_space_yes": space_yes, "sum_space_no": space_no,
                            "sum_other": 1 - yes - no - sum_space, "top_token": tok.decode([top_id.item()]),
                            "top_prob": top_prob.item()})
            print(f"{i}/{len(rows)} {name}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{args.name}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(out)
    for key, suffix in (("p_yes", "_pyes"), ("p_yes_with_space", "_pyes_with_space")):
        with open(OUT_DIR / f"{args.name}{suffix}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["video"] + [f"{q.id}_p_yes" for q in questions])
            for r in rows:
                values = {o["question"]: o[key] for o in out if o["video"] == r["video"]}
                w.writerow([r["video"]] + [values[q.id] for q in questions])

    print(f"\n{len(out)} answers ({len(rows)} videos x {len(questions)} questions), model {args.model}")
    for key, label in (("sum_answer", "four answer tokens"), ("sum_space", "four leading-space tokens"),
                       ("sum_other", "all other tokens")):
        v = [o[key] for o in out]
        print(f"{label}: min {min(v):.6f}  mean {sum(v) / len(v):.6f}  max {max(v):.6f}")
    shift = [abs(o["p_yes_with_space"] - o["p_yes"]) for o in out]
    print(f"abs(p_yes_with_space - p_yes): mean {sum(shift) / len(shift):.2e}  max {max(shift):.2e}")
    compared = [abs(o["p_yes"] - float(o["p_yes_saved"])) for o in out if o["p_yes_saved"] != ""]
    if compared:
        print(f"abs(p_yes - p_yes_saved): max {max(compared):.2e} ({len(compared)} answers)")
    tops: dict[str, int] = {}
    for o in out:
        tops[o["top_token"]] = tops.get(o["top_token"], 0) + 1
    print("top token counts: " + ", ".join(f"{t!r} {n}" for t, n in sorted(tops.items(), key=lambda x: -x[1])))
    print(path)


if __name__ == "__main__":
    main()
