"""VLM quality check of one video: Y/N questions about its first, middle and final frames, asked to a
local Qwen2.5-VL model (ported from VLM-ppl's evaluate_videos.py).

Each Y adds 1/M to the anomaly score (M questions): 0 = no issue found, 1 = every issue found.
The batch pipeline runs it on video B of every demo (scripts/batch_pipeline.py, step "vlm").

Questions: a JSON list of {"id": ..., "question": ...} (default scripts/vlm_questions.json); ids are
letters, digits and underscores and become the column names in the GUI's batch table.
Model weights: checkpoints/vlm/ (Hugging Face cache layout); a model that is not there yet is
downloaded into it.

Single video:
    python scripts/vlm_check.py --video path/to/demo_001_01.mp4 [--device cuda:0] [--questions q.json]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_QUESTIONS = REPO / "scripts" / "vlm_questions.json"
MODEL_CACHE = REPO / "checkpoints" / "vlm"
QUESTION_ID = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class Question:
    id: str
    text: str


def load_questions(path: Path) -> list[Question]:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read questions from {path}: {exc}") from exc
    if not isinstance(raw, list) or not raw:
        raise ValueError("Questions file must contain a non-empty JSON list")
    questions: list[Question] = []
    seen: set[str] = set()
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Question {index} must be a JSON object")
        question_id, text = item.get("id"), item.get("question")
        if not isinstance(question_id, str) or not QUESTION_ID.fullmatch(question_id):
            raise ValueError(f"Question {index} has invalid id {question_id!r}; use letters, digits, and underscores")
        if question_id in seen:
            raise ValueError(f"Duplicate question id: {question_id}")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Question {index} has empty question text")
        questions.append(Question(question_id, text.strip()))
        seen.add(question_id)
    return questions


def run_checked(command: list[str]) -> str:
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Required executable not found: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Command failed ({command[0]}): {(exc.stderr or exc.stdout).strip()}") from exc
    return result.stdout


def get_frame_count(video: Path) -> int:
    # The stream metadata usually has the exact count; decode the stream only when it is missing.
    probe = run_checked(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                         "stream=nb_frames", "-of", "json", str(video)])
    try:
        count = int(json.loads(probe)["streams"][0]["nb_frames"])
        if count > 0:
            return count
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
        pass
    probe = run_checked(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                         "-show_entries", "stream=nb_read_frames", "-of", "json", str(video)])
    try:
        count = int(json.loads(probe)["streams"][0]["nb_read_frames"])
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not determine frame count for {video}") from exc
    if count < 1:
        raise RuntimeError(f"Video has no readable frames: {video}")
    return count


def extract_frames(video: Path, out_dir: Path) -> dict[str, Path]:
    """first / middle / final frame of `video` as JPEGs in out_dir/<video stem>/."""
    count = get_frame_count(video)
    indexes = {"first": 0, "middle": (count - 1) // 2, "final": count - 1}
    frame_dir = out_dir / video.stem
    frame_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    for label, index in indexes.items():
        output = outputs[label] = frame_dir / f"{label}.jpg"
        run_checked(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
                     "-vf", f"select=eq(n\\,{index})", "-frames:v", "1", "-q:v", "2", str(output)])
        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError(f"Failed to extract frame {index} from {video}")
    return outputs


def prompt(questions: list[Question]) -> str:
    properties = ", ".join(f'"{q.id}": {{"answer": "Y or N", "reason": "brief evidence"}}' for q in questions)
    question_text = "\n".join(f"- {q.id}: {q.text}" for q in questions)
    return (
        "The three images are, in order, the FIRST, MIDDLE, and FINAL frames "
        "from one robot video. Each image contains two side-by-side camera views. "
        "Act as a strict visual quality-control inspector. Judge only visible "
        "evidence. For every question, answer Y if the described issue occurs in "
        "any frame; otherwise answer N.\n\n"
        f"Questions:\n{question_text}\n\n"
        "Return only one valid JSON object, with no markdown or extra text, in "
        f"exactly this shape: {{\"answers\": {{{properties}}}}}"
    )


def parse_json_object(text: str) -> dict[str, Any]:
    """The first JSON object in the model output (it sometimes adds markdown around it)."""
    decoder = json.JSONDecoder()
    for position, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[position:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise RuntimeError(f"Model returned no valid JSON object: {text[:500]!r}")


def validate_answers(result: dict[str, Any], questions: list[Question]) -> dict[str, dict[str, str]]:
    answers = result.get("answers") if isinstance(result, dict) else None
    expected = {q.id for q in questions}
    if not isinstance(answers, dict) or set(answers) != expected:
        raise RuntimeError(f"Model answer ids do not match questions: expected {sorted(expected)}")
    for question_id, item in answers.items():
        if not isinstance(item, dict) or item.get("answer") not in {"Y", "N"}:
            raise RuntimeError(f"Invalid Y/N answer for {question_id}")
        if not isinstance(item.get("reason"), str):
            raise RuntimeError(f"Missing reason for {question_id}")
    return answers


def anomaly_score(answers: dict[str, dict[str, str]]) -> float:
    return sum(item["answer"] == "Y" for item in answers.values()) / len(answers)


class QwenChecker:
    """Qwen2.5-VL on one device ("cpu", "cuda:N"), loaded once and reused for every video."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cpu", model_cache: Path = MODEL_CACHE):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.torch, self.device = torch, device
        model_cache.mkdir(parents=True, exist_ok=True)

        def load(local_only):
            processor = AutoProcessor.from_pretrained(model_name, cache_dir=model_cache, local_files_only=local_only)
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name, torch_dtype="auto", attn_implementation="sdpa", cache_dir=model_cache,
                local_files_only=local_only)
            return processor, model

        try:
            self.processor, model = load(True)
        except OSError:  # not in checkpoints/vlm yet: download it there
            print(f"Downloading {model_name} into {model_cache}...", flush=True)
            self.processor, model = load(False)
        self.model = model.to(device).eval()

    def ask(self, frames: dict[str, Path], questions: list[Question], max_new_tokens: int = 512):
        from qwen_vl_utils import process_vision_info

        content: list[dict[str, Any]] = [
            {"type": "image", "image": frames[label].resolve().as_uri()} for label in ("first", "middle", "final")
        ]
        content.append({"type": "text", "text": prompt(questions)})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True,
                                return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
        output = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return validate_answers(parse_json_object(output), questions)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--device", default="cuda:0", help="cpu or cuda:N")
    p.add_argument("--frames_dir", type=Path, default=REPO / "outputs" / "vlm_frames")
    p.add_argument("--max_new_tokens", type=int, default=512)
    args = p.parse_args()
    questions = load_questions(args.questions)
    frames = extract_frames(args.video, args.frames_dir)
    answers = QwenChecker(args.model, args.device).ask(frames, questions, args.max_new_tokens)
    for q in questions:
        print(f"{q.id}: {answers[q.id]['answer']} - {answers[q.id]['reason']}")
    print(f"anomaly score: {anomaly_score(answers):.3f}")


if __name__ == "__main__":
    main()
