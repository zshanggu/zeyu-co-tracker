# VLM question loop

The VLM check of the batch pipeline (`scripts/vlm_check.py`) asks a local Qwen2.5-VL model yes/no questions
about a few frames of a generated video; a video with any Yes is marked bad. This folder finds those
questions automatically: a question writer (Claude, through the Claude Code CLI) looks at frame sheets,
the labels of a set of videos and the checker's last results, and writes new questions. The checker runs
them on the set, the score goes back to the writer, and so on, until the score stops going up.

Goal of the questions: only good videos are used to train a robot policy. A video is bad when, over the
whole video, a problem is big enough to hurt robot learning; small problems are fine. The worst mistake is a
bad video that passes as good.

## Files

| file | what it does | where it runs |
|---|---|---|
| `make_set.py` | set folder: links to the chosen demo folders, and `labels.csv` from a verdict CSV | GUI machine |
| `make_sheets.py` | frame sheets for the writer: `frames11/` (11 moments, full frame), `crops11/` (object close-ups) | GUI machine |
| `run_batch.py` | one VLM-only batch through the GUI server (like its Start button); stops when the checker would run on the CPU | GUI machine |
| `score.py` | the batch against the labels: `score.json`, `pyes.csv` | GUI machine |
| `prompt_template.txt` | the prompt of the writer, filled each round | - |
| `loop.py` | the whole loop; only the Python standard library | machine with the `claude` command |
| `config.example.json` | all settings of one run | - |

The loop runs on the machine with the Claude Code CLI and reaches the GUI machine with the `remote.prefix`
command of the config (for example `ssh user@host docker exec -i container`), or runs there directly with an
empty prefix.

## Run

1. Make the set (on the GUI machine):

       python scripts/vlm_loop/make_set.py --demos_dir DEMOS --demos 0-49 --verdicts VERDICTS.csv \
           --video_col demo --verdict_col verdict --good approved,weak --bad rejected --out SET_DIR

2. Copy `config.example.json`, fill in the paths, and start the loop (on the machine with `claude`):

       python scripts/vlm_loop/loop.py CONFIG.json

   The same command goes on after a stop: rounds with a score are not run again.
3. Stop it with a file `STOP` in the run folder, or with SIGTERM; it ends after the current step.

## A round

- Round 0 runs `round0_questions` (default `scripts/vlm_questions.json`).
- Round r >= 1:
  1. `round<r>/prompt.txt`: `prompt_template.txt` filled with the earlier rounds and their scores.
  2. The writer runs as `claude -p` in the run folder. It may only read (Read, Glob, Grep) inside the run folder; Bash, Edit, Write and web tools are off. Its report goes to `round<r>/writer_report.md`, every tool call to `round<r>/writer_stream.jsonl` and the files it opened to `round<r>/files_read.json`. A file outside the run folder stops the loop.
  3. The questions come from the ```json block of the report: `round<r>/questions.json`.
  4. `run_batch.py` runs them on the set (GUI batch `<batch_prefix><r>`, visible in the Batch dialog) and `score.py` scores them.
  5. `score.json`, `pyes.csv` and the frames the checker saw (`qwen_frames/`) are copied into `round<r>/` for the next writer.

## Stop and pick

- Stop: `patience` rounds in a row without more correct videos (a tie is no gain), or round `max_rounds`.
- Pick: the round with the most correct videos; tie -> fewer bad videos passed as good; tie -> the larger
  margin (the smallest distance between a threshold and any video's P(yes), see `score.py`); tie -> the earlier
  round.
- `summary.md` and `summary.json` (with the picked questions) in the run folder.

## Status files (for a GUI panel)

- `loop_status.json`: the current step, one row per finished round (right, bad passed, good flagged, margin,
  question ids), the stop reason and the pick when finished. Written after every step.
- `loop.log`: the text log.
