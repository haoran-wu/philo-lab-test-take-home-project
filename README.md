# VLM Blind Spot Discovery

This repository contains a lightweight research pipeline for the Philo Labs Track D take-home:

- curate human-labeled AI video failures
- query frontier VLMs with targeted questions
- identify blind spots where `human=FAIL` and `vlm=PASS`
- compare models, confidence, and prompt/harness rescue rates

## Repo layout

```text
.
├── PLAN_ClaudeCode.md          # shared plan (ClaudeCode)
├── PLAN_codex.md               # shared plan (Codex)
├── README.md
├── requirements.txt
├── .env.example
├── data/
│   ├── annotations.json        # 30 human-labeled failures (ground truth)
│   ├── annotations.example.json
│   ├── videos/                 # source videos
│   └── frames/                 # extracted key frames per failure
├── output/
│   ├── review.html             # human review page (frames + questions)
│   ├── contact_sheets/         # full-video thumbnail grids
│   └── figures/                # analysis charts (generated after evaluation)
├── report/
│   └── report.tex
└── src/
    ├── __init__.py
    ├── analysis.py
    ├── models.py
    ├── pipeline.py
    ├── review.py               # generates output/review.html
    └── utils.py
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Primary path:

- set `OPENROUTER_API_KEY`
- use logical model names:
  - `gemini-3.1-pro`
  - `gemini-3-flash`
  - `qwen-3.5-omni`
  - `gpt-5.4`

If OpenRouter changes a model slug, update the matching `MODEL_ID_*` value in `.env` instead of editing code.

## Annotation format

`data/annotations.json` should look like:

```json
{
  "failures": [
    {
      "id": "F001",
      "source_video": "dr_doctor_v1.mp4",
      "source_url": "sample_AI_videos/dr_doctor_v1.mp4",
      "source_model": "unknown",
      "category": "character_consistency",
      "frame_range": [120, 145],
      "timestamp_sec": [5.0, 6.0],
      "key_frames": [130, 140],
      "severity": "high",
      "human_noticeability": 5,
      "human_judgment": "Doctor's face changes during the head turn.",
      "targeted_question": "Compare frame 130 and 140. Are the facial features consistent?",
      "expected_answer": "No. The face changes noticeably.",
      "difficulty": "medium"
    }
  ]
}
```

Notes:

- `key_frames` drive baseline and `probe_a`.
- `frame_range` is used to expand temporal/camera failures for `probe_b`.
- the pipeline only assumes `human=FAIL` failure cases.

## Commands

Extract baseline frames:

```bash
python3 -m src.pipeline extract-frames
```

Build video inventory and contact sheets for manual review:

```bash
python3 -m src.pipeline inventory-videos
```

Run baseline evaluation:

```bash
python3 -m src.pipeline evaluate --prompt-type baseline
```

Run prompt-only rescue on blind spots:

```bash
python3 -m src.pipeline evaluate --prompt-type probe_a
```

Run harness rescue on temporal/camera blind spots:

```bash
python3 -m src.pipeline evaluate --prompt-type probe_b
```

Run analysis:

```bash
python3 -m src.pipeline analyze
```

Generate TSV:

```bash
python3 -m src.pipeline generate-tsv
```

End-to-end:

```bash
python3 -m src.pipeline run
```

## Outputs

Main outputs:

- `output/model_responses.jsonl`
- `output/tasks_and_rubrics.tsv`
- `output/summary_blind_spot_rates.csv`
- `output/summary_correlation.csv`
- `output/summary_confidence.csv`
- `output/summary_rescue_rates.csv`
- `output/figures/*.png`

Each result row stores:

- `failure_id`
- `model`
- `prompt_type`
- `frame_ids_used`
- `image_paths_used`
- `vlm_judgment`
- `vlm_confidence`
- `blind_spot_flag`
- `parse_method`

This makes baseline vs probe comparisons auditable.

## Implementation notes

- The pipeline prefers OpenCV for frame extraction.
- If OpenCV is unavailable, it raises a clear dependency error instead of silently failing.
- VLM responses are requested in JSON form first, with regex fallback parsing when necessary.
- Main analysis only uses `prompt_type=baseline`.
- Rescue-rate analysis compares probe rows back to their corresponding baseline rows.

## Known limitations

- No negative-control dataset is included yet, so there is no false-alarm analysis.
- Direct-provider fallback keys are scaffolded, but the current implementation uses OpenRouter as the active evaluation backend.
- The report template is a stub and should be filled after the first full run.
