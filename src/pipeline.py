from __future__ import annotations

import argparse
import threading
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional at runtime
    load_dotenv = None

from .analysis import generate_tsv, load_results_dataframe, run_analysis
from .models import MODEL_REGISTRY, OpenRouterClient
from .utils import (
    append_jsonl,
    build_contact_sheet,
    ensure_dir,
    extract_frames,
    get_video_metadata,
    load_annotations,
    normalize_video_path,
    read_jsonl,
    select_frame_numbers,
    video_stem,
)


DEFAULT_MODELS = list(MODEL_REGISTRY.keys())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VLM blind spot discovery pipeline")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--annotations", default="data/annotations.json", help="Path to annotations JSON")
    parser.add_argument("--results-path", default="output/model_responses.jsonl", help="Path to JSONL results file")
    parser.add_argument("--output-dir", default="output", help="Directory for analysis outputs")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS), help="Comma-separated logical model names")
    parser.add_argument("--overwrite", action="store_true", help="Re-run targets even if they already exist")

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("extract-frames", help="Extract baseline frames for all annotations")

    evaluate = subparsers.add_parser("evaluate", help="Run one evaluation pass")
    evaluate.add_argument("--prompt-type", required=True, choices=["baseline", "probe_a", "probe_b"])

    subparsers.add_parser("analyze", help="Generate CSV summaries and figures")
    subparsers.add_parser("generate-tsv", help="Generate tasks_and_rubrics.tsv from JSONL results")
    inventory = subparsers.add_parser("inventory-videos", help="Build contact sheets and metadata for local videos")
    inventory.add_argument("--videos-dir", default="data/videos", help="Directory containing source videos")
    inventory.add_argument("--num-frames", default=12, type=int, help="Number of evenly sampled frames per contact sheet")
    subparsers.add_parser("run", help="Run baseline, probes, analysis, and TSV generation")

    return parser.parse_args()


def _load_env(project_root: Path) -> None:
    if load_dotenv is not None:
        load_dotenv(project_root / ".env", override=False)


def parse_models(raw_models: str) -> list[str]:
    models = [value.strip() for value in raw_models.split(",") if value.strip()]
    unsupported = [model for model in models if model not in MODEL_REGISTRY]
    if unsupported:
        raise ValueError(f"unsupported models: {unsupported}")
    return models


def _results_index(rows: list[dict[str, Any]]) -> set[tuple[str, str, str]]:
    return {(str(row["failure_id"]), str(row["model"]), str(row["prompt_type"])) for row in rows}


def _should_evaluate_failure_for_probe_b(failure: dict[str, Any]) -> bool:
    category = str(failure.get("category", "")).lower()
    return "temporal" in category or "camera" in category


def _baseline_blind_spot_targets(results_rows: list[dict[str, Any]]) -> set[tuple[str, str]]:
    return {
        (str(row["failure_id"]), str(row["model"]))
        for row in results_rows
        if row.get("prompt_type") == "baseline" and bool(row.get("blind_spot_flag"))
    }


def build_evaluation_targets(
    failures: list[dict[str, Any]],
    models: list[str],
    prompt_type: str,
    results_rows: list[dict[str, Any]],
    overwrite: bool,
) -> list[tuple[dict[str, Any], str]]:
    indexed_results = _results_index(results_rows)
    baseline_blind_spots = _baseline_blind_spot_targets(results_rows)
    failure_by_id = {str(failure["id"]): failure for failure in failures}

    targets: list[tuple[dict[str, Any], str]] = []
    for failure in failures:
        failure_id = str(failure["id"])
        for model in models:
            key = (failure_id, model, prompt_type)
            if not overwrite and key in indexed_results:
                continue

            if prompt_type == "baseline":
                targets.append((failure, model))
                continue

            if (failure_id, model) not in baseline_blind_spots:
                continue

            if prompt_type == "probe_b" and not _should_evaluate_failure_for_probe_b(failure_by_id[failure_id]):
                continue

            targets.append((failure, model))
    return targets


def _frame_output_dir(project_root: Path, failure: dict[str, Any], prompt_type: str) -> Path:
    video_name = video_stem(failure["source_video"])
    return project_root / "data" / "frames" / failure["id"] / prompt_type / video_name


def extract_failure_frames(project_root: Path, failure: dict[str, Any], prompt_type: str, overwrite: bool = False) -> tuple[list[int], list[Path]]:
    frame_numbers = select_frame_numbers(failure, prompt_type)
    video_path = normalize_video_path(failure["source_url"], project_root)
    output_dir = _frame_output_dir(project_root, failure, prompt_type)
    image_paths = extract_frames(video_path, frame_numbers, output_dir, overwrite=overwrite)
    return frame_numbers, image_paths


def evaluate_once(
    project_root: Path,
    failure: dict[str, Any],
    model: str,
    prompt_type: str,
    client: OpenRouterClient,
    results_path: Path,
    overwrite: bool,
) -> dict[str, Any]:
    frame_ids, image_paths = extract_failure_frames(project_root, failure, prompt_type, overwrite=overwrite)
    response = client.query_vlm(
        model_name=model,
        image_paths=image_paths,
        frame_ids=frame_ids,
        question=str(failure["targeted_question"]),
        failure_id=str(failure["id"]),
        prompt_type=prompt_type,
    )
    record = {
        "failure_id": str(failure["id"]),
        "source_video": str(failure["source_video"]),
        "source_url": str(failure.get("source_url", "")),
        "source_model": str(failure.get("source_model", "")),
        "category": str(failure["category"]),
        "frame_range": failure.get("frame_range", []),
        "timestamp_sec": failure.get("timestamp_sec", []),
        "frame_ids_used": response.frame_ids_used,
        "image_paths_used": response.image_paths_used,
        "severity": failure.get("severity"),
        "human_noticeability": failure.get("human_noticeability"),
        "human_judgment": str(failure["human_judgment"]),
        "targeted_question": str(failure["targeted_question"]),
        "expected_answer": str(failure.get("expected_answer", "")),
        "difficulty": str(failure.get("difficulty", "")),
        "model": response.model,
        "prompt_type": response.prompt_type,
        "judgment": response.judgment,
        "confidence": response.confidence,
        "explanation": response.explanation,
        "raw_response": response.raw_response,
        "parse_method": response.parse_method,
        "blind_spot_flag": response.judgment == "PASS",
    }
    append_jsonl(results_path, record)
    return record


def command_extract_frames(project_root: Path, annotations_path: Path, overwrite: bool) -> None:
    failures = load_annotations(annotations_path)
    for failure in failures:
        extract_failure_frames(project_root, failure, prompt_type="baseline", overwrite=overwrite)


def command_evaluate(args: argparse.Namespace, project_root: Path, annotations_path: Path, results_path: Path, models: list[str]) -> None:
    failures = load_annotations(annotations_path)
    ensure_dir(results_path.parent)
    results_rows = read_jsonl(results_path)
    targets = build_evaluation_targets(failures, models, args.prompt_type, results_rows, overwrite=args.overwrite)
    if not targets:
        print(f"No evaluation targets for prompt_type={args.prompt_type}.")
        return

    # Group targets by model and run each model in its own thread (one client per thread)
    by_model: dict[str, list[tuple[dict[str, Any], str]]] = {}
    for failure, model in targets:
        by_model.setdefault(model, []).append((failure, model))

    # Pre-warm SSL context in main thread to avoid macOS/Anaconda SSL threading issues
    import ssl
    ssl.create_default_context()

    write_lock = threading.Lock()
    results: list[dict[str, Any]] = []

    def run_model(model_targets: list[tuple[dict[str, Any], str]]) -> None:
        client = OpenRouterClient()
        for failure, model in model_targets:
            record = evaluate_once(project_root, failure, model, args.prompt_type, client, results_path, overwrite=args.overwrite)
            with write_lock:
                results.append(record)
            print(f"Evaluated {record['failure_id']} x {record['model']} x {record['prompt_type']} -> {record['judgment']}")

    threads = [threading.Thread(target=run_model, args=(model_targets,)) for model_targets in by_model.values()]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def command_analyze(results_path: Path, output_dir: Path) -> None:
    outputs = run_analysis(results_path, output_dir)
    for label, path in outputs.items():
        print(f"{label}: {path}")


def command_generate_tsv(results_path: Path, output_dir: Path) -> None:
    df = load_results_dataframe(results_path)
    output_path = output_dir / "tasks_and_rubrics.tsv"
    generate_tsv(df, output_path)
    print(output_path)


def command_inventory_videos(project_root: Path, videos_dir: Path, output_dir: Path, num_frames: int) -> None:
    candidates = sorted(videos_dir.glob("*.mp4"))
    if not candidates:
        print(f"No .mp4 files found in {videos_dir}")
        return

    contact_dir = ensure_dir(output_dir / "contact_sheets")
    rows: list[dict[str, Any]] = []
    for video_path in candidates:
        metadata = get_video_metadata(video_path)
        contact_sheet_path = contact_dir / f"{video_path.stem}.png"
        build_contact_sheet(video_path, contact_sheet_path, num_frames=num_frames)
        rows.append(
            {
                "video_name": video_path.name,
                "video_path": str(video_path.resolve().relative_to(project_root)),
                "fps": metadata["fps"],
                "frame_count": metadata["frame_count"],
                "duration_sec": metadata["duration_sec"],
                "width": metadata["width"],
                "height": metadata["height"],
                "contact_sheet": str(contact_sheet_path.resolve().relative_to(project_root)),
            }
        )

    import pandas as pd

    inventory_path = output_dir / "video_inventory.csv"
    pd.DataFrame(rows).to_csv(inventory_path, index=False)
    print(inventory_path)


def command_run(args: argparse.Namespace, project_root: Path, annotations_path: Path, results_path: Path, output_dir: Path, models: list[str]) -> None:
    failures = load_annotations(annotations_path)
    if not failures:
        print(f"No failures found in {annotations_path}. Fill annotations.json before running end-to-end.")
        return

    ensure_dir(results_path.parent)
    client = OpenRouterClient()
    results_rows = read_jsonl(results_path)

    for prompt_type in ("baseline", "probe_a", "probe_b"):
        targets = build_evaluation_targets(failures, models, prompt_type, results_rows, overwrite=args.overwrite)
        for failure, model in targets:
            record = evaluate_once(project_root, failure, model, prompt_type, client, results_path, overwrite=args.overwrite)
            results_rows.append(record)
            print(f"Evaluated {record['failure_id']} x {record['model']} x {record['prompt_type']} -> {record['judgment']}")

    command_analyze(results_path, output_dir)
    command_generate_tsv(results_path, output_dir)


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    _load_env(project_root)

    annotations_path = (project_root / args.annotations).resolve()
    results_path = (project_root / args.results_path).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    models = parse_models(args.models)

    if args.command == "extract-frames":
        command_extract_frames(project_root, annotations_path, overwrite=args.overwrite)
        return

    if args.command == "evaluate":
        command_evaluate(args, project_root, annotations_path, results_path, models)
        return

    if args.command == "analyze":
        command_analyze(results_path, output_dir)
        return

    if args.command == "generate-tsv":
        command_generate_tsv(results_path, output_dir)
        return

    if args.command == "inventory-videos":
        videos_dir = (project_root / args.videos_dir).resolve()
        command_inventory_videos(project_root, videos_dir, output_dir, num_frames=args.num_frames)
        return

    if args.command == "run":
        command_run(args, project_root, annotations_path, results_path, output_dir, models)
        return

    raise RuntimeError(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
