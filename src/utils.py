from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - optional dependency at runtime
    cv2 = None


REQUIRED_FAILURE_FIELDS = {
    "id",
    "source_video",
    "category",
    "frame_range",
    "key_frames",
    "human_judgment",
    "targeted_question",
}


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump_json(path: str | Path, payload: Any) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    rows: list[dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_annotations(path: str | Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    failures = payload.get("failures", [])
    if not isinstance(failures, list):
        raise ValueError("annotations.json must contain a top-level 'failures' list")
    seen_ids: set[str] = set()
    for failure in failures:
        missing = REQUIRED_FAILURE_FIELDS - set(failure)
        if missing:
            raise ValueError(f"failure {failure.get('id', '<missing-id>')} is missing fields: {sorted(missing)}")
        failure_id = str(failure["id"])
        if failure_id in seen_ids:
            raise ValueError(f"duplicate failure id: {failure_id}")
        seen_ids.add(failure_id)
    return failures


def unique_preserve_order(values: Iterable[int]) -> list[int]:
    ordered: list[int] = []
    seen: set[int] = set()
    for value in values:
        cast_value = int(value)
        if cast_value in seen:
            continue
        ordered.append(cast_value)
        seen.add(cast_value)
    return ordered


def is_temporal_or_camera_category(category: str) -> bool:
    lowered = category.lower()
    return "temporal" in lowered or "camera" in lowered


def expand_frame_numbers(frame_range: list[int] | tuple[int, int], key_frames: Iterable[int], target_count: int) -> list[int]:
    start, end = int(frame_range[0]), int(frame_range[1])
    if end < start:
        start, end = end, start
    if start == end:
        return unique_preserve_order([start, *key_frames])
    linspace = np.linspace(start, end, num=max(target_count, 2))
    candidates = [int(round(value)) for value in linspace]
    combined = unique_preserve_order([start, *key_frames, *candidates, end])
    return sorted(combined)


def select_frame_numbers(failure: dict[str, Any], prompt_type: str) -> list[int]:
    key_frames = unique_preserve_order(failure.get("key_frames", []))
    if prompt_type in {"baseline", "probe_a"}:
        if not key_frames:
            raise ValueError(f"failure {failure['id']} must define key_frames for {prompt_type}")
        return sorted(key_frames)

    if prompt_type == "probe_b":
        frame_range = failure.get("frame_range")
        if not frame_range:
            raise ValueError(f"failure {failure['id']} must define frame_range for probe_b")
        target_count = 6 if is_temporal_or_camera_category(str(failure.get("category", ""))) else max(len(key_frames), 3)
        return expand_frame_numbers(frame_range, key_frames, target_count=target_count)

    raise ValueError(f"unsupported prompt_type: {prompt_type}")


def video_stem(video_path: str | Path) -> str:
    return Path(video_path).stem


def normalize_video_path(video_path: str | Path, project_root: str | Path) -> Path:
    candidate = Path(video_path)
    if candidate.is_absolute():
        return candidate
    root = Path(project_root)
    if candidate.exists():
        return candidate.resolve()
    return (root / candidate).resolve()


def get_video_fps(video_path: str | Path) -> float:
    if cv2 is None:
        raise RuntimeError("OpenCV is not installed. Install requirements.txt to enable frame extraction.")
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"failed to open video: {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            raise RuntimeError(f"failed to read FPS from video: {video_path}")
        return fps
    finally:
        capture.release()


def get_video_metadata(video_path: str | Path) -> dict[str, Any]:
    if cv2 is None:
        raise RuntimeError("OpenCV is not installed. Install requirements.txt to enable video metadata inspection.")
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"failed to open video: {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = frame_count / fps if fps > 0 else None
        return {
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration_sec": duration_sec,
        }
    finally:
        capture.release()


def extract_frames(video_path: str | Path, frame_numbers: Iterable[int], output_dir: str | Path, overwrite: bool = False) -> list[Path]:
    if cv2 is None:
        raise RuntimeError(
            "OpenCV is not installed. Install requirements.txt to enable frame extraction "
            "or add an alternative backend."
        )

    ordered_frames = sorted(unique_preserve_order(frame_numbers))
    destination = ensure_dir(output_dir)
    capture = cv2.VideoCapture(str(video_path))
    saved_paths: list[Path] = []
    try:
        if not capture.isOpened():
            raise RuntimeError(f"failed to open video: {video_path}")
        for frame_number in ordered_frames:
            target_path = destination / f"frame_{frame_number:06d}.png"
            if target_path.exists() and not overwrite:
                saved_paths.append(target_path)
                continue
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"failed to decode frame {frame_number} from {video_path}")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            Image.fromarray(rgb).save(target_path)
            saved_paths.append(target_path)
    finally:
        capture.release()
    return saved_paths


def extract_frames_by_timestamp(video_path: str | Path, timestamps_sec: Iterable[float], output_dir: str | Path, overwrite: bool = False) -> list[Path]:
    fps = get_video_fps(video_path)
    frame_numbers = [int(math.floor(float(timestamp) * fps)) for timestamp in timestamps_sec]
    return extract_frames(video_path, frame_numbers, output_dir, overwrite=overwrite)


def build_contact_sheet(
    video_path: str | Path,
    output_path: str | Path,
    num_frames: int = 12,
    columns: int = 4,
    thumb_width: int = 320,
) -> Path:
    metadata = get_video_metadata(video_path)
    frame_count = max(int(metadata["frame_count"]), 1)
    frame_numbers = expand_frame_numbers([0, max(frame_count - 1, 0)], [], target_count=num_frames)
    temp_dir = ensure_dir(Path(output_path).parent / ".tmp_contact_sheet")
    frame_paths = extract_frames(video_path, frame_numbers, temp_dir, overwrite=True)
    images = [Image.open(frame_path).convert("RGB") for frame_path in frame_paths]

    if not images:
        raise RuntimeError(f"failed to extract contact-sheet frames from {video_path}")

    resized_images: list[Image.Image] = []
    labels: list[str] = []
    for frame_number, image in zip(frame_numbers, images):
        ratio = thumb_width / image.width
        resized = image.resize((thumb_width, max(1, int(image.height * ratio))))
        resized_images.append(resized)
        labels.append(f"Frame {frame_number}")

    rows = math.ceil(len(resized_images) / columns)
    row_height = max(image.height for image in resized_images) + 32
    sheet = Image.new("RGB", (columns * thumb_width, rows * row_height), color=(245, 245, 245))

    from PIL import ImageDraw

    draw = ImageDraw.Draw(sheet)
    for index, (image, label) in enumerate(zip(resized_images, labels)):
        row = index // columns
        column = index % columns
        x = column * thumb_width
        y = row * row_height
        sheet.paste(image, (x, y + 24))
        draw.text((x + 8, y + 4), label, fill=(20, 20, 20))

    final_path = Path(output_path)
    ensure_dir(final_path.parent)
    sheet.save(final_path)
    return final_path
