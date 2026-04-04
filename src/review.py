"""Generate a single-file HTML review of all annotated failures."""
from __future__ import annotations

import base64
import json
from pathlib import Path

try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None


def _get_fps(video_path: Path) -> float | None:
    if cv2 is None or not video_path.exists():
        return None
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps > 0 else None


CATEGORY_COLORS = {
    "shadow_lighting_physics": ("#fef9c3", "#92400e"),
    "character_consistency": ("#dbeafe", "#1e40af"),
    "object_permanence_logical_consistency": ("#f3e8ff", "#6b21a8"),
    "camera_movement_impossibilities": ("#ffedd5", "#9a3412"),
    "text_logo_corruption": ("#dcfce7", "#166534"),
    "temporal_coherence": ("#fce7f3", "#9d174d"),
}


def _img_to_data_url(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/png;base64,{data}"


def _frame_label(filename: str, fps: float | None = None) -> str:
    # frame_000030.png -> Frame 30 (1.2s)
    stem = Path(filename).stem  # frame_000030
    try:
        num = int(stem.split("_")[-1])
        if fps and fps > 0:
            sec = num / fps
            return f"Frame {num} ({sec:.2f}s)"
        return f"Frame {num}"
    except ValueError:
        return stem


def _build_failure_section(failure: dict, frames_dir: Path, videos_dir: Path | None = None) -> str:
    fid = failure["id"]
    video_stem = Path(failure["source_video"]).stem
    frame_dir = frames_dir / fid / "baseline" / video_stem
    frame_paths = sorted(frame_dir.glob("frame_*.png")) if frame_dir.exists() else []

    fps = None
    if videos_dir is not None:
        fps = _get_fps(videos_dir / failure["source_video"])

    bg, fg = CATEGORY_COLORS.get(failure.get("category", ""), ("#e5e7eb", "#374151"))
    sev = failure.get("severity", "")
    sev_colors = {"high": ("#fee2e2", "#991b1b"), "medium": ("#fef9c3", "#92400e"), "low": ("#dcfce7", "#166534")}
    sev_bg, sev_fg = sev_colors.get(sev, ("#e5e7eb", "#374151"))

    img_html = ""
    if frame_paths:
        for p in frame_paths:
            label = _frame_label(p.name, fps=fps)
            url = _img_to_data_url(p)
            img_html += (
                f'<div class="frame">'
                f'<img src="{url}" alt="{label}">'
                f'<div class="frame-label">{label}</div>'
                f"</div>"
            )
    else:
        img_html = '<p style="color:#9ca3af;font-style:italic">No frames extracted yet</p>'

    noticeability = failure.get("human_noticeability", "")
    stars = "★" * int(noticeability) + "☆" * (5 - int(noticeability)) if noticeability else ""

    rows = [
        ("Category", f'<span class="tag" style="background:{bg};color:{fg}">{failure.get("category","")}</span>'),
        ("Severity", f'<span class="tag" style="background:{sev_bg};color:{sev_fg}">{sev}</span>'),
        ("Difficulty", failure.get("difficulty", "")),
        ("Human noticeability", f'{noticeability}/5 {stars}'),
        ("Human judgment", failure.get("human_judgment", "")),
        ("Targeted question", f'<strong>{failure.get("targeted_question","")}</strong>'),
        ("Expected answer", failure.get("expected_answer", "")),
        ("Video", f'{failure.get("source_video","")} &nbsp; frames {failure.get("key_frames","")} &nbsp; ts {failure.get("timestamp_sec","")}s'),
    ]
    table_rows = "".join(
        f'<tr><th>{k}</th><td>{v}</td></tr>' for k, v in rows
    )

    return f"""
<div class="failure" id="{fid}">
  <div class="failure-header">
    <span class="fid">{fid}</span>
    <span class="video">{failure.get("source_video","")}</span>
  </div>
  <div class="frames">{img_html}</div>
  <table class="meta">{table_rows}</table>
</div>
"""


CSS = """
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       margin: 0; padding: 24px; background: #f3f4f6; color: #111827; }
h1 { font-size: 20px; margin: 0 0 20px; color: #1f2937; }
.failure { background: white; border-radius: 10px; padding: 20px;
           margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,.08); }
.failure-header { display: flex; align-items: center; gap: 12px; margin-bottom: 14px; }
.fid { font-size: 18px; font-weight: 700; color: #111827; }
.video { font-size: 13px; color: #6b7280; }
.frames { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 14px; }
.frame { text-align: center; }
.frame img { max-height: 220px; max-width: 360px; border-radius: 6px;
             border: 1px solid #e5e7eb; display: block; }
.frame-label { font-size: 11px; color: #6b7280; margin-top: 4px; }
table.meta { width: 100%; border-collapse: collapse; font-size: 13.5px; }
table.meta th { width: 170px; text-align: left; padding: 5px 10px; color: #6b7280;
                vertical-align: top; white-space: nowrap; font-weight: 500; }
table.meta td { padding: 5px 10px; vertical-align: top; }
table.meta tr:nth-child(odd) td { background: #f9fafb; }
.tag { display: inline-block; padding: 2px 8px; border-radius: 4px;
       font-size: 12px; font-weight: 500; }
"""


def generate_review_html(annotations_path: Path, frames_dir: Path, output_path: Path) -> None:
    payload = json.loads(annotations_path.read_text(encoding="utf-8"))
    failures = payload.get("failures", [])

    videos_dir = annotations_path.parent / "videos"
    sections = [_build_failure_section(f, frames_dir, videos_dir=videos_dir) for f in failures]

    nav_links = " &nbsp;·&nbsp; ".join(
        f'<a href="#{f["id"]}">{f["id"]}</a>' for f in failures
    )

    html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<title>VLM Blind Spot Review</title>
<style>{CSS}</style>
</head><body>
<h1>VLM Blind Spot Review — {len(failures)} failures</h1>
<p style="font-size:13px;color:#6b7280;margin-bottom:20px">{nav_links}</p>
{''.join(sections)}
</body></html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"Review: {output_path}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Generate HTML review of annotated failures")
    p.add_argument("--annotations", default="data/annotations.json")
    p.add_argument("--frames-dir", default="data/frames")
    p.add_argument("--output", default="output/review.html")
    args = p.parse_args()

    root = Path(".")
    generate_review_html(
        annotations_path=root / args.annotations,
        frames_dir=root / args.frames_dir,
        output_path=root / args.output,
    )