from __future__ import annotations

import os
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd

from .utils import ensure_dir, read_jsonl


_MPL_DIR = Path.cwd() / ".cache" / "matplotlib"
_MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULT_COLUMNS = [
    "failure_id",
    "category",
    "source_video",
    "frame_range",
    "frame_ids_used",
    "human_judgment",
    "question",
    "model",
    "prompt_type",
    "vlm_response",
    "vlm_confidence",
    "blind_spot_flag",
    "parse_method",
]


def load_results_dataframe(results_path: str | Path) -> pd.DataFrame:
    rows = read_jsonl(results_path)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Normalize field names from JSONL (judgment/confidence) to analysis names (vlm_judgment/vlm_confidence)
    df = df.rename(columns={"judgment": "vlm_judgment", "confidence": "vlm_confidence", "targeted_question": "question", "raw_response": "vlm_response"})
    if "blind_spot_flag" in df.columns:
        df["blind_spot_flag"] = df["blind_spot_flag"].astype(bool)
    if "vlm_confidence" in df.columns:
        df["vlm_confidence"] = pd.to_numeric(df["vlm_confidence"], errors="coerce")
    return df


def baseline_only(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df[df["prompt_type"] == "baseline"].copy()


def _format_rate(rate: float, n: int) -> str:
    return f"{rate:.0%} (n={n})"


def blind_spot_rate_by_category_and_model(df: pd.DataFrame) -> pd.DataFrame:
    baseline = baseline_only(df)
    if baseline.empty:
        return pd.DataFrame(columns=["category", "model", "total_cases", "blind_spot_cases", "blind_spot_rate", "label"])
    grouped = (
        baseline.groupby(["category", "model"], dropna=False)
        .agg(total_cases=("failure_id", "count"), blind_spot_cases=("blind_spot_flag", "sum"))
        .reset_index()
    )
    grouped["blind_spot_rate"] = grouped["blind_spot_cases"] / grouped["total_cases"]
    grouped["label"] = grouped.apply(lambda row: _format_rate(float(row["blind_spot_rate"]), int(row["total_cases"])), axis=1)
    return grouped.sort_values(["category", "model"]).reset_index(drop=True)


def cross_model_correlation(df: pd.DataFrame) -> pd.DataFrame:
    baseline = baseline_only(df)
    if baseline.empty:
        return pd.DataFrame(columns=["model_a", "model_b", "overlap_count", "union_count", "jaccard"])
    blind_spots = baseline[baseline["blind_spot_flag"]]
    model_to_failures = {
        model: set(group["failure_id"].tolist())
        for model, group in blind_spots.groupby("model", dropna=False)
    }
    models = sorted(set(baseline["model"].dropna().tolist()))
    records: list[dict[str, Any]] = []
    for model_a, model_b in combinations(models, 2):
        failures_a = model_to_failures.get(model_a, set())
        failures_b = model_to_failures.get(model_b, set())
        overlap = failures_a & failures_b
        union = failures_a | failures_b
        jaccard = len(overlap) / len(union) if union else 0.0
        records.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "overlap_count": len(overlap),
                "union_count": len(union),
                "jaccard": jaccard,
            }
        )
    return pd.DataFrame(records)


def confidence_calibration(df: pd.DataFrame) -> pd.DataFrame:
    baseline = baseline_only(df)
    if baseline.empty:
        return pd.DataFrame(columns=["model", "outcome", "n", "mean_confidence", "median_confidence"])
    work = baseline.copy()
    work["outcome"] = work["blind_spot_flag"].map({True: "blind_spot", False: "correct_detection"})
    summary = (
        work.groupby(["model", "outcome"], dropna=False)
        .agg(
            n=("failure_id", "count"),
            mean_confidence=("vlm_confidence", "mean"),
            median_confidence=("vlm_confidence", "median"),
        )
        .reset_index()
    )
    return summary


def rescue_rate_analysis(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "prompt_type",
                "category",
                "model",
                "baseline_blind_spots",
                "rescued_cases",
                "rescue_rate",
            ]
        )

    baseline = baseline_only(df)
    baseline_blind_spots = baseline[baseline["blind_spot_flag"]].copy()
    if baseline_blind_spots.empty:
        return pd.DataFrame(columns=["prompt_type", "category", "model", "baseline_blind_spots", "rescued_cases", "rescue_rate"])

    records: list[dict[str, Any]] = []
    for prompt_type in ("probe_a", "probe_b"):
        probes = df[df["prompt_type"] == prompt_type].copy()
        if probes.empty:
            continue
        merged = baseline_blind_spots.merge(
            probes[["failure_id", "model", "prompt_type", "vlm_judgment"]],
            on=["failure_id", "model"],
            how="left",
            suffixes=("_baseline", "_probe"),
        )
        merged["rescued"] = merged["vlm_judgment_probe"].eq("FAIL")
        summary = (
            merged.groupby(["category", "model"], dropna=False)
            .agg(
                baseline_blind_spots=("failure_id", "count"),
                rescued_cases=("rescued", "sum"),
            )
            .reset_index()
        )
        summary["rescue_rate"] = summary["rescued_cases"] / summary["baseline_blind_spots"]
        summary["prompt_type"] = prompt_type
        records.extend(summary.to_dict(orient="records"))
    return pd.DataFrame(records)


def generate_tsv(df: pd.DataFrame, output_path: str | Path) -> Path:
    ensure_dir(Path(output_path).parent)
    if df.empty:
        pd.DataFrame(columns=RESULT_COLUMNS).to_csv(output_path, sep="\t", index=False)
        return Path(output_path)

    table = df.copy()
    table["frame_range"] = table["frame_range"].apply(lambda value: ",".join(str(item) for item in value) if isinstance(value, list) else value)
    table["frame_ids_used"] = table["frame_ids_used"].apply(lambda value: ",".join(str(item) for item in value) if isinstance(value, list) else value)
    table["image_paths_used"] = table["image_paths_used"].apply(lambda value: ",".join(value) if isinstance(value, list) else value)
    table = table.rename(
        columns={
            "targeted_question": "question",
            "raw_response": "vlm_response",
            "confidence": "vlm_confidence",
            "judgment": "vlm_judgment",
        }
    )
    columns = [
        "failure_id",
        "category",
        "source_video",
        "frame_range",
        "frame_ids_used",
        "human_judgment",
        "question",
        "model",
        "prompt_type",
        "vlm_judgment",
        "vlm_response",
        "vlm_confidence",
        "blind_spot_flag",
        "parse_method",
        "image_paths_used",
    ]
    table.to_csv(output_path, sep="\t", index=False, columns=[column for column in columns if column in table.columns])
    return Path(output_path)


def _save_csv(df: pd.DataFrame, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    df.to_csv(output_path, index=False)


def _plot_blind_spot_heatmap(summary: pd.DataFrame, output_dir: Path) -> None:
    if summary.empty:
        return
    pivot = summary.pivot(index="category", columns="model", values="blind_spot_rate").fillna(0.0)
    labels = summary.pivot(index="category", columns="model", values="label").fillna("")
    fig, ax = plt.subplots(figsize=(8, max(4, len(pivot.index) * 0.8)))
    image = ax.imshow(pivot.values, aspect="auto", cmap="Reds", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for row_index, category in enumerate(pivot.index):
        for col_index, model in enumerate(pivot.columns):
            ax.text(col_index, row_index, labels.loc[category, model], ha="center", va="center", fontsize=8)
    ax.set_title("Blind Spot Rate by Category and Model")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "blind_spot_heatmap.png", dpi=200)
    plt.close(fig)


def _plot_confidence_boxplot(df: pd.DataFrame, output_dir: Path) -> None:
    baseline = baseline_only(df)
    if baseline.empty:
        return
    work = baseline.copy()
    work["outcome"] = work["blind_spot_flag"].map({True: "blind_spot", False: "correct_detection"})
    groups = []
    labels = []
    for model in sorted(work["model"].dropna().unique()):
        for outcome in ("blind_spot", "correct_detection"):
            subset = work[(work["model"] == model) & (work["outcome"] == outcome)]["vlm_confidence"].dropna()
            if subset.empty:
                continue
            groups.append(subset.tolist())
            labels.append(f"{model}\n{outcome}\n(n={len(subset)})")
    if not groups:
        return
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.5), 5))
    ax.boxplot(groups, tick_labels=labels)
    ax.set_ylabel("Confidence")
    ax.set_title("Baseline Confidence by Outcome")
    fig.tight_layout()
    fig.savefig(output_dir / "confidence_boxplot.png", dpi=200)
    plt.close(fig)


def _plot_rescue_rates(summary: pd.DataFrame, output_dir: Path) -> None:
    if summary.empty:
        return
    grouped = (
        summary.groupby("prompt_type", dropna=False)
        .agg(total_baseline_blind_spots=("baseline_blind_spots", "sum"), rescued_cases=("rescued_cases", "sum"))
        .reset_index()
    )
    grouped["rescue_rate"] = grouped["rescued_cases"] / grouped["total_baseline_blind_spots"]
    labels = [f"{row.prompt_type}\n(n={int(row.total_baseline_blind_spots)})" for row in grouped.itertuples()]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, grouped["rescue_rate"], color=["#4C72B0", "#55A868"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Rescue rate")
    ax.set_title("Probe Rescue Rates")
    for index, row in enumerate(grouped.itertuples()):
        ax.text(index, row.rescue_rate + 0.03, f"{row.rescue_rate:.0%}", ha="center")
    fig.tight_layout()
    fig.savefig(output_dir / "rescue_rates.png", dpi=200)
    plt.close(fig)


def run_analysis(results_path: str | Path, output_dir: str | Path) -> dict[str, Path]:
    output_root = ensure_dir(output_dir)
    figures_dir = ensure_dir(output_root / "figures")
    df = load_results_dataframe(results_path)

    blind_spot_summary = blind_spot_rate_by_category_and_model(df)
    correlation_summary = cross_model_correlation(df)
    confidence_summary = confidence_calibration(df)
    rescue_summary = rescue_rate_analysis(df)

    blind_spot_csv = output_root / "summary_blind_spot_rates.csv"
    correlation_csv = output_root / "summary_correlation.csv"
    confidence_csv = output_root / "summary_confidence.csv"
    rescue_csv = output_root / "summary_rescue_rates.csv"

    _save_csv(blind_spot_summary, blind_spot_csv)
    _save_csv(correlation_summary, correlation_csv)
    _save_csv(confidence_summary, confidence_csv)
    _save_csv(rescue_summary, rescue_csv)

    _plot_blind_spot_heatmap(blind_spot_summary, figures_dir)
    _plot_confidence_boxplot(df, figures_dir)
    _plot_rescue_rates(rescue_summary, figures_dir)

    return {
        "blind_spot_summary": blind_spot_csv,
        "correlation_summary": correlation_csv,
        "confidence_summary": confidence_csv,
        "rescue_summary": rescue_csv,
        "figures_dir": figures_dir,
    }

