"""Cross-run experiment summarization utilities.

This module scans experiment run directories, aggregates global metrics from
run-level CSV artifacts, and writes deterministic summary outputs.
"""

from __future__ import annotations

import logging
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

LOGGER = logging.getLogger(__name__)

SUMMARY_STATUS_OK = "ok"
SUMMARY_STATUS_FAILED = "failed"

RESOURCE_AGGREGATE_COLUMNS: tuple[str, ...] = (
    "max_ram_mb",
    "max_vram_mb",
    "max_cpu_pct",
    "max_gpu_util_pct",
    "mean_ram_mb",
    "mean_vram_mb",
    "mean_cpu_pct",
    "mean_gpu_util_pct",
)

TIMING_AGGREGATE_COLUMNS: tuple[str, ...] = (
    "run_total_time_s",
    "sum_fit_s",
    "sum_calibration_s",
    "sum_infer_s",
    "sum_metrics_s",
    "sum_artifacts_s",
)

_RESOURCE_SOURCE_COLUMNS: tuple[str, ...] = (
    "ram_mb",
    "vram_mb",
    "cpu_pct",
    "gpu_util_pct",
)


@dataclass(frozen=True)
class RunSummaryRecord:
    """Summary of one experiment run."""

    experiment_name: str
    run_timestamp: str
    run_dir: str
    status: str
    error: str
    metrics: dict[str, float]
    resources: dict[str, float]
    timings: dict[str, float]


def discover_run_dirs(root: Path) -> list[Path]:
    """Discover candidate run directories under an experiment root.

    A candidate run is any directory matching:
    ``<root>/<experiment_name>/<run_dir>/metrics``.
    """
    if not root.exists():
        raise ValueError(f"Root path does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"Root path is not a directory: {root}")

    run_dirs: list[Path] = []
    for experiment_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for run_dir in sorted(path for path in experiment_dir.iterdir() if path.is_dir()):
            if (run_dir / "metrics").is_dir():
                run_dirs.append(run_dir)
    return run_dirs


def summarize_run(run_dir: Path) -> RunSummaryRecord:
    """Summarize one run directory into a typed record."""
    experiment_name = run_dir.parent.name
    run_timestamp = run_dir.name
    metrics_dir = run_dir / "metrics"

    status = SUMMARY_STATUS_OK
    errors: list[str] = []

    metrics: dict[str, float] = {}
    resources = {name: float("nan") for name in RESOURCE_AGGREGATE_COLUMNS}
    timings = {name: float("nan") for name in TIMING_AGGREGATE_COLUMNS}

    try:
        metrics, warning = _parse_summary_metrics(metrics_dir / "summary.csv")
        if warning:
            errors.append(warning)
    except Exception as exc:
        status = SUMMARY_STATUS_FAILED
        errors.append(str(exc))

    try:
        resources = _parse_resource_aggregates(metrics_dir / "profiling_samples.csv")
    except Exception as exc:
        status = SUMMARY_STATUS_FAILED
        errors.append(str(exc))

    try:
        timings = _parse_timing_aggregates(metrics_dir / "profiling_timings.csv")
    except Exception as exc:
        status = SUMMARY_STATUS_FAILED
        errors.append(str(exc))

    error_msg = "; ".join(errors)
    if status == SUMMARY_STATUS_FAILED:
        LOGGER.warning(
            "Run summarization failed: run_dir=%s error=%s",
            run_dir,
            error_msg,
        )

    return RunSummaryRecord(
        experiment_name=experiment_name,
        run_timestamp=run_timestamp,
        run_dir=str(run_dir),
        status=status,
        error=error_msg,
        metrics=metrics,
        resources=resources,
        timings=timings,
    )


def build_summary_table(records: list[RunSummaryRecord]) -> pd.DataFrame:
    """Build a deterministic summary dataframe from run records."""
    metric_columns = sorted(
        {
            metric_name
            for record in records
            for metric_name in record.metrics
        }
    )
    resource_columns = list(RESOURCE_AGGREGATE_COLUMNS)
    timing_columns = list(TIMING_AGGREGATE_COLUMNS)

    rows: list[dict[str, Any]] = []
    for record in records:
        row: dict[str, Any] = {
            "experiment_name": record.experiment_name,
            "run_timestamp": record.run_timestamp,
            "run_dir": record.run_dir,
            "status": record.status,
            "error": record.error,
        }
        for column in metric_columns:
            row[column] = record.metrics.get(column, float("nan"))
        for column in resource_columns:
            row[column] = record.resources.get(column, float("nan"))
        for column in timing_columns:
            row[column] = record.timings.get(column, float("nan"))
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    return (
        df.sort_values(
            by=["experiment_name", "run_timestamp", "run_dir"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def summarize_experiment_root(
    root: Path,
    output_csv_name: str = "experiments_global_summary.csv",
    output_md_name: str = "experiments_global_summary.md",
    *,
    write_markdown: bool = True,
) -> tuple[pd.DataFrame, Path, Path | None, int]:
    """Summarize all runs in one experiment root and write outputs."""
    run_dirs = discover_run_dirs(root)
    if not run_dirs:
        raise ValueError(
            f"No run directories discovered under root: {root}. "
            "Expected <root>/<experiment>/<run>/metrics."
        )

    records = [summarize_run(run_dir) for run_dir in run_dirs]
    summary_df = build_summary_table(records)
    if summary_df.empty:
        raise ValueError(f"No summary rows produced for root: {root}")

    csv_path = root / output_csv_name
    write_summary_csv(summary_df, csv_path)

    md_path: Path | None = None
    if write_markdown:
        md_path = root / output_md_name
        write_markdown_report(
            df=summary_df,
            out_path=md_path,
            root=root,
            csv_path=csv_path,
        )

    failed_count = int((summary_df["status"] == SUMMARY_STATUS_FAILED).sum())
    return summary_df, csv_path, md_path, failed_count


def write_summary_csv(df: pd.DataFrame, out_path: Path) -> None:
    """Write summary dataframe atomically as CSV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_dataframe_atomic(df=df, out_path=out_path)


def write_markdown_report(
    df: pd.DataFrame,
    out_path: Path,
    root: Path,
    csv_path: Path,
) -> None:
    """Write a human-readable Markdown report for cross-run summaries."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    total_runs = int(df.shape[0])
    failed_runs = int((df["status"] == SUMMARY_STATUS_FAILED).sum())
    ok_runs = total_runs - failed_runs

    lines: list[str] = []
    lines.append("# Experiments Global Summary")
    lines.append("")
    lines.append(f"- Generated: `{generated_at}`")
    lines.append(f"- Root: `{root}`")
    lines.append(f"- Total runs: `{total_runs}`")
    lines.append(f"- OK runs: `{ok_runs}`")
    lines.append(f"- Failed runs: `{failed_runs}`")
    lines.append("")

    lines.extend(
        _render_leaderboard_sections(
            df=df,
            title="Metric Leaderboards",
            metrics=[
                "image_auroc",
                "image_ap",
                "pixel_auroc",
                "pixel_ap",
                "image_f1",
                "pixel_f1",
            ],
        )
    )
    lines.extend(
        _render_leaderboard_sections(
            df=df,
            title="Resource Peak Leaderboards",
            metrics=[
                "max_ram_mb",
                "max_vram_mb",
                "max_cpu_pct",
                "max_gpu_util_pct",
            ],
        )
    )

    lines.append("## Runtime")
    lines.append("")
    lines.extend(
        _render_fastest_slowest_section(
            df=df,
            metric="run_total_time_s",
            top_k=10,
        )
    )

    lines.append("## Failed Runs")
    lines.append("")
    failed_df = df[df["status"] == SUMMARY_STATUS_FAILED][
        ["experiment_name", "run_timestamp", "run_dir", "error"]
    ]
    if failed_df.empty:
        lines.append("No failed runs detected.")
        lines.append("")
    else:
        lines.extend(_render_markdown_table(failed_df))
        lines.append("")

    lines.append("## CSV Reference")
    lines.append("")
    lines.append(
        "- Full machine-readable table: "
        f"`{csv_path}` "
        f"({df.shape[0]} rows, {df.shape[1]} columns)"
    )
    lines.append("")

    _write_text_atomic(text="\n".join(lines), out_path=out_path)


def _parse_summary_metrics(summary_csv: Path) -> tuple[dict[str, float], str]:
    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing required file: {summary_csv}")

    try:
        df = pd.read_csv(summary_csv)
    except Exception as exc:
        raise ValueError(f"Failed to parse summary CSV {summary_csv}: {exc}") from exc

    if "dataset" not in df.columns:
        raise ValueError(f"summary.csv missing required column 'dataset': {summary_csv}")

    numeric_df = df.copy()
    for column in (column for column in df.columns if column != "dataset"):
        numeric_df[column] = pd.to_numeric(numeric_df[column], errors="coerce")
    metric_columns = [
        column
        for column in numeric_df.columns
        if column != "dataset" and numeric_df[column].notna().any()
    ]
    if not metric_columns:
        raise ValueError(f"summary.csv has no numeric metric columns: {summary_csv}")

    mean_rows = numeric_df[numeric_df["dataset"].astype(str) == "MEAN"]
    warning = ""
    if not mean_rows.empty:
        metric_series = mean_rows.iloc[0][metric_columns]
    else:
        non_mean_rows = numeric_df[numeric_df["dataset"].astype(str) != "MEAN"]
        if non_mean_rows.empty:
            raise ValueError(
                f"summary.csv has no MEAN row and no dataset rows to aggregate: {summary_csv}"
            )
        metric_series = non_mean_rows[metric_columns].mean(axis=0, skipna=True)
        warning = (
            "MEAN row missing in summary.csv; using numeric mean across dataset rows."
        )

    metrics: dict[str, float] = {}
    for column in metric_columns:
        value = metric_series[column]
        metrics[column] = (
            float(value) if pd.notna(value) else float("nan")
        )
    return metrics, warning


def _parse_resource_aggregates(samples_csv: Path) -> dict[str, float]:
    if not samples_csv.exists():
        raise FileNotFoundError(f"Missing required file: {samples_csv}")

    try:
        df = pd.read_csv(samples_csv)
    except Exception as exc:
        raise ValueError(
            f"Failed to parse profiling samples CSV {samples_csv}: {exc}"
        ) from exc

    missing_columns = [
        column for column in _RESOURCE_SOURCE_COLUMNS if column not in df.columns
    ]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(
            f"profiling_samples.csv missing required columns [{missing}]: {samples_csv}"
        )

    for column in _RESOURCE_SOURCE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return {
        "max_ram_mb": _safe_float(df["ram_mb"].max(skipna=True)),
        "max_vram_mb": _safe_float(df["vram_mb"].max(skipna=True)),
        "max_cpu_pct": _safe_float(df["cpu_pct"].max(skipna=True)),
        "max_gpu_util_pct": _safe_float(df["gpu_util_pct"].max(skipna=True)),
        "mean_ram_mb": _safe_float(df["ram_mb"].mean(skipna=True)),
        "mean_vram_mb": _safe_float(df["vram_mb"].mean(skipna=True)),
        "mean_cpu_pct": _safe_float(df["cpu_pct"].mean(skipna=True)),
        "mean_gpu_util_pct": _safe_float(df["gpu_util_pct"].mean(skipna=True)),
    }


def _parse_timing_aggregates(timings_csv: Path) -> dict[str, float]:
    if not timings_csv.exists():
        raise FileNotFoundError(f"Missing required file: {timings_csv}")

    try:
        df = pd.read_csv(timings_csv)
    except Exception as exc:
        raise ValueError(
            f"Failed to parse profiling timings CSV {timings_csv}: {exc}"
        ) from exc

    required_columns = {"phase", "duration_s"}
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(
            f"profiling_timings.csv missing required columns [{missing}]: {timings_csv}"
        )

    df["duration_s"] = pd.to_numeric(df["duration_s"], errors="coerce")
    phase_sums = df.groupby("phase", dropna=True)["duration_s"].sum(min_count=1)

    return {
        "run_total_time_s": _safe_float(phase_sums.get("total", float("nan"))),
        "sum_fit_s": _safe_float(phase_sums.get("fit", 0.0)),
        "sum_calibration_s": _safe_float(phase_sums.get("calibration", 0.0)),
        "sum_infer_s": _safe_float(phase_sums.get("infer", 0.0)),
        "sum_metrics_s": _safe_float(phase_sums.get("metrics", 0.0)),
        "sum_artifacts_s": _safe_float(phase_sums.get("artifacts", 0.0)),
    }


def _safe_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if math.isnan(parsed):
        return float("nan")
    return parsed


def _render_leaderboard_sections(
    df: pd.DataFrame,
    title: str,
    metrics: list[str],
) -> list[str]:
    lines: list[str] = []
    lines.append(f"## {title}")
    lines.append("")

    ok_df = df[df["status"] == SUMMARY_STATUS_OK]
    if ok_df.empty:
        lines.append("No successful runs available for ranking.")
        lines.append("")
        return lines

    for metric in metrics:
        if metric not in ok_df.columns:
            continue
        metric_values = pd.to_numeric(ok_df[metric], errors="coerce")
        ranked_df = ok_df.loc[metric_values.notna()].copy()
        if ranked_df.empty:
            continue
        ranked_df[metric] = pd.to_numeric(ranked_df[metric], errors="coerce")
        top_df = ranked_df.nlargest(10, metric)[
            ["experiment_name", "run_timestamp", metric, "run_dir"]
        ]
        lines.append(f"### Top 10 by `{metric}`")
        lines.append("")
        lines.extend(_render_markdown_table(top_df))
        lines.append("")

    return lines


def _render_fastest_slowest_section(
    df: pd.DataFrame,
    metric: str,
    top_k: int,
) -> list[str]:
    lines: list[str] = []
    if metric not in df.columns:
        lines.append(f"Metric `{metric}` not available.")
        lines.append("")
        return lines

    ok_df = df[df["status"] == SUMMARY_STATUS_OK].copy()
    ok_df[metric] = pd.to_numeric(ok_df[metric], errors="coerce")
    ok_df = ok_df[ok_df[metric].notna()]
    if ok_df.empty:
        lines.append("No successful runs with valid runtime information.")
        lines.append("")
        return lines

    fastest_df = ok_df.nsmallest(top_k, metric)[
        ["experiment_name", "run_timestamp", metric, "run_dir"]
    ]
    slowest_df = ok_df.nlargest(top_k, metric)[
        ["experiment_name", "run_timestamp", metric, "run_dir"]
    ]

    lines.append(f"### Fastest {top_k} by `{metric}`")
    lines.append("")
    lines.extend(_render_markdown_table(fastest_df))
    lines.append("")

    lines.append(f"### Slowest {top_k} by `{metric}`")
    lines.append("")
    lines.extend(_render_markdown_table(slowest_df))
    lines.append("")
    return lines


def _render_markdown_table(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return ["No rows.", ""]

    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        values = [_format_markdown_value(row[column]) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _format_markdown_value(value: Any) -> str:
    if pd.isna(value):
        return "NaN"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _write_dataframe_atomic(df: pd.DataFrame, out_path: Path) -> None:
    temp_path = _make_temp_path(out_path)
    try:
        df.to_csv(temp_path, index=False)
        temp_path.replace(out_path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def _write_text_atomic(text: str, out_path: Path) -> None:
    temp_path = _make_temp_path(out_path)
    try:
        temp_path.write_text(text, encoding="utf-8")
        temp_path.replace(out_path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def _make_temp_path(target_path: Path) -> Path:
    fd, raw_path = tempfile.mkstemp(
        prefix=f".{target_path.name}.tmp.",
        dir=target_path.parent,
    )
    os.close(fd)
    return Path(raw_path)
