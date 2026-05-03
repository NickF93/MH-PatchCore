#!/usr/bin/env python3
"""CLI entrypoint for MH-PatchCore experiment execution."""

from __future__ import annotations

import logging
from pathlib import Path

import click

from mhpc.eval.cli_report import (
    render_loaded_pipeline_banner,
    save_run_reproducibility_artifacts,
)
from mhpc.eval.config import load_run_config
from mhpc.eval.pipeline import run_experiment
from mhpc.util.progress import (
    configure_progress_rendering,
    configure_root_logger,
)
from mhpc.util.repo_paths import resolve_repo_root


def _configure_logging(level: str = "INFO") -> None:
    configure_root_logger(level=level)


def _resolve_latest_run_dir(output_root: Path, experiment_name: str) -> Path:
    experiment_root = output_root / experiment_name
    if not experiment_root.exists() or not experiment_root.is_dir():
        raise ValueError(f"Experiment output directory does not exist: {experiment_root}")

    run_dirs = sorted(
        (
            path
            for path in experiment_root.iterdir()
            if path.is_dir() and (path / "metrics").is_dir()
        ),
        key=lambda path: path.name,
        reverse=True,
    )
    if not run_dirs:
        raise ValueError(f"No run directories with metrics were found under: {experiment_root}")
    return run_dirs[0]


@click.command()
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to an experiment YAML configuration file.",
)
def main(config_path: Path) -> None:
    """Run an MH-PatchCore experiment from a validated YAML config."""
    _configure_logging(level="INFO")
    logger = logging.getLogger(__name__)

    try:
        cfg = load_run_config(config_path)
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    _configure_logging(level=cfg.render.logging.level)
    configure_progress_rendering(
        enabled=cfg.render.progress.enabled,
        leave=cfg.render.progress.leave,
        dynamic_ncols=cfg.render.progress.dynamic_ncols,
        min_interval=cfg.render.progress.min_interval,
    )
    logger.info(
        "Loaded config=%s training_contract=%s",
        config_path,
        cfg.training.contract,
    )
    click.echo(render_loaded_pipeline_banner(cfg=cfg, config_path=config_path))

    summary_df = run_experiment(cfg)
    if cfg.artifacts.enabled:
        run_dir = _resolve_latest_run_dir(
            output_root=cfg.paths.output_root,
            experiment_name=cfg.experiment.name,
        )
        save_run_reproducibility_artifacts(
            run_dir=run_dir,
            config_path=config_path,
            repo_root=resolve_repo_root(__file__),
        )
        logger.info("Saved run reproducibility artifacts under %s", run_dir)

    click.echo("\nExperiment summary:\n")
    click.echo(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
