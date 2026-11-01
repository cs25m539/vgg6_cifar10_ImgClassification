"""
CSV-only reporting for W&B dashboards.

This module reads epoch_history.csv and experiment_summary.csv created by
Step 5. It never imports the model/data/training modules and never trains.
"""
from __future__ import annotations

import logging
import re
import time
from pathlib import Path

import pandas as pd

logger = logging.getLogger("vgg6.csv_reporting")

HISTORY_REQUIRED = {
    "run", "epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"
}
SUMMARY_REQUIRED = {
    "run", "activation", "optimizer", "batch_size", "learning_rate",
    "best_val_accuracy"
}
METRICS = ("train_loss", "train_accuracy", "val_loss", "val_accuracy")


def _read_and_validate(history_csv, summary_csv):
    logger.info("CSV LOAD START | history=%s summary=%s", history_csv, summary_csv)
    history = pd.read_csv(history_csv)
    summary = pd.read_csv(summary_csv)

    missing_h = sorted(HISTORY_REQUIRED - set(history.columns))
    missing_s = sorted(SUMMARY_REQUIRED - set(summary.columns))
    if missing_h:
        raise ValueError(f"epoch_history.csv is missing required columns: {missing_h}")
    if missing_s:
        raise ValueError(f"experiment_summary.csv is missing required columns: {missing_s}")
    if history.empty or summary.empty:
        raise ValueError("Both CSV files must contain at least one data row.")

    for col in ("epoch", *METRICS):
        history[col] = pd.to_numeric(history[col], errors="coerce")
    if history[list(("epoch", *METRICS))].isna().any().any():
        raise ValueError("History CSV contains missing or non-numeric epoch/metric values.")

    history["run"] = history["run"].astype(str)
    summary["run"] = summary["run"].astype(str)
    summary["best_val_accuracy"] = pd.to_numeric(
        summary["best_val_accuracy"], errors="coerce"
    )
    if summary["best_val_accuracy"].isna().any():
        raise ValueError("Summary CSV contains missing/non-numeric best_val_accuracy.")
    if history.duplicated(["run", "epoch"]).any():
        raise ValueError("History CSV has duplicate (run, epoch) rows.")

    history = history.sort_values(["run", "epoch"]).reset_index(drop=True)
    logger.info(
        "CSV LOAD COMPLETE | history_rows=%d summary_rows=%d history_runs=%d summary_runs=%d",
        len(history), len(summary), history["run"].nunique(), summary["run"].nunique()
    )
    history_runs = set(history["run"])
    summary_runs = set(summary["run"])
    if history_runs != summary_runs:
        logger.warning(
            "RUN SET MISMATCH | history_only=%s summary_only=%s",
            sorted(history_runs - summary_runs), sorted(summary_runs - history_runs)
        )
    return history, summary


def create_local_plots(history, output_dir):
    """Create four separate epoch curves and a validation-accuracy scatter per run."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_names = list(history["run"].drop_duplicates())
    logger.info("LOCAL PLOT STAGE START | runs=%d output=%s", len(run_names), output_dir)

    plot_specs = [
        ("train_loss", "Training loss", "Cross-entropy loss"),
        ("train_accuracy", "Training accuracy", "Accuracy (%)"),
        ("val_loss", "Validation loss", "Cross-entropy loss"),
        ("val_accuracy", "Validation accuracy", "Accuracy (%)"),
    ]
    for i, (run_name, frame) in enumerate(history.groupby("run", sort=False), start=1):
        logger.info("LOCAL PLOT RUN START | run=%s index=%d/%d", run_name, i, len(run_names))
        frame = frame.sort_values("epoch")
        safe = re.sub(r"[^A-Za-z0-9_-]+", "_", str(run_name)).strip("_") or f"run_{i}"
        for metric, title, ylabel in plot_specs:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(frame["epoch"], frame[metric], marker="o")
            ax.set(title=f"{title}: {run_name}", xlabel="Epoch", ylabel=ylabel)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(output_dir / f"{safe}_{metric}.png", dpi=160)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(frame["epoch"], frame["val_accuracy"])
        ax.set(
            title=f"Validation accuracy vs. step: {run_name}",
            xlabel="Step (epoch)", ylabel="Validation accuracy (%)"
        )
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / f"{safe}_validation_scatter.png", dpi=160)
        plt.close(fig)
        logger.info("LOCAL PLOT RUN COMPLETE | run=%s plots=5", run_name)

    logger.info("LOCAL PLOT STAGE COMPLETE | output=%s", output_dir)


def log_to_wandb(history, summary, project, entity=None):
    """
    Upload existing CSV results only. No training code is called.

    Creates:
      * one W&B run per experiment containing four epoch metrics;
      * one custom validation-accuracy scatter chart;
      * a configuration-results table and parallel-coordinates chart.
    """
    import wandb

    logger.info("WANDB LOGIN START")
    wandb.login()
    logger.info("WANDB LOGIN COMPLETE")

    # Metric curves: each original experiment is represented as one W&B run.
    grouped = list(history.groupby("run", sort=False))
    for index, (run_name, frame) in enumerate(grouped, start=1):
        logger.info(
            "WANDB CURVE RUN START | run=%s index=%d/%d",
            run_name, index, len(grouped)
        )
        matching = summary.loc[summary["run"] == run_name]
        config = matching.iloc[0].to_dict() if not matching.empty else {}
        init_args = {"project": f"{project}-curves", "name": str(run_name),
                     "config": config, "reinit": True}
        if entity:
            init_args["entity"] = entity
        with wandb.init(**init_args):
            for _, row in frame.sort_values("epoch").iterrows():
                step = int(row["epoch"])
                wandb.log({
                    "epoch": step,
                    "train_loss": float(row["train_loss"]),
                    "train_accuracy": float(row["train_accuracy"]),
                    "val_loss": float(row["val_loss"]),
                    "val_accuracy": float(row["val_accuracy"]),
                }, step=step)
        logger.info(
            "WANDB CURVE RUN COMPLETE | run=%s points=%d",
            run_name, len(frame)
        )

    # A single custom scatter chart with points from every experiment.
    logger.info("WANDB SCATTER STAGE START | points=%d", len(history))
    scatter_table = wandb.Table(
        dataframe=history[["run", "epoch", "val_accuracy"]].rename(
            columns={"epoch": "step"}
        )
    )
    init_args = {"project": f"{project}-scatter",
                 "name": "validation-accuracy-vs-step", "reinit": True}
    if entity:
        init_args["entity"] = entity
    with wandb.init(**init_args):
        wandb.log({
            "validation_accuracy_scatter": wandb.plot.scatter(
                scatter_table, "step", "val_accuracy",
                title="Validation accuracy vs. step"
            )
        })
    logger.info("WANDB SCATTER STAGE COMPLETE")

    # Configuration comparison table and W&B parallel-coordinates preset.
    candidate_columns = [
        "run", "activation", "optimizer", "batch_size", "learning_rate",
        "epochs_requested", "epochs", "best_epoch", "best_val_accuracy",
        "test_accuracy", "test_loss"
    ]
    columns = [c for c in candidate_columns if c in summary.columns]
    if "best_val_accuracy" not in columns:
        raise ValueError("Summary CSV must include best_val_accuracy for the parallel plot.")
    logger.info("WANDB CONFIGURATION STAGE START | rows=%d columns=%s",
                len(summary), columns)
    config_table = wandb.Table(dataframe=summary[columns])
    init_args = {"project": f"{project}-configurations",
                 "name": "configuration-results", "reinit": True}
    if entity:
        init_args["entity"] = entity
    with wandb.init(**init_args):
        payload = {"configuration_results": config_table}
        axes = [c for c in columns if c not in ("run", "test_accuracy", "test_loss")]
        try:
            payload["parallel_coordinates_validation_accuracy"] = wandb.plot_table(
                vega_spec_name="wandb/parallel-coordinates/v0",
                data_table=config_table,
                fields={
                    "label": "run",
                    "value": "best_val_accuracy",
                    "columns": axes,
                },
            )
            logger.info("WANDB PARALLEL COORDINATES CHART CREATED | axes=%s", axes)
        except (AttributeError, ValueError, KeyError, TypeError) as exc:
            logger.exception(
                "WANDB PARALLEL COORDINATES PRESET FAILED | reason=%s. "
                "The configuration table was still uploaded; add a Parallel Coordinates "
                "panel in the W&B workspace and select configuration_results.", exc
            )
        wandb.log(payload)
    logger.info("WANDB CONFIGURATION STAGE COMPLETE | project=%s-configurations", project)


def run_csv_reporting(history_csv, summary_csv, output_dir, project,
                      entity=None, local_plots=True, wandb_upload=True):
    """Top-level reporting workflow. Reads CSVs; does not invoke training."""
    started = time.perf_counter()
    logger.info("=" * 78)
    logger.info("CSV REPORTING START | training_enabled=false")
    history, summary = _read_and_validate(history_csv, summary_csv)

    if local_plots:
        create_local_plots(history, Path(output_dir) / "plots")
    if wandb_upload:
        log_to_wandb(history, summary, project=project, entity=entity)

    elapsed = time.perf_counter() - started
    logger.info("CSV REPORTING COMPLETE | elapsed_seconds=%.2f training_enabled=false", elapsed)
    return history, summary
