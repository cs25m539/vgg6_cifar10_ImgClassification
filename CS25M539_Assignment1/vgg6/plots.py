"""Save required local plots and log charts/tables to Weights & Biases."""
import logging
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger("vgg6.plots")

def save_plots(history_df: pd.DataFrame, output_dir, logger_override=None):
    log = logger_override or logger
    out = Path(output_dir) / "plots"
    out.mkdir(parents=True, exist_ok=True)
    log.info("PLOT GENERATION START | output_dir=%s run_count=%d", out, history_df["run"].nunique() if not history_df.empty else 0)
    # Four separate required plots, plus a combined convenience figure per run.
    for run_index, (run_name, frame) in enumerate(history_df.groupby("run", sort=False), start=1):
        log.info("PLOT RUN START | run=%s index=%d/%d", run_name, run_index, history_df["run"].nunique())
        frame = frame.sort_values("epoch")
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(run_name))
        specs = [
            ("train_loss", "Training loss", "Cross-entropy loss"),
            ("train_accuracy", "Training accuracy", "Accuracy (%)"),
            ("val_loss", "Validation loss", "Cross-entropy loss"),
            ("val_accuracy", "Validation accuracy", "Accuracy (%)"),
        ]
        for col, title, ylabel in specs:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(frame["epoch"], frame[col], marker="o")
            ax.set(title=f"{title}: {run_name}", xlabel="Epoch", ylabel=ylabel)
            ax.grid(True, alpha=.3); fig.tight_layout()
            fig.savefig(out / f"{safe}_{col}.png", dpi=150); plt.close(fig)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(frame["epoch"], frame["val_accuracy"])
        ax.set(title=f"Validation accuracy vs. step: {run_name}",
               xlabel="Step (epoch)", ylabel="Validation accuracy (%)")
        ax.grid(True, alpha=.3); fig.tight_layout()
        fig.savefig(out / f"{safe}_validation_scatter.png", dpi=150); plt.close(fig)
        log.info("PLOT RUN COMPLETE | run=%s files=5", run_name)
    log.info("PLOT GENERATION COMPLETE | directory=%s", out)

def log_wandb(history_df, summary_df, project="vgg6-cifar10", logger_override=None):
    """Log each configuration as its own run and log a parallel-coordinate plot table."""
    log = logger_override or logger
    import wandb
    log.info("WANDB LOGIN START")
    wandb.login()
    log.info("WANDB LOGIN COMPLETE")
    for run_index, (run_name, frame) in enumerate(history_df.groupby("run", sort=False), start=1):
        log.info("WANDB RUN START | run=%s index=%d/%d project=%s-curves", run_name, run_index, history_df["run"].nunique(), project)
        config_row = summary_df.loc[summary_df["run"] == run_name]
        cfg = config_row.iloc[0].to_dict() if not config_row.empty else {}
        with wandb.init(project=f"{project}-curves", name=str(run_name), config=cfg, reinit=True):
            for _, r in frame.sort_values("epoch").iterrows():
                wandb.log({
                    "epoch": int(r["epoch"]),
                    "train_loss": float(r["train_loss"]),
                    "train_accuracy": float(r["train_accuracy"]),
                    "val_loss": float(r["val_loss"]),
                    "val_accuracy": float(r["val_accuracy"]),
                }, step=int(r["epoch"]))
        log.info("WANDB RUN COMPLETE | run=%s epochs_logged=%d", run_name, len(frame))
    # Validation accuracy scatter: W&B custom chart.
    log.info("WANDB SCATTER STAGE START | rows=%d", len(history_df))
    scatter = wandb.Table(dataframe=history_df[["run", "epoch", "val_accuracy"]])
    with wandb.init(project=f"{project}-scatter", name="validation-accuracy-vs-step", reinit=True):
        wandb.log({"validation_accuracy_scatter": wandb.plot.scatter(
            scatter, "epoch", "val_accuracy", "Validation accuracy vs. step"
        )})
    log.info("WANDB SCATTER STAGE COMPLETE")
    # Log configuration table and native parallel-coordinate preset.
    columns = [c for c in ["run", "activation", "optimizer", "batch_size",
        "learning_rate", "epochs_requested", "epochs", "best_val_accuracy", "test_accuracy"]
        if c in summary_df.columns]
    log.info("WANDB CONFIGURATION STAGE START | rows=%d columns=%s", len(summary_df), columns)
    table = wandb.Table(dataframe=summary_df[columns])
    with wandb.init(project=f"{project}-configurations", name="configuration-results", reinit=True):
        payload = {"configuration_results": table}
        try:
            payload["parallel_coordinates_validation_accuracy"] = wandb.plot_table(
                vega_spec_name="wandb/parallel-coordinates/v0",
                data_table=table,
                fields={"value": "best_val_accuracy",
                        "columns": [c for c in columns if c not in ("run",)]}
            )
        except (AttributeError, ValueError, KeyError) as exc:
            log.warning("Parallel-coordinate preset unavailable in this W&B SDK: %s", exc)
            log.warning("Configuration table is still logged; add a native Parallel Coordinates panel in W&B.")
        wandb.log(payload)
    log.info("WANDB CONFIGURATION STAGE COMPLETE | project=%s-configurations", project)
