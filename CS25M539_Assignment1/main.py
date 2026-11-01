"""CLI entry point for baseline, assignment sweeps, and best-config verification."""
import argparse
import logging
from pathlib import Path
import pandas as pd
import torch
from vgg6.engine import train_experiment
from vgg6.experiments import build_experiments, best_config_from_csv
from vgg6.plots import save_plots, log_wandb
from vgg6.logging_utils import configure_logging

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--group", choices=["baseline","activation","optimizer","hyperparameters","all","verify-best"], default="baseline")
    p.add_argument("--epochs", type=int, default=20, help="Epochs for baseline/activation/optimizer groups.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--activation", choices=["relu","sigmoid","tanh","silu","gelu"], default="relu")
    p.add_argument("--optimizer", choices=["sgd","nesterov-sgd","adam","adagrad","rmsprop","nadam"], default="adam")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--output-dir", default="./vgg6_results")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--summary-csv", default=None, help="Required for --group verify-best.")
    p.add_argument("--wandb", action="store_true", help="Log curves, scatter, and configuration table to W&B.")
    p.add_argument("--wandb-project", default="vgg6-cifar10")
    return p.parse_args()

def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise SystemExit("epochs/batch-size must be positive and learning-rate must be > 0")
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(out / "logs" / "project.log")
    logger.info("=" * 88)
    logger.info("PROJECT START | group=%s seed=%s output_dir=%s", args.group, args.seed, out.resolve())
    logger.info("Arguments: %s", vars(args))
    if args.group == "verify-best":
        if not args.summary_csv:
            raise SystemExit("--summary-csv is required for --group verify-best")
        best, original = best_config_from_csv(args.summary_csv)
        logger.info("BEST-CONFIG SELECTION | criterion=maximum best_val_accuracy")
        logger.info("Selected source row:\n%s", original.to_string())
        logger.info("Verification configuration: %s", best)
        configs = [best]
    elif args.group == "baseline":
        configs = [{"run":"baseline_custom","group":"baseline","activation":args.activation,
                    "optimizer":args.optimizer,"batch_size":args.batch_size,
                    "learning_rate":args.learning_rate,"epochs":args.epochs}]
    else:
        configs = build_experiments(args.group, args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("SETUP COMPLETE | device=%s seed=%s experiment_count=%d", device, args.seed, len(configs))
    summaries, histories = [], []
    for index, cfg in enumerate(configs, start=1):
        logger.info("EXPERIMENT QUEUED | index=%d/%d run=%s config=%s", index, len(configs), cfg["run"], cfg)
        result, history = train_experiment(cfg, args, device)
        summaries.append(result); histories.extend(history)
        # Save incrementally so results survive interruptions.
        pd.DataFrame(summaries).to_csv(out/"experiment_summary.csv", index=False)
        pd.DataFrame(histories).to_csv(out/"epoch_history.csv", index=False)
        logger.info("CHECKPOINTED RESULTS | completed_runs=%d summary_csv=%s history_csv=%s", len(summaries), out/"experiment_summary.csv", out/"epoch_history.csv")
    summary_df, history_df = pd.DataFrame(summaries), pd.DataFrame(histories)
    # CSVs from verification are intentionally in a separate output directory.
    summary_df.to_csv(out/"experiment_summary.csv", index=False)
    history_df.to_csv(out/"epoch_history.csv", index=False)
    logger.info("PLOT STAGE START | runs=%d", history_df["run"].nunique() if not history_df.empty else 0)
    save_plots(history_df, out, logger_override=logger)
    logger.info("PLOT STAGE COMPLETE | directory=%s", out / "plots")
    logger.info("FINAL SUMMARY | selected by best validation accuracy:\n%s", summary_df.to_string(index=False))
    logger.info("OUTPUTS | directory=%s", out.resolve())
    if args.wandb:
        logger.info("WANDB STAGE START | project_prefix=%s", args.wandb_project)
        log_wandb(history_df, summary_df, args.wandb_project, logger_override=logger)
        logger.info("WANDB STAGE COMPLETE | charts and configuration table submitted")
    logger.info("PROJECT COMPLETE | group=%s runs=%d", args.group, len(summary_df))

if __name__ == "__main__":
    main()
