"""Generate W&B reports from existing CSVs; does not train the model."""
import argparse
from pathlib import Path

from vgg6.csv_reporting import run_csv_reporting
from vgg6.logging_utils import configure_logging


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create local plots and W&B reports from existing CSV results."
    )
    parser.add_argument("--history-csv", required=True,
                        help="Path to epoch_history.csv from Step 5.")
    parser.add_argument("--summary-csv", required=True,
                        help="Path to experiment_summary.csv from Step 5.")
    parser.add_argument("--output-dir", default="./vgg6_plot_reports",
                        help="Directory for generated PNGs and reporting log.")
    parser.add_argument("--wandb-project", default="vgg6-cifar10",
                        help="W&B project prefix; suffixes are added automatically.")
    parser.add_argument("--wandb-entity", default=None,
                        help="Optional W&B username/team.")
    parser.add_argument("--local-only", action="store_true",
                        help="Create local PNGs without uploading to W&B.")
    parser.add_argument("--wandb-only", action="store_true",
                        help="Upload to W&B without creating local PNGs.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.local_only and args.wandb_only:
        raise SystemExit("Choose at most one of --local-only and --wandb-only.")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(output / "logs" / "csv_reporting.log")
    logger.info("REPORT CLI ARGUMENTS | %s", vars(args))

    run_csv_reporting(
        history_csv=args.history_csv,
        summary_csv=args.summary_csv,
        output_dir=output,
        project=args.wandb_project,
        entity=args.wandb_entity,
        local_plots=not args.wandb_only,
        wandb_upload=not args.local_only,
    )


if __name__ == "__main__":
    main()
