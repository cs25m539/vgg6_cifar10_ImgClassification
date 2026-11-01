"""Assignment experiment matrix and best-configuration selection."""
import logging
import pandas as pd

logger = logging.getLogger("vgg6.experiments")

def build_experiments(group: str, args):
    """Build the same assignment groups and comparison values as the reference notebook."""
    configs = []
    def add(run, category, activation="relu", optimizer="adam", batch=64, lr=0.001, epochs=None):
        configs.append({
            "run": run, "group": category, "activation": activation, "optimizer": optimizer,
            "batch_size": batch, "learning_rate": lr,
            "epochs": args.epochs if epochs is None else epochs,
        })
    if group in ("baseline", "all"):
        add("baseline_relu_adam", "baseline")
    if group in ("activation", "all"):
        for a in ("relu", "sigmoid", "tanh", "silu", "gelu"):
            add(f"activation_{a}", "activation", activation=a)
    if group in ("optimizer", "all"):
        for o in ("sgd", "nesterov-sgd", "adam", "adagrad", "rmsprop", "nadam"):
            add(f"optimizer_{o}", "optimizer", optimizer=o)
    if group in ("hyperparameters", "all"):
        for batch in (32, 64, 128):
            add(f"batch_size_{batch}", "batch_size", batch=batch)
        for lr in (0.0001, 0.001, 0.01):
            add(f"learning_rate_{lr:g}", "learning_rate", lr=lr)
        for ep in (10, 20, 30):
            add(f"epochs_{ep}", "epochs", epochs=ep)
    logger.info("EXPERIMENT MATRIX BUILT | group=%s run_count=%d runs=%s", group, len(configs), [c["run"] for c in configs])
    return configs

def best_config_from_csv(path):
    """Return exactly one row with the maximum best validation accuracy."""
    logger.info("BEST-CONFIG READ START | summary_csv=%s", path)
    df = pd.read_csv(path)
    if df.empty or "best_val_accuracy" not in df:
        raise ValueError(f"{path} is empty or missing best_val_accuracy")
    df["best_val_accuracy"] = pd.to_numeric(df["best_val_accuracy"], errors="coerce")
    df = df.dropna(subset=["best_val_accuracy"])
    if df.empty:
        raise ValueError("No numeric best_val_accuracy values.")
    row = df.loc[df["best_val_accuracy"].idxmax()]
    logger.info("BEST-CONFIG SELECTED | source_run=%s best_val_accuracy=%.4f", row.get("run", "best"), float(row["best_val_accuracy"]))
    return {
        "run": f'verify_{row.get("run", "best")}',
        "group": "verification",
        "activation": str(row["activation"]).lower(),
        "optimizer": str(row["optimizer"]).lower(),
        "batch_size": int(row["batch_size"]),
        "learning_rate": float(row["learning_rate"]),
        "epochs": int(row.get("epochs_requested", row.get("epochs", 20))),
    }, row
