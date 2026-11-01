"""Training, validation, checkpoint selection, and final test evaluation.

The public API separates the phases clearly:
- train_one_epoch(): enables gradients and updates model weights.
- evaluate(): measures loss/accuracy without gradients or weight updates.
"""
import copy
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn

from .data import build_loaders
from .model import VGG6, make_optimizer
from .reproducibility import set_seed

logger = logging.getLogger("vgg6.engine")


def _run_pass(model, loader, criterion, device, optimizer=None):
    """Internal loop shared by training and evaluation.

    Call train_one_epoch() or evaluate() from the training workflow so the
    intended phase is explicit. An optimizer indicates training mode.
    """
    training = optimizer is not None
    phase = "TRAIN" if training else "EVALUATE"
    logger.debug("%s PASS START | batches=%d", phase, len(loader))
    model.train(training)
    loss_sum, correct, count = 0.0, 0, 0

    # Gradients are enabled only when training; evaluation never changes weights.
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if training:
                # Clear gradients accumulated for the previous mini-batch.
                optimizer.zero_grad(set_to_none=True)

            # Forward pass: predict classes and calculate cross-entropy loss.
            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                # Backward pass calculates gradients; optimizer updates parameters.
                loss.backward()
                optimizer.step()

            # Weight loss by the number of examples, including a smaller last batch.
            batch_count = labels.size(0)
            loss_sum += loss.item() * batch_count
            correct += (logits.argmax(dim=1) == labels).sum().item()
            count += batch_count

    metrics = {"loss": loss_sum / count, "accuracy": 100.0 * correct / count}
    logger.debug("%s PASS COMPLETE | examples=%d loss=%.4f accuracy=%.2f%%", phase, count, metrics["loss"], metrics["accuracy"])
    return metrics


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one full epoch and return average loss and top-1 accuracy."""
    if optimizer is None:
        raise ValueError("train_one_epoch() requires an optimizer.")
    return _run_pass(model, loader, criterion, device, optimizer=optimizer)


def evaluate(model, loader, criterion, device):
    """Evaluate loss and top-1 accuracy without gradients or parameter updates."""
    return _run_pass(model, loader, criterion, device, optimizer=None)


def train_experiment(config: dict, args, device):
    """Train one configuration, select by validation accuracy, then test once.

    Test data is not used to choose checkpoints or tune hyperparameters. The
    checkpoint with the highest validation accuracy is restored before testing.
    """
    run_name = config["run"]
    run_started = time.time()
    logger.info("RUN START | run=%s group=%s activation=%s optimizer=%s batch_size=%s learning_rate=%s epochs=%s seed=%s device=%s", run_name, config.get("group"), config["activation"], config["optimizer"], config["batch_size"], config["learning_rate"], config["epochs"], args.seed, device)
    logger.info("STAGE START | run=%s stage=seed_and_reproducibility", run_name)
    set_seed(args.seed)
    logger.info("STAGE COMPLETE | run=%s stage=seed_and_reproducibility seed=%s", run_name, args.seed)
    logger.info("STAGE START | run=%s stage=data_loading", run_name)
    data_started = time.time()
    train_loader, val_loader, test_loader = build_loaders(
        args.data_dir, config["batch_size"], args.seed, args.num_workers
    )
    logger.info("STAGE COMPLETE | run=%s stage=data_loading train_batches=%d val_batches=%d test_batches=%d elapsed_sec=%.2f", run_name, len(train_loader), len(val_loader), len(test_loader), time.time() - data_started)

    logger.info("STAGE START | run=%s stage=model_and_optimizer_setup", run_name)
    model = VGG6(config["activation"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(
        config["optimizer"], model, config["learning_rate"]
    )
    logger.info("STAGE COMPLETE | run=%s stage=model_and_optimizer_setup parameters=%d", run_name, sum(p.numel() for p in model.parameters()))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(
        char if char.isalnum() or char in "-_" else "_"
        for char in config["run"]
    )
    checkpoint_path = output_dir / "checkpoints" / f"{safe_name}.pth"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    history = []
    best_val_accuracy = -1.0
    best_epoch = 0
    best_state = None
    start_time = time.time()
    logger.info("STAGE START | run=%s stage=train_validate requested_epochs=%d checkpoint=%s", run_name, config["epochs"], checkpoint_path)

    for epoch in range(1, config["epochs"] + 1):
        epoch_started = time.time()
        logger.info("EPOCH START | run=%s epoch=%d/%d", run_name, epoch, config["epochs"])
        # Phase 1: train on the training subset; this phase updates weights.
        logger.info("PHASE START | run=%s epoch=%d phase=training", run_name, epoch)
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        logger.info("PHASE COMPLETE | run=%s epoch=%d phase=training loss=%.4f accuracy=%.2f%%", run_name, epoch, train_metrics["loss"], train_metrics["accuracy"])
        # Phase 2: validate on held-out validation data; weights stay unchanged.
        logger.info("PHASE START | run=%s epoch=%d phase=validation", run_name, epoch)
        val_metrics = evaluate(model, val_loader, criterion, device)
        logger.info("PHASE COMPLETE | run=%s epoch=%d phase=validation loss=%.4f accuracy=%.2f%%", run_name, epoch, val_metrics["loss"], val_metrics["accuracy"])

        history.append({
            **config,
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        })

        logger.info("EPOCH METRICS | run=%s epoch=%d/%d train_loss=%.4f train_accuracy=%.2f%% val_loss=%.4f val_accuracy=%.2f%% best_val_accuracy_before=%.2f%% elapsed_sec=%.2f", run_name, epoch, config["epochs"], train_metrics["loss"], train_metrics["accuracy"], val_metrics["loss"], val_metrics["accuracy"], best_val_accuracy, time.time() - epoch_started)

        # Save a checkpoint whenever validation accuracy improves.
        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            logger.info("CHECKPOINT UPDATE | run=%s epoch=%d best_val_accuracy=%.2f%% path=%s", run_name, best_epoch, best_val_accuracy, checkpoint_path)
            torch.save(
                {
                    "model_state_dict": best_state,
                    "config": config,
                    "best_epoch": best_epoch,
                    "best_val_accuracy": best_val_accuracy,
                    "seed": args.seed,
                },
                checkpoint_path,
            )

    logger.info("STAGE COMPLETE | run=%s stage=train_validate best_epoch=%d best_val_accuracy=%.2f%%", run_name, best_epoch, best_val_accuracy)
    # Phase 3: restore the best validation checkpoint and evaluate test data once.
    logger.info("STAGE START | run=%s stage=final_test_evaluation checkpoint_epoch=%d", run_name, best_epoch)
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, criterion, device)
    logger.info("STAGE COMPLETE | run=%s stage=final_test_evaluation test_loss=%.4f test_accuracy=%.2f%%", run_name, test_metrics["loss"], test_metrics["accuracy"])

    result = {
        **config,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "minutes": (time.time() - start_time) / 60.0,
        "checkpoint": str(checkpoint_path),
    }
    logger.info("RUN COMPLETE | run=%s best_epoch=%d best_val_accuracy=%.2f%% test_accuracy=%.2f%% elapsed_min=%.2f checkpoint=%s", run_name, best_epoch, best_val_accuracy, test_metrics["accuracy"], (time.time() - run_started) / 60.0, checkpoint_path)
    return result, history
