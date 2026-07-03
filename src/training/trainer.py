import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR

from src.training.metrics import EarlyStopping, MetricTracker
from src.utils.common import count_parameters, get_device, set_seed
from src.utils.config_loader import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Trainer:
    def __init__(self, model: nn.Module, train_loader, val_loader, label_info: dict, config=None):
        self.config = config or get_config()
        set_seed()

        self.device = get_device()
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.label_info = label_info

        self.epochs = self.config.get("training.epochs", 30)
        self.lr = self.config.get("training.learning_rate", 0.001)
        self.weight_decay = self.config.get("training.weight_decay", 0.0001)
        self.grad_clip_norm = self.config.get("training.grad_clip_norm", 5.0)
        self.label_smoothing = self.config.get("training.label_smoothing", 0.05)
        self.log_interval = self.config.get("training.log_interval", 50)

        self.checkpoint_dir = Path(self.config.resolve_path(self.config.get("training.checkpoint_dir", "checkpoints")))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_best_only = self.config.get("training.save_best_only", True)

        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.early_stopping = EarlyStopping(
            patience=self.config.get("training.early_stopping_patience", 6), mode="max"
        )

        self.mlflow_enabled = self.config.get("mlflow.enabled", True)
        self._mlflow = None
        if self.mlflow_enabled:
            self._setup_mlflow()

        logger.info(f"Model initialized on {self.device} with {count_parameters(self.model):,} trainable params")

    def _build_optimizer(self):
        name = self.config.get("training.optimizer", "adamw")
        if name == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if name == "sgd":
            return torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def _build_scheduler(self):
        name = self.config.get("training.scheduler", "cosine")
        if name == "step":
            return StepLR(self.optimizer, step_size=10, gamma=0.5)
        if name == "plateau":
            return ReduceLROnPlateau(self.optimizer, mode="max", factor=0.5, patience=3)
        return CosineAnnealingLR(self.optimizer, T_max=self.epochs)

    def _setup_mlflow(self) -> None:
        try:
            import mlflow
            from pathlib import Path as _Path
            tracking_dir = _Path(self.config.resolve_path(self.config.get("mlflow.tracking_uri", "mlruns")))
            tracking_dir.mkdir(parents=True, exist_ok=True)
            mlflow.set_tracking_uri(tracking_dir.as_uri())
            mlflow.set_experiment(self.config.get("mlflow.experiment_name", "handwritten_char_recognition"))
            self._mlflow = mlflow
        except ImportError:
            logger.warning("mlflow not installed; proceeding without experiment tracking")
            self.mlflow_enabled = False
        except Exception as e:
            logger.warning(f"mlflow setup failed ({e}); proceeding without experiment tracking")
            self.mlflow_enabled = False

    def _train_one_epoch(self, epoch: int) -> dict:
        self.model.train()
        tracker = MetricTracker()

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.optimizer.step()

            preds = outputs.argmax(dim=1)
            tracker.update(preds, labels, loss.item())

            if (batch_idx + 1) % self.log_interval == 0:
                logger.info(
                    f"Epoch {epoch} | Batch {batch_idx + 1}/{len(self.train_loader)} | Loss {loss.item():.4f}"
                )

        return tracker.compute()

    @torch.no_grad()
    def _validate(self) -> dict:
        self.model.eval()
        tracker = MetricTracker()

        for images, labels in self.val_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            preds = outputs.argmax(dim=1)
            tracker.update(preds, labels, loss.item())

        return tracker.compute()

    def _save_checkpoint(self, epoch: int, val_metrics: dict, is_best: bool) -> None:
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_metrics": val_metrics,
            "label_info": self.label_info,
            "config": self.config.all,
        }

        if is_best:
            path = self.checkpoint_dir / "best_model.pt"
            torch.save(state, path)
            logger.info(f"Saved best checkpoint to {path} (val_accuracy={val_metrics['accuracy']:.4f})")

        if not self.save_best_only:
            path = self.checkpoint_dir / f"epoch_{epoch:03d}.pt"
            torch.save(state, path)

    def fit(self) -> dict:
        history = {"train": [], "val": []}
        try:
            run_context = self._mlflow.start_run() if self.mlflow_enabled else _NullContext()
        except Exception as e:
            logger.warning(f"mlflow.start_run failed ({e}); continuing without experiment tracking")
            self.mlflow_enabled = False
            run_context = _NullContext()

        with run_context:
            if self.mlflow_enabled:
                self._mlflow.log_params({
                    "epochs": self.epochs, "lr": self.lr, "weight_decay": self.weight_decay,
                    "optimizer": self.config.get("training.optimizer"),
                    "architecture": self.config.get("model.architecture"),
                })

            for epoch in range(1, self.epochs + 1):
                start = time.time()
                train_metrics = self._train_one_epoch(epoch)
                val_metrics = self._validate()
                elapsed = time.time() - start

                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["accuracy"])
                else:
                    self.scheduler.step()

                history["train"].append(train_metrics)
                history["val"].append(val_metrics)

                logger.info(
                    f"Epoch {epoch}/{self.epochs} ({elapsed:.1f}s) | "
                    f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} | "
                    f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['f1']:.4f}"
                )

                if self.mlflow_enabled:
                    self._mlflow.log_metrics({
                        f"train_{k}": v for k, v in train_metrics.items()
                    }, step=epoch)
                    self._mlflow.log_metrics({
                        f"val_{k}": v for k, v in val_metrics.items()
                    }, step=epoch)

                is_best = self.early_stopping.step(val_metrics["accuracy"])
                self._save_checkpoint(epoch, val_metrics, is_best)

                if self.early_stopping.should_stop:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break

        return history


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False
