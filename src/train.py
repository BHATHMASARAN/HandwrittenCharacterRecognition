import argparse

from src.data.dataset import get_dataloaders
from src.models.cnn import CharCNN
from src.training.trainer import Trainer
from src.utils.common import set_seed
from src.utils.config_loader import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train handwritten character recognition model")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--dataset", type=str, default=None, choices=["mnist", "emnist"])
    return parser.parse_args()


def main():
    args = parse_args()
    config = get_config(args.config)

    if args.epochs is not None:
        config._config["training"]["epochs"] = args.epochs
    if args.dataset is not None:
        config._config["data"]["dataset"] = args.dataset

    set_seed()

    logger.info(f"Loading dataset: {config.get('data.dataset')}")
    train_loader, val_loader, test_loader, label_info = get_dataloaders()
    logger.info(f"Classes: {label_info['num_classes']} | Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = CharCNN(num_classes=label_info["num_classes"], config=config)

    trainer = Trainer(model, train_loader, val_loader, label_info, config=config)
    history = trainer.fit()

    logger.info("Training complete. Running final evaluation on test set...")
    from src.training.evaluate import evaluate_test_set
    test_metrics = evaluate_test_set(trainer.model, test_loader, trainer.device)
    logger.info(f"Test metrics: {test_metrics}")


if __name__ == "__main__":
    main()
