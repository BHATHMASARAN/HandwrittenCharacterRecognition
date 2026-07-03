import torch

from src.training.metrics import EarlyStopping, MetricTracker


def test_metric_tracker_perfect_predictions():
    tracker = MetricTracker()
    preds = torch.tensor([0, 1, 2, 3])
    labels = torch.tensor([0, 1, 2, 3])
    tracker.update(preds, labels, loss=0.1)
    metrics = tracker.compute()
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0


def test_metric_tracker_partial_predictions():
    tracker = MetricTracker()
    preds = torch.tensor([0, 1, 2, 2])
    labels = torch.tensor([0, 1, 2, 3])
    tracker.update(preds, labels, loss=0.5)
    metrics = tracker.compute()
    assert metrics["accuracy"] == 0.75


def test_early_stopping_triggers_after_patience():
    stopper = EarlyStopping(patience=2, mode="max")
    scores = [0.5, 0.4, 0.3]
    for score in scores:
        stopper.step(score)
    assert stopper.should_stop is True


def test_early_stopping_resets_on_improvement():
    stopper = EarlyStopping(patience=2, mode="max")
    stopper.step(0.5)
    stopper.step(0.4)
    improved = stopper.step(0.6)
    assert improved is True
    assert stopper.counter == 0
    assert stopper.should_stop is False
