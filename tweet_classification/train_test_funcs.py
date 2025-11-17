import constants as const
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def train_step(
    model: nn.Module,
    train_dataloader: DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    l1_regularization: float | None = None,
) -> tuple:
    model.train()
    train_loss, train_acc = 0, 0

    for batch in train_dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        y_pred = model(input_ids, attention_mask)

        loss = loss_fn(y_pred, labels)

        if l1_regularization:
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = loss + const.L1_LAMBDA * l1_norm

        train_loss += loss.item()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == labels).sum().item() / len(y_pred)

    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)
    return train_loss, train_acc


def test_step(
    model: nn.Module,
    test_dataloader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device | str,
) -> tuple:
    model.eval()

    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for batch in test_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            test_pred_logits = model(input_ids, attention_mask)

            loss = loss_fn(test_pred_logits, labels)
            test_loss += loss.item()

            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += (test_pred_labels == labels).sum().item() / len(test_pred_labels)

        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)
        return test_loss, test_acc


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device | str,
    epochs: int,
) -> dict:
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, device)
        test_loss, test_acc = test_step(model, test_dataloader, loss_fn, device)

        # if (epoch + 1) % 100 == 0:
        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.3f} | "
            f"train_acc: {train_acc:.3f} | "
            f"test_loss: {test_loss:.3f} | "
            f"test_acc: {test_acc:.3f}"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results


def human_inference_eval(
    model: nn.Module, test_dataloader: DataLoader, loss_fn: nn.Module, device: torch.device | str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate a multiclass classification model and return metrics and predictions.

    Returns:
        dict: {
            "loss": float,
            "accuracy": float,
            "precision": float,
            "recall": float,
            "f1": float,
            "predictions": list,
            "true_labels": list
        }
    """
    model.eval()

    all_preds = []
    all_labels = []
    total_loss = 0

    with torch.inference_mode():
        for batch in tqdm(test_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(test_dataloader)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    return (
        pd.DataFrame.from_dict(
            {
                "loss": [avg_loss],
                "accuracy": [accuracy],
                "precision": [precision],
                "recall": [recall],
                "f1": [f1],
            }
        ),
        pd.DataFrame.from_dict(
            {
                "predictions": all_preds,
                "true_labels": all_labels,
            }
        ),
    )


def info_inference_eval(
    model: nn.Module,
    test_dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device | str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate a binary classification model and return metrics and predictions.

    Args:
        model: nn.Module, trained model
        test_dataloader: DataLoader
        loss_fn: loss function
        device: torch.device or str
        return_probs: if True, returns predicted probabilities for the positive class

    Returns:
        tuple with two pd.DataFrames
    """
    model.eval()

    all_preds = []
    all_labels = []
    total_loss = 0

    with torch.inference_mode():
        for batch in tqdm(test_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits.squeeze(), labels.float())
            total_loss += loss.item()

            probs = torch.sigmoid(logits).squeeze()
            preds = (probs >= 0.5).long()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    all_preds_indices = [np.argmax(pred) for pred in all_preds]

    avg_loss = total_loss / len(test_dataloader)
    accuracy = accuracy_score(all_labels, all_preds_indices)
    precision = precision_score(all_labels, all_preds_indices, zero_division=0)
    recall = recall_score(all_labels, all_preds_indices, zero_division=0)
    f1 = f1_score(all_labels, all_preds_indices, zero_division=0)

    return (
        pd.DataFrame.from_dict(
            {
                "loss": [avg_loss],
                "accuracy": [accuracy],
                "precision": [precision],
                "recall": [recall],
                "f1": [f1],
            }
        ),
        pd.DataFrame.from_dict(
            {
                "predictions": all_preds_indices,
                "true_labels": all_labels,
            }
        ),
    )


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, labels):
        ce = F.cross_entropy(logits, labels, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()
