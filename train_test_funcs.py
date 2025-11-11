import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def train_step(
    model: nn.Module,
    train_dataloader: DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
) -> tuple:
    model.train()
    train_loss, train_acc = 0, 0

    for batch in train_dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        y_pred = model(input_ids, attention_mask)

        loss = loss_fn(y_pred, labels)
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

        results["train_loss"].append(
            train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss
        )
        (
            results["train_acc"].append(train_acc.item())
            if isinstance(train_acc, torch.Tensor)
            else train_acc
        )
        results["test_loss"].append(
            test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss
        )
        (
            results["test_acc"].append(test_acc.item())
            if isinstance(test_acc, torch.Tensor)
            else test_acc
        )

    return results
