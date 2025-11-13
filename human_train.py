import os

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from classifier import TweetClassifier
from constants import EXPERIMENT_NR, PROJECT_ROOT
from train_test_funcs import train
from utils import get_human_datasets, read_en_humanitarian_data

HUMAN_RESULT_PATH = PROJECT_ROOT / "results" / "human_results"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
BATCH_SIZE = 64
NUM_WORKERS = 1
RANDOM_SEED = 42
DROPOUT = 0.1
NUM_EPOCHS = 50


def main():
    print(f"Using device: {DEVICE}")

    human_temp_df, _, _ = read_en_humanitarian_data()
    NUM_LABELS = human_temp_df["class_label"].nunique()
    train_dataset, dev_dataset, test_dataset = get_human_datasets()

    torch.manual_seed(RANDOM_SEED)
    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    dev_dataloader = DataLoader(
        dev_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    torch.manual_seed(RANDOM_SEED)
    model = TweetClassifier(num_labels=NUM_LABELS, dropout=DROPOUT).to(DEVICE)
    print(model)

    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)

    inverse_weights = train_dataset.get_inverse_weights().to(DEVICE)

    loss_fn = nn.CrossEntropyLoss(weight=inverse_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    print("Training is about to start...")
    model_dev_results = train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=dev_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=DEVICE,
        epochs=NUM_EPOCHS,
    )
    print("Training finished.")

    RESULTS_PATH = HUMAN_RESULT_PATH / f"exp_{EXPERIMENT_NR}" / "results"
    os.makedirs(RESULTS_PATH, exist_ok=True)
    print(model_dev_results)
    model_dev_results_df = pd.DataFrame(model_dev_results)
    model_dev_results_df.to_csv(RESULTS_PATH / "model_dev_results.csv", index=False)
    print("Results saved.")

    MODEL_PATH = HUMAN_RESULT_PATH / f"exp_{EXPERIMENT_NR}" / "models"
    os.makedirs(MODEL_PATH, exist_ok=True)
    MODEL_NAME = "human_classifier_01.pth"
    torch.save(obj=model.state_dict(), f=MODEL_PATH / MODEL_NAME)
    print("Model saved.")


if __name__ == "__main__":
    main()
