import os

import pandas as pd
import torch

import tweet_classification.constants as const
from tweet_classification.utils import (
    get_info_datasets,
    read_en_informativeness_data,
    set_seed,
)

set_seed(const.SEED)

from torch import nn
from torch.utils.data import DataLoader

from tweet_classification.classifier import TweetClassifier
from tweet_classification.train_test_funcs import train

INFO_RESULT_PATH = const.RESULTS_PATH / "info_results"


def main():
    print(f"Using device: {const.DEVICE}")

    info_temp_df, _, _ = read_en_informativeness_data()
    NUM_LABELS = info_temp_df["class_label"].nunique()
    train_dataset, dev_dataset, test_dataset = get_info_datasets()

    train_dataloader = DataLoader(
        train_dataset, batch_size=const.BATCH_SIZE, shuffle=True, num_workers=const.NUM_WORKERS
    )
    dev_dataloader = DataLoader(
        dev_dataset, batch_size=const.BATCH_SIZE, shuffle=False, num_workers=const.NUM_WORKERS
    )

    model = TweetClassifier(num_labels=NUM_LABELS, dropout=const.DROPOUT).to(const.DEVICE)
    print(model)

    inverse_weights = train_dataset.get_inverse_weights().to(const.DEVICE)

    loss_fn = nn.CrossEntropyLoss(weight=inverse_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    print("Training is about to start...")
    model_dev_results = train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=dev_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=const.DEVICE,
        epochs=const.NUM_EPOCHS,
    )
    print("Training finished.")

    RESULTS_PATH = INFO_RESULT_PATH / f"exp_{const.INFO_EXPERIMENT_NR}"
    os.makedirs(RESULTS_PATH, exist_ok=True)
    print(model_dev_results)
    model_dev_results_df = pd.DataFrame(model_dev_results)
    model_dev_results_df.to_csv(RESULTS_PATH / "model_dev_results.csv", index=False)
    print("Results saved.")

    MODEL_PATH = RESULTS_PATH / "models"
    os.makedirs(MODEL_PATH, exist_ok=True)
    MODEL_NAME = f"info_classifier_exp_{const.INFO_EXPERIMENT_NR}.pth"
    torch.save(obj=model.state_dict(), f=MODEL_PATH / MODEL_NAME)
    print("Model saved.")


if __name__ == "__main__":
    main()
