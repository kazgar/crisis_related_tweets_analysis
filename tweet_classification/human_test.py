import constants as const
import pandas as pd
import torch
from utils import get_human_datasets, read_en_humanitarian_data, set_seed

set_seed(const.SEED)

from classifier import TweetClassifier
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from train_test_funcs import FocalLoss, test_step

HUMAN_RESULTS_PATH = const.RESULTS_PATH / "human_results"


def main():
    print(f"Using device: {const.DEVICE}")

    human_temp_df, _, _ = read_en_humanitarian_data()
    NUM_LABELS = human_temp_df["class_label"].nunique()
    _, _, test_dataset = get_human_datasets()

    test_dataloader = DataLoader(
        test_dataset, batch_size=const.BATCH_SIZE, shuffle=False, num_workers=const.NUM_WORKERS
    )

    model = TweetClassifier(num_labels=NUM_LABELS, dropout=const.DROPOUT).to(const.DEVICE)
    model.load_state_dict(
        torch.load(
            HUMAN_RESULTS_PATH
            / f"exp_{const.HUMAN_EXPERIMENT_NR}"
            / "models"
            / f"human_classifier_exp_{const.HUMAN_EXPERIMENT_NR}.pth",
            map_location=const.DEVICE,
        )
    )

    loss_fn = FocalLoss()

    test_loss, test_acc = test_step(
        model=model, test_dataloader=test_dataloader, loss_fn=loss_fn, device=const.DEVICE
    )

    print(test_loss, test_acc)


if __name__ == "__main__":
    main()
