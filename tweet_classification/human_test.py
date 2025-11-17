import torch

import tweet_classification.constants as const
from tweet_classification.utils import (
    get_human_datasets,
    read_en_humanitarian_data,
    set_seed,
)

set_seed(const.SEED)

from torch.utils.data import DataLoader

from tweet_classification.classifier import TweetClassifier
from tweet_classification.train_test_funcs import FocalLoss, human_inference_eval

HUMAN_RESULTS_PATH = const.RESULTS_PATH / "human_results" / f"exp_{const.HUMAN_EXPERIMENT_NR}"


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
            HUMAN_RESULTS_PATH / "models" / f"human_classifier_exp_{const.HUMAN_EXPERIMENT_NR}.pth",
            map_location=const.DEVICE,
        )
    )

    loss_fn = FocalLoss()

    metrics_df, preds_labels_df = human_inference_eval(
        model=model, test_dataloader=test_dataloader, loss_fn=loss_fn, device=const.DEVICE
    )

    metrics_df.to_csv(HUMAN_RESULTS_PATH / "performance_metrics.csv", index=False)
    preds_labels_df.to_csv(HUMAN_RESULTS_PATH / "predictions.csv", index=False)


if __name__ == "__main__":
    main()
