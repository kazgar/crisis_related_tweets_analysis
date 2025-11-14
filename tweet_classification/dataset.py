from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class TweetDataset(Dataset):
    def __init__(self, csv_file: Path | pd.DataFrame, max_len: int = 128):
        if isinstance(csv_file, Path):
            self.tweet_data = pd.read_csv(csv_file)
        elif isinstance(csv_file, pd.DataFrame):
            self.tweet_data = csv_file

        self.texts = self.tweet_data["text"].tolist()
        self.labels = self.tweet_data["class_label"].tolist()
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=True)
        self.max_len = max_len

    def get_inverse_weights(self):
        label_counts = self.tweet_data["class_label"].value_counts().sort_index()

        inverse_freqs = 1.0 / label_counts

        normalized_weights = inverse_freqs / inverse_freqs.sum()

        weights_tensor = torch.tensor(normalized_weights.values, dtype=torch.float)

        return weights_tensor

    def __len__(self):
        return len(self.tweet_data)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long),
        }
