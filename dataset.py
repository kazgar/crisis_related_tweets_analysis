from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset


class TweetDataset(Dataset):
    def __init__(self, csv_file: Path | pd.DataFrame):
        if isinstance(csv_file, Path):
            self.tweet_data = pd.read_csv(csv_file)
        elif isinstance(csv_file, pd.DataFrame):
            self.tweet_data = csv_file

    def __len__(self):
        return len(self.tweet_data)

    def __getitem__(self, idx):
        text, label = self.tweet_data.iloc[idx, :]

        sample = {"text": text, "label": label}

        return sample
