import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from constants import CLEAN_DATA_PATH, EN_DATA_PATH, PROJECT_ROOT
from dataset import TweetDataset
from torch.nn.functional import cross_entropy


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_en_humanitarian_data(
    data_path: Path = EN_DATA_PATH,
    basename: str = "crisis_consolidated_humanitarian_filtered_lang_en",
    filetype: str = ".tsv",
) -> tuple:
    """Reads English humanitarian data.

    Args:
        data_path (Path): Path to the data directory.
        basename (str): Base name of the data files. Defaults to "crisis_consolidated_humanitarian_filtered_lang_en".
        filetype (str): Type of the data file. Defaults to ".tsv".

    Returns:
        (
            train_df: pd.DataFrame,
            dev_df: pd.DataFrame,
            test_df: pd.DataFrame
        )
    """
    sep = "\t" if filetype == ".tsv" else ","

    humanitarian_train_df = pd.read_csv(
        data_path / f"{basename}_train{filetype}",
        sep=sep,
        engine="python",
        on_bad_lines="skip",
    )
    humanitarian_dev_df = pd.read_csv(
        data_path / f"{basename}_dev{filetype}",
        sep=sep,
        engine="python",
        on_bad_lines="skip",
    )
    humanitarian_test_df = pd.read_csv(
        data_path / f"{basename}_test{filetype}",
        sep=sep,
        engine="python",
        on_bad_lines="skip",
    )

    return (
        humanitarian_train_df,
        humanitarian_dev_df,
        humanitarian_test_df,
    )


def read_en_informativeness_data(
    data_path: Path = EN_DATA_PATH,
    basename: str = "crisis_consolidated_informativeness_filtered_lang_en",
    filetype: str = ".tsv",
) -> tuple:
    """Reads English informativeness data.

    Args:
        data_path (Path): Path to the data directory.
        basename (str): Base name of the data files. Defaults to "crisis_consolidated_informativeness_filtered_lang_en".
        filetype (str): Type of the data file. Defaults to ".tsv".

    Returns:
        (
            train_df: pd.DataFrame,
            dev_df: pd.DataFrame,
            test_df: pd.DataFrame
        )
    """
    sep = "\t" if filetype == ".tsv" else ","

    informativeness_train_df = pd.read_csv(
        data_path / f"{basename}_train{filetype}",
        sep=sep,
        engine="python",
        on_bad_lines="skip",
    )
    informativeness_dev_df = pd.read_csv(
        data_path / f"{basename}_dev{filetype}",
        sep=sep,
        engine="python",
        on_bad_lines="skip",
    )
    informativeness_test_df = pd.read_csv(
        data_path / f"{basename}_test{filetype}",
        sep=sep,
        engine="python",
        on_bad_lines="skip",
    )

    return (
        informativeness_train_df,
        informativeness_dev_df,
        informativeness_test_df,
    )


def get_human_datasets():
    """Creates and returns torch.utils.data.Dataset's for the humanitarian data.
    Returns:
        (
            train_df: torch.utils.data.Dataset,
            dev_df: torch.utils.data.Dataset,
            test_df: torch.utils.data.Dataset
        )
    """
    human_train_df, human_dev_df, human_test_df = read_en_humanitarian_data(
        data_path=CLEAN_DATA_PATH, basename="human", filetype=".csv"
    )

    return (TweetDataset(human_train_df), TweetDataset(human_dev_df), TweetDataset(human_test_df))


def get_info_datasets():
    """Creates and returns torch.utils.data.Dataset's for the informativeness data.
    Returns:
        (
            train_df: torch.utils.data.Dataset,
            dev_df: torch.utils.data.Dataset,
            test_df: torch.utils.data.Dataset
        )
    """
    info_train_df, info_dev_df, info_test_df = read_en_informativeness_data(
        data_path=CLEAN_DATA_PATH, basename="info", filetype=".csv"
    )

    return (TweetDataset(info_train_df), TweetDataset(info_dev_df), TweetDataset(info_test_df))


def focal_loss(logits, labels, alpha=None, gamma=2):
    ce = cross_entropy(logits, labels, weight=alpha, reduction="none")
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()
