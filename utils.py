from pathlib import Path

import pandas as pd

from constants import PROJECT_ROOT

EN_DATA_PATH = PROJECT_ROOT / "data" / "all_data_en"


def read_en_humanitarian_data(data_path: Path = EN_DATA_PATH) -> tuple:
    """Reads English humanitarian data.

    Args:
        data_path (pathlib.Path): Path to the English dataset file. Defaults to EN_DATA_PATH.

    Returns:
        (
            train_df: pd.DataFrame,
            dev_df: pd.DataFrame,
            test_df: pd.DataFrame
        )
    """
    humanitarian_data = "crisis_consolidated_humanitarian_filtered_lang_en"

    humanitarian_train_df = pd.read_csv(
        data_path / f"{humanitarian_data}_train.tsv", sep="\t", engine="python", on_bad_lines="skip"
    )
    humanitarian_dev_df = pd.read_csv(
        data_path / f"{humanitarian_data}_dev.tsv", sep="\t", engine="python", on_bad_lines="skip"
    )
    humanitarian_test_df = pd.read_csv(
        data_path / f"{humanitarian_data}_test.tsv", sep="\t", engine="python", on_bad_lines="skip"
    )

    return (
        humanitarian_train_df,
        humanitarian_dev_df,
        humanitarian_test_df,
    )


def read_en_informativeness_data(data_path: Path = EN_DATA_PATH) -> tuple:
    """Reads English informativeness data.

    Args:
        data_path (pathlib.Path): Path to the English dataset file. Defaults to EN_DATA_PATH.

    Returns:
        (
            train_df: pd.DataFrame,
            dev_df: pd.DataFrame,
            test_df: pd.DataFrame
        )
    """
    informativeness_data = "crisis_consolidated_informativeness_filtered_lang_en"

    informativeness_train_df = pd.read_csv(
        data_path / f"{informativeness_data}_train.tsv",
        sep="\t",
        engine="python",
        on_bad_lines="skip",
    )
    informativeness_dev_df = pd.read_csv(
        data_path / f"{informativeness_data}_dev.tsv",
        sep="\t",
        engine="python",
        on_bad_lines="skip",
    )
    informativeness_test_df = pd.read_csv(
        data_path / f"{informativeness_data}_test.tsv",
        sep="\t",
        engine="python",
        on_bad_lines="skip",
    )

    return (
        informativeness_train_df,
        informativeness_dev_df,
        informativeness_test_df,
    )
