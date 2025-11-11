import pandas as pd

from constants import PROJECT_ROOT

EN_DATA_PATH = PROJECT_ROOT / "data"


def read_en_humanitarian_data(
    dataset: str = "all_data_en",
    basename: str = "crisis_consolidated_humanitarian_filtered_lang_en",
    filetype: str = ".tsv",
) -> tuple:
    """Reads English humanitarian data.

    Args:
        dataset (str): Name of the English dataset file. Defaults to "all_data_en".
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
        EN_DATA_PATH / dataset / f"{basename}_train{filetype}",
        sep=sep,
        engine="python",
        on_bad_lines="skip",
    )
    humanitarian_dev_df = pd.read_csv(
        EN_DATA_PATH / dataset / f"{basename}_dev{filetype}",
        sep=sep,
        engine="python",
        on_bad_lines="skip",
    )
    humanitarian_test_df = pd.read_csv(
        EN_DATA_PATH / dataset / f"{basename}_test{filetype}",
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
    dataset: str = "all_data_en",
    basename: str = "crisis_consolidated_informativeness_filtered_lang_en",
    filetype: str = ".tsv",
) -> tuple:
    """Reads English informativeness data.

    Args:
        dataset (str): Name of the English dataset file. Defaults to "all_data_en".
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
        EN_DATA_PATH / dataset / f"{basename}_train{filetype}",
        sep=sep,
        engine="python",
        on_bad_lines="skip",
    )
    informativeness_dev_df = pd.read_csv(
        EN_DATA_PATH / dataset / f"{basename}_dev{filetype}",
        sep=sep,
        engine="python",
        on_bad_lines="skip",
    )
    informativeness_test_df = pd.read_csv(
        EN_DATA_PATH / dataset / f"{basename}_test{filetype}",
        sep=sep,
        engine="python",
        on_bad_lines="skip",
    )

    return (
        informativeness_train_df,
        informativeness_dev_df,
        informativeness_test_df,
    )
