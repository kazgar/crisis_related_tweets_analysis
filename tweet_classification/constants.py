from pathlib import Path

import torch

EXPERIMENT_NR = 1

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_PATH = PROJECT_ROOT / "data"
EN_DATA_PATH = DATA_PATH / "all_data_en"
CLEAN_DATA_PATH = DATA_PATH / "clean_en_data"
RESULTS_PATH = PROJECT_ROOT / "results"
GRAPHS_PATH = PROJECT_ROOT / "graphs"

L1_LAMBDA = 1e-4
NUM_WORKERS = 1
DROPOUT = 0.2
SEED = 42
NUM_EPOCHS = 50
BATCH_SIZE = 64
DEVICE = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps") if torch.mps.is_available() else torch.device("cpu")
)
