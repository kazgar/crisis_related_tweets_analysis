from pathlib import Path

EXPERIMENT_NR = 1

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_PATH = PROJECT_ROOT / "data"
EN_DATA_PATH = DATA_PATH / "all_data_en"
CLEAN_DATA_PATH = DATA_PATH / "clean_en_data"
RESULTS_PATH = PROJECT_ROOT / "results"
GRAPHS_PATH = PROJECT_ROOT / "graphs"
