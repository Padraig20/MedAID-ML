from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
ROOT_DIR = MODULE_DIR.parent.parent
DATA_DIR = ROOT_DIR / "data"
DATA_TEST_JSON = DATA_DIR / "dataset_test.json"
DATA_TRAIN_JSON = DATA_DIR / "dataset_train.json"
RESULTS_DIR = ROOT_DIR / "results"
NO_DATALEAK_NAME = "results_no_dataleak.csv"
TEST_NAME = "results_test.csv"

ID_TO_LABEL = {
    0: "Human",
    1: "AI"
}

LABEL_TO_ID = {
    "Human": 0,
    "AI": 1
}