from pathlib import Path

# model params
BODY_SIZE: int = 128
REFINER_SIZE: int = 384
BODY_DEPTH: int = 3
REFINER_DEPTH: int = 5
THRESHOLD: int = 5
FILTER_SIZE: int = 5

# training params
EPOCHS: int = 20
BATCH_SIZE: int = 16
LEARNING_RATE: float = 1e-3
DEVICE: str = "cuda"
DEVICE_ID: int = 0
DO_TRAIN_BODY: bool = True
DO_TRAIN_REFINER: bool = False

# dataset params
DATASET_PATH: str = Path(__file__).parent / "dataset"
TRAIN_SIZE: int = 15000
VAL_SIZE: int = 3000
DATASET_CLASSES = None

# save-load params
CHECKPOINT_PATH: str = "bgremover_model"
CONTINUE: bool = True

# logging params
CLEARML_PROJECT: str = "BGRemover"
CLEARML_TASK: str = "train2"
