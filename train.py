import logging
import os

import torch
from clearml import Task

import config
from models import BGRemover
from train_utils import Trainer

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    device = torch.device("cpu")
    if config.DEVICE == "cuda":
        if torch.cuda.is_available() and config.DEVICE_ID < torch.cuda.device_count():
            device = torch.device(f"cuda:{config.DEVICE_ID}")
        else:
            logging.warning(
                f"GPU {config.DEVICE_ID} is not available, using CPU instead"
            )
    elif config.DEVICE == "mps":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            logging.warning("MPS is not available, using CPU instead")
    elif config.DEVICE != "cpu":
        logging.warning(f"Unknown device {config.DEVICE}, using CPU instead")

    logging.info(f"Using device {device}")

    model = BGRemover(
        config.BODY_SIZE,
        config.REFINER_SIZE,
        config.BODY_DEPTH,
        config.THRESHOLD,
        config.FILTER_SIZE,
    )

    os.makedirs(config.CHECKPOINT_PATH, exist_ok=True)
    if not config.CONTINUE or not len(os.listdir(config.CHECKPOINT_PATH)):
        initial_epoch = 1
    else:
        load_file = (
            config.CHECKPOINT_PATH
            + "/"
            + sorted(os.listdir(config.CHECKPOINT_PATH))[-1]
        )
        state_dict = torch.load(load_file, map_location=device)
        model.load_state_dict(state_dict)
        logging.info(f"Model loaded from {load_file}")
        initial_epoch = int(load_file.split("_")[-2]) + 1

    task: Task = Task.init(
        project_name=config.CLEARML_PROJECT, task_name=config.CLEARML_TASK
    )
    logger = task.get_logger()

    trainer = Trainer(
        model=model,
        device=device,
        epochs=config.EPOCHS,
        initial_epoch=initial_epoch,
        dataset_path=config.DATASET_PATH,
        batch_size=config.BATCH_SIZE,
        train_size=config.TRAIN_SIZE,
        val_size=config.VAL_SIZE,
        learning_rate=config.LEARNING_RATE,
        checkpoint_path=config.CHECKPOINT_PATH,
        logger=logger,
        do_train_body=config.DO_TRAIN_BODY,
        do_train_refiner=config.DO_TRAIN_REFINER,
    )

    trainer.train()
    task.close()
