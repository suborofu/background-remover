import logging
import os
from pathlib import Path

import torch
import tqdm
from clearml import Logger
from torch import optim
from torch.utils.data import DataLoader

from models.bgremover import BGRemover
from train_utils.data_loader import collate_fn, create_datasets
from train_utils.scores import Metrics


class Trainer:
    def __init__(
        self,
        model: BGRemover,
        device: torch.device | str,
        dataset_path: str,
        checkpoint_path: str,
        epochs: int = 5,
        initial_epoch: int = 1,
        batch_size: int = 1,
        train_size: int = 5000,
        val_size: int = 1000,
        learning_rate: float = 1e-5,
        weight_decay: float = 1e-6,
        betas: tuple[float, float] = (0.9, 0.999),
        logger: Logger | None = None,
        do_train_body: bool = True,
        do_train_refiner: bool = True,
    ):
        self.model = model.to(device)
        self.device = torch.device(device)
        train_set, val_set = create_datasets(
            image_size=model.refiner_size,
            dataset_path=dataset_path,
            train_size=train_size,
            val_size=val_size,
            device=device,
        )
        loader_args = dict(
            batch_size=batch_size,
            num_workers=min(os.cpu_count(), batch_size, 8),
            drop_last=True,
            multiprocessing_context="fork" if self.device.type == "mps" else None,
            collate_fn=collate_fn,
        )
        self.train_dataloader = DataLoader(train_set, **loader_args)
        self.eval_dataloader = DataLoader(val_set, **loader_args)

        self.body_optimizer = optim.AdamW(
            model.body.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
            foreach=True,
        )

        self.body_scheduler = optim.lr_scheduler.OneCycleLR(
            self.body_optimizer,
            total_steps=(epochs - initial_epoch + 1) * len(self.train_dataloader),
            max_lr=learning_rate,
            pct_start=0.1,
            cycle_momentum=False,
        )

        self.refiner_optimizer = optim.AdamW(
            model.refiner.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
            foreach=True,
        )

        self.refiner_scheduler = optim.lr_scheduler.OneCycleLR(
            self.refiner_optimizer,
            total_steps=(epochs - initial_epoch + 1) * len(self.train_dataloader),
            max_lr=learning_rate,
            pct_start=0.1,
            cycle_momentum=False,
        )

        self.metrics = Metrics(device=device)
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_path.mkdir(exist_ok=True, parents=True)

        self.epochs = epochs
        self.initial_epoch = initial_epoch

        self.logger = logger

        self.do_train_body = do_train_body
        self.do_train_refiner = do_train_refiner

        self.pbar: tqdm.tqdm | None = None

    def train_epoch(self, epoch: int):
        self.on_train_epoch_start(epoch)

        scores_epic = dict()
        for i, batch in enumerate(self.train_dataloader):
            self.on_train_batch_start(epoch, i)
            scores_batch = {}

            images, true_masks = batch["images"], batch["masks"]
            x1 = self.model.preprocess(images)
            x2 = self.model.segment_image(x1)

            if self.do_train_body:
                masks_pred = self.model.postprocess(x2, self.model.refiner_size)
                scores = self.metrics.calc(masks_pred, true_masks, prefix="body")
                scores_batch.update(scores)
                for key, value in scores.items():
                    if key not in scores_epic:
                        scores_epic[key] = []
                    scores_epic[key].append(value)

                self.body_optimizer.zero_grad(True)
                scores["body_iou_loss"].backward(retain_graph=True)
                scores["body_bce_loss"].backward()
                self.body_optimizer.step()
                self.body_scheduler.step()

            if self.do_train_refiner:
                x = self.model.forward_refiner(x1, x2.detach())
                masks_pred = self.model.postprocess(x, self.model.refiner_size)
                scores = self.metrics.calc(masks_pred, true_masks, prefix="refiner")
                scores_batch.update(scores)
                for key, value in scores.items():
                    if key not in scores_epic:
                        scores_epic[key] = []
                    scores_epic[key].append(value)

                self.refiner_optimizer.zero_grad(True)
                scores["refiner_iou_loss"].backward(retain_graph=True)
                scores["refiner_bce_loss"].backward()
                self.refiner_optimizer.step()
                self.refiner_scheduler.step()

            self.on_train_batch_end(epoch, i, scores_batch)

        self.on_train_epoch_end(epoch, scores_epic)

    def eval_epoch(self, epoch: int):
        self.on_eval_epoch_start(epoch)

        scores_epoch = dict()
        for i, batch in enumerate(self.eval_dataloader):
            self.on_eval_batch_start(epoch, i)

            images, true_masks = batch["images"], batch["masks"]
            masks_pred = self.model(images)
            scores = self.metrics.calc(masks_pred, true_masks)
            for key, value in scores.items():
                if key not in scores_epoch:
                    scores_epoch[key] = []
                scores_epoch[key].append(value)

            self.on_eval_batch_end(epoch, i, scores)

        self.on_eval_epoch_end(epoch, scores_epoch)

    def train(self):
        for epoch in range(self.initial_epoch, self.epochs + 1):
            self.train_epoch(epoch)
            self.eval_epoch(epoch)

    def on_train_batch_start(self, epoch: int, batch: int):
        pass

    def on_train_batch_end(self, epoch: int, batch: int, scores: dict[str, float]):
        scores = {k: float(v) for k, v in scores.items()}
        self.pbar.set_postfix(
            {key: value for key, value in scores.items() if "loss" in key}
        )
        self.pbar.update(1)
        if self.logger is None:
            return
        iteration = (epoch - 1) * len(self.train_dataloader) + batch
        for key, value in scores.items():
            self.logger.report_scalar(
                "train_batch", key, iteration=iteration, value=value
            )
        self.logger.report_scalar(
            "learning_rate",
            "body_learning_rate",
            iteration=iteration,
            value=self.body_scheduler.get_last_lr()[0],
        )
        self.logger.report_scalar(
            "learning_rate",
            "refiner_learning_rate",
            iteration=iteration,
            value=self.refiner_scheduler.get_last_lr()[0],
        )

    def on_train_epoch_start(self, epoch: int):
        self.model = self.model.train()
        self.pbar = tqdm.tqdm(
            total=len(self.train_dataloader), desc=f"Train epoch {epoch}/{self.epochs}"
        )

    def on_train_epoch_end(self, epoch: int, scores: dict[str, list[float]]):
        self.pbar.close()
        if self.logger is None:
            return
        for key, values in scores.items():
            self.logger.report_scalar(
                "train_epoch",
                key,
                iteration=epoch,
                value=(sum(values) / len(values)) if len(values) > 0 else torch.nan,
            )

    def on_eval_batch_start(self, epoch: int, batch: int):
        pass

    def on_eval_batch_end(self, epoch: int, batch: int, scores: dict[str, float]):
        scores = {k: float(v) for k, v in scores.items()}
        self.pbar.set_postfix(
            {key: value for key, value in scores.items() if "loss" in key}
        )
        self.pbar.update(1)
        if self.logger is None:
            return
        iteration = (epoch - 1) * len(self.eval_dataloader) + batch
        for key, value in scores.items():
            self.logger.report_scalar(
                "eval_batch", key, iteration=iteration, value=value
            )

    def on_eval_epoch_start(self, epoch: int):
        self.model = self.model.eval()
        self.pbar = tqdm.tqdm(
            total=len(self.eval_dataloader), desc=f"Eval epoch {epoch}/{self.epochs}"
        )

    def on_eval_epoch_end(self, epoch: int, scores: dict[str, list[float]]):
        self.pbar.close()
        iou = float(1 - sum(scores["iou_loss"]) / len(scores["iou_loss"]))
        state_dict = self.model.state_dict()
        torch.save(
            state_dict,
            self.checkpoint_path / f"checkpoint_{str(epoch).zfill(3)}_{iou:.4f}.pth",
        )
        logging.info(f"Checkpoint {epoch} saved!")
        if self.logger is None:
            return
        for key, values in scores.items():
            self.logger.report_scalar(
                "eval_epoch",
                key,
                iteration=epoch,
                value=(sum(values) / len(values)) if len(values) > 0 else torch.nan,
            )
