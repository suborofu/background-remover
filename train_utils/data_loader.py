import csv
import random
from pathlib import Path

import albumentations as A
import cv2 as cv
import numpy as np
import torch.utils.data
from PIL import Image


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, path, dataset_size, image_size=128, is_train=True, device="cpu"):
        self.dataset_size = dataset_size
        self.image_size = image_size
        self.is_train = is_train
        self.path = path
        self.device = device
        self.content = self.__load_dataset(path)

    def __len__(self):
        return self.dataset_size

    def __load_dataset(self, path):
        with open(Path(path) / "annotations/meta/meta.csv", "r") as f:
            reader = csv.reader(f)
            content = []
            for idx, row in enumerate(reader):
                if idx == 0:
                    continue
                image_id, _, _, _, _, _, test, train, valid = row
                if self.is_train:
                    if train == "True":
                        content.append("train/" + image_id)
                    elif test == "True":
                        content.append("test/" + image_id)
                else:
                    if valid == "True":
                        content.append("val/" + image_id)
        return sorted(content)

    def __transform_aug(self, image, mask):
        H, W = image.shape[:2]

        mask[mask > 0] = 1
        if self.is_train:
            transform = A.Compose(
                [
                    A.Rotate(limit=10, border_mode=cv.BORDER_REFLECT, p=0.5),
                    A.RandomCrop(
                        height=H * random.randint(80, 100) // 100,
                        width=W * random.randint(80, 100) // 100,
                        p=1.0,
                    ),
                    A.Resize(self.image_size, self.image_size),
                    A.HorizontalFlip(p=0.5),
                    A.ColorJitter(p=1),
                ]
            )
        else:
            transform = A.Resize(self.image_size, self.image_size)
        result = transform(image=image, mask=mask)
        image = torch.tensor(
            result["image"] / 255, dtype=torch.float32, device=self.device
        ).permute(2, 0, 1)
        mask = torch.tensor(result["mask"], dtype=torch.uint8, device=self.device)
        return image, mask

    def __getitem__(self, index):
        index = (
            random.randint(0, len(self.content) - 1)
            if self.is_train
            else (index % len(self.content))
        )
        image_id = self.content[index]

        image = np.array(
            Image.open(self.path / "images" / (image_id + ".jpg")).convert("RGB")
        )
        mask = np.array(
            Image.open(self.path / "annotations" / (image_id + ".png")).convert("L")
        )

        image, mask = self.__transform_aug(image, mask)
        result = {
            "images": image,
            "masks": mask,
        }
        return result


def create_datasets(
    dataset_path, image_size=512, train_size=100, val_size=100, device="cpu"
):
    dataset_path = Path(dataset_path)
    train_ds = CustomDataset(
        dataset_path,
        image_size=image_size,
        dataset_size=train_size,
        is_train=True,
        device=device,
    )
    val_ds = CustomDataset(
        dataset_path,
        image_size=image_size,
        dataset_size=val_size,
        is_train=False,
        device=device,
    )

    return train_ds, val_ds


def collate_fn(data):
    images, masks = [], []
    for d in data:
        images.append(d["images"])
        masks.append(d["masks"])
    return {
        "images": torch.stack(images),
        "masks": torch.stack(masks),
    }


if __name__ == "__main__":
    import config

    train, val = create_datasets(dataset_path=config.DATASET_PATH)
    for i, data in enumerate(train):
        image = data["images"].numpy().transpose(1, 2, 0)
        mask = data["masks"].numpy()
        Image.fromarray(np.uint8(image * 255)).save(f"temp/test_{i}_img.png")
        Image.fromarray(np.uint8(mask * 255)).save(f"temp/test_{i}_mask.png")
        if i == 10:
            break

    for i, data in enumerate(val):
        image = data["images"].numpy().transpose(1, 2, 0)
        mask = data["masks"].numpy()
        Image.fromarray(np.uint8(image * 255)).save(f"temp/val_{i}_img.png")
        Image.fromarray(np.uint8(mask * 255)).save(f"temp/val_{i}_mask.png")
        if i == 10:
            break
