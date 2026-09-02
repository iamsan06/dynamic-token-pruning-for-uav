import torch

from datasets.visdrone import VisDroneDataset
from datasets.transforms import TrainTransforms


def test_dataset_with_transforms():
    transform = TrainTransforms(size=640)

    dataset = VisDroneDataset(
        root="data/raw/VisDrone2019-DET",
        split="train",
        transform=transform,
    )

    image, target = dataset[0]

    assert image.shape == (3, 640, 640)
    assert image.dtype == torch.float32

    assert target["boxes"].ndim == 2
    assert target["boxes"].shape[1] == 4

    assert target["labels"].ndim == 1
    assert target["boxes"].shape[0] == target["labels"].shape[0]

    assert torch.all(target["boxes"] >= 0)
    assert torch.all(target["boxes"] <= 640)