from datasets.visdrone import VisDroneDataset


dataset = VisDroneDataset(
    root="data/raw/VisDrone2019-DET",
    split="train",
)

print("Dataset size:", len(dataset))

image, target = dataset[0]

print("Image shape:", image.shape)
print("Image dtype:", image.dtype)

print("Boxes shape:", target["boxes"].shape)
print("Labels shape:", target["labels"].shape)

print("First 5 boxes:")
print(target["boxes"][:5])

print("First 5 labels:")
print(target["labels"][:5])