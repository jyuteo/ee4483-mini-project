import os
import torch
import random
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn

from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

IMAGE_SIZE = (224, 224)


class TestImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_list = os.listdir(self.image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_name = self.image_list[index]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image, image_name


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_model(model_path):
    model = torch.load(os.path.join(model_path, "model.pt"))
    return model


def get_labels(batch_images, model, device):
    batch_images = batch_images.to(device)
    with torch.no_grad():
        output = model(batch_images)
        labels = torch.argmax(output, dim=1)
    return labels.tolist()


def predict(
    model, dataloader, device, model_path,
):
    model.eval()
    id = []
    label = []
    pbar = tqdm(dataloader)

    for batch_images, batch_image_names in pbar:
        image_names = list(batch_image_names)
        id += [int(name.split(".")[0]) for name in image_names]
        batch_labels = get_labels(batch_images, model, device)
        label += batch_labels

    df = pd.DataFrame({"id": id, "label": label})
    df = df.sort_values(by="id")
    df.to_csv(os.path.join(model_path, "submission.csv"), index=False)


def main(args):
    set_seed(0)

    data_dir = args.data_dir
    transformation = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize(IMAGE_SIZE),]
    )
    batch_size = args.batch_size
    num_workers = args.num_workers

    image_dataset = TestImageDataset(
        os.path.join(data_dir, "test"), transform=transformation
    )

    dataloader = torch.utils.data.DataLoader(
        image_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))

    model_path = args.model_path
    model = get_model(model_path)
    model.to(device)

    predict(model, dataloader, device, model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, default="../datasets/", help="path to dataset folder"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="../models/resnet50_lr_0.0001_augment_completed",
        help="path to CNN model",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    print(args)
    main(args)
