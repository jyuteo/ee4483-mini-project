import os
import torch
import random
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn

from tqdm import tqdm
from torchvision import datasets, transforms

IMAGE_SIZE = (224, 224)


class ImageFolderWithPaths(datasets.ImageFolder):
    """
    Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        return original_tuple + (path,)


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


def predict(batch_images, model, device):
    batch_images = batch_images.to(device)
    with torch.no_grad():
        output = model(batch_images)
    return output.cpu().tolist()


def calculate_metrics(output, labels, criterion):
    output = torch.Tensor(output).float()
    labels = torch.Tensor(labels).long()
    preds = torch.argmax(output, dim=1)
    max_prob = output[torch.arange(labels.size(0)), preds]
    accuracy = torch.mean((preds == labels).float())
    loss = criterion(output, labels)
    return accuracy.item(), loss.item(), preds.tolist(), max_prob.tolist()


def evaluate(
    model, dataloader, criterion, device, model_path,
):
    model.eval()
    paths = []
    labels = []
    output = []
    pbar = tqdm(dataloader)

    for batch_images, batch_labels, batch_paths in pbar:
        paths += list(batch_paths)
        labels += batch_labels.cpu().tolist()
        batch_pred = predict(batch_images, model, device)
        output += batch_pred

    df = pd.DataFrame({"image_path": paths, "image_label": labels})
    test_acc, test_loss, pred_classes, class_prob = calculate_metrics(
        output, labels, criterion
    )
    df["predicted_class"] = pred_classes
    df["predicted_class_probability"] = class_prob
    df.to_csv(os.path.join(model_path, "val.csv"), index=False)

    print("Valid Loss: {:.5f} Acc: {:.4f}".format(test_loss, test_acc))


def main(args):
    torch.cuda.empty_cache()

    set_seed(0)

    data_dir = args.data_dir
    transformation = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize(IMAGE_SIZE),]
    )
    batch_size = args.batch_size
    num_workers = args.num_workers

    image_dataset = ImageFolderWithPaths(os.path.join(data_dir, "val"), transformation)

    dataloader = torch.utils.data.DataLoader(
        image_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))

    model_path = args.model_path
    model = get_model(model_path)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    evaluate(model, dataloader, criterion, device, model_path)


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
