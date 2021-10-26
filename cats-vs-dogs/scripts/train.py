import time
import os
import copy
import time
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
from models.resnet import resnet50

IMAGE_SIZE = (224, 224)


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_model(model_name):
    if model_name == "resnet50":
        model = resnet50(pretrained=True)
        model.fc = nn.Linear(in_features=2048, out_features=2, bias=True)
    else:
        raise Exception("Invalid model name")
    return model


def get_transformations(augment):
    transformations = dict()
    if augment == True:
        transformations["train"] = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=45),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                transforms.Resize(IMAGE_SIZE),
            ]
        )
    else:
        transformations["train"] = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(IMAGE_SIZE),
            ]
        )
    transformations["val"] = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(IMAGE_SIZE),
        ]
    )
    return transformations


def train_step(batch_images, batch_labels, model, criterion, optimizer, device):
    batch_images = batch_images.to(device)
    labels = batch_labels.to(device)
    optimizer.zero_grad()

    with torch.set_grad_enabled(mode=True):
        output = model(batch_images)
        _, preds = torch.max(output, 1)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

    batch_loss = (loss * batch_images.size(0)).item()
    batch_corrects = torch.sum(preds == labels.data).item()
    return batch_loss, batch_corrects


def val_step(batch_images, batch_labels, model, criterion, device):
    batch_images = batch_images.to(device)
    labels = batch_labels.to(device)

    with torch.set_grad_enabled(False):
        output = model(batch_images)
        _, preds = torch.max(output, 1)
        loss = criterion(output, labels)

    batch_loss = (loss * batch_images.size(0)).item()
    batch_corrects = torch.sum(preds == labels.data).item()
    return batch_loss, batch_corrects


def train(
    model,
    image_datasets,
    dataloaders,
    criterion,
    optimizer,
    lr_scheduler,
    num_epochs,
    device,
    output_dir,
):
    since = time.time()

    best_model_weight = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # open log file
    log_file_path = os.path.join(output_dir, "log.csv")
    with open(log_file_path, "w") as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc\n")

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch + 1, num_epochs))

        # train loop
        model.train()

        train_running_loss = 0.0
        train_corrects = 0

        for batch_images, batch_labels in tqdm(dataloaders["train"]):
            batch_loss, batch_corrects = train_step(
                batch_images, batch_labels, model, criterion, optimizer, device
            )
            train_running_loss += batch_loss
            train_corrects += batch_corrects
            lr_scheduler.step()

        train_loss = train_running_loss / len(image_datasets["train"])
        train_acc = float(train_corrects) / len(image_datasets["train"])

        # val loop
        model.eval()

        val_running_loss = 0.0
        val_corrects = 0

        for batch_images, batch_labels in tqdm(dataloaders["val"]):
            batch_loss, batch_corrects = val_step(
                batch_images, batch_labels, model, criterion, device
            )
            val_running_loss += batch_loss
            val_corrects += batch_corrects

        val_loss = val_running_loss / len(image_datasets["val"])
        val_acc = float(val_corrects) / len(image_datasets["val"])

        # log
        with open(log_file_path, "a") as f:
            f.write(
                "{},{:.5f},{:.4f},{:.5f},{:.4f}\n".format(
                    epoch + 1, train_loss, train_acc, val_loss, val_acc
                )
            )

        print("Train Loss: {:.5f} Acc: {:.4f}".format(train_loss, train_acc))
        print("Valid Loss: {:.5f} Acc: {:.4f}".format(val_loss, val_acc))

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_weight = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(
        "Training completed in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best valid accuracy: {:4f}".format(best_acc))

    model.load_state_dict(best_model_weight)
    return model


def main(args):
    # Empty cache
    torch.cuda.empty_cache()

    # Set seed
    set_seed(0)

    # Create directory to save model and csv output
    if args.augment:
        output_dir_name = "{}_lr_{}_augment".format(args.model_name, args.lr)
    else:
        output_dir_name = "{}_lr_{}".format(args.model_name, args.lr)
    output_dir = os.path.join(args.model_dir, output_dir_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_dir = args.data_dir
    transformations = get_transformations(args.augment)
    batch_size = args.batch_size
    num_workers = args.num_workers

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), transformations[x])
        for x in ["train", "val"]
    }

    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        for x in ["train", "val"]
    }

    class_names = image_datasets["train"].classes
    print("Classes: {}".format(class_names))
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))
    print()

    # Load model
    model_name = args.model_name
    model = get_model(model_name)
    model.to(device)

    lr = args.lr
    num_epochs = args.num_epochs
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    trained_model = train(
        model,
        image_datasets,
        dataloaders,
        criterion,
        optimizer,
        exp_lr_scheduler,
        num_epochs,
        device,
        output_dir,
    )

    torch.save(trained_model, os.path.join(output_dir, "model.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, default="../datasets/", help="path to dataset folder"
    )
    parser.add_argument(
        "--model_name", type=str, default="resnet50", help="name of CNN model"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="../models/",
        help="dir for saving trained models",
    )
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001, help="learning_rate")
    parser.add_argument(
        "--augment",
        default=False,
        action="store_true",
        help="whether to perform image augmentation to train images",
    )
    args = parser.parse_args()
    print(args)
    main(args)
