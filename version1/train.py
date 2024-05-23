import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
import matplotlib.pyplot as plt
from utils2 import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 32
NUM_WORKERS = 2
IMAGE_HEIGHT = 572  # 1280 originally
IMAGE_WIDTH = 572  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "/home/valentina/training_images"
TRAIN_MASK_DIR = "/home/valentina/training_binary_masks"
VAL_IMG_DIR = "/home/valentina/validation_images"
VAL_MASK_DIR = "/home/valentina/validation_binary_masks"

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            ToTensorV2(),
        ],
    )

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("/home/valentina/my_checkpoint.pth.tar"), model)

    scaler = torch.cuda.amp.GradScaler()

    # Initialize lists to store metrics
    train_losses = []
    val_losses = []
    val_dice_scores = []
    val_iou_scores = []

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # Save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # Check accuracy and get metrics
        val_loss, val_dice, val_iou = check_accuracy(val_loader, model, loss_fn, device=DEVICE)

        # Append metrics to lists
        val_losses.append(val_loss)
        val_dice_scores.append(val_dice)
        val_iou_scores.append(val_iou)

        # Print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="/home/valentina/saved_images", device=DEVICE
        )

    # Plot metrics
    epochs = range(1, NUM_EPOCHS + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Validation Loss over Epochs')
    plt.legend()
    plt.savefig('/home/valentina/validation_loss.png')

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, val_dice_scores, label='Validation Dice Score')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Score')
    plt.title('Validation Dice Score over Epochs')
    plt.legend()
    plt.savefig('/home/valentina/validation_dice_score.png')

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, val_iou_scores, label='Validation IoU Score')
    plt.xlabel('Epochs')
    plt.ylabel('IoU Score')
    plt.title('Validation IoU Score over Epochs')
    plt.legend()
    plt.savefig('/home/valentina/validation_iou_score.png')


if __name__ == "__main__":
    main()
