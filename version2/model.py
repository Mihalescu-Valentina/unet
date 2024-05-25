import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from metrics import LossMetric
from dataset import MVDataset
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from torchmetrics import DiceScore
from torchmetrics.classification import BinaryJaccardIndex
import torchvision
import matplotlib.pyplot as plt

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512], config=None
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dice_score = DiceScore()
        self.iou = BinaryJaccardIndex()
        self.config = nn.ParameterDict(config)
        self.to(self.config['device'])
        self.loss_fn = nn.BCEWithLogitsLoss()

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.optimizer = self.configure_optimizer(self.config['optimizer'], self.config['lr'])
        self.scaler = torch.cuda.amp.GradScaler()

        self.metrics = {'train': {'loss': [], 'dice_score': [], 'iou': []},
                        'val': {'loss': [], 'dice_score': [], 'iou': []}}
        self.load_data()

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)

    def save_checkpoint(self, state, filename):
        print("=> Saving checkpoint")
        torch.save(state, filename)

    def load_checkpoint(self, checkpoint):
        print("=> Loading checkpoint")
        self.load_state_dict(checkpoint["state_dict"])

    def get_loaders(self,
                    train_dir,
                    train_maskdir,
                    val_dir,
                    val_maskdir,
                    batch_size,
                    train_transform,
                    val_transform,
                    num_workers=4,
                    pin_memory=True,
                    ):
        train_ds = MVDataset(
            image_dir=train_dir,
            mask_dir=train_maskdir,
            transform=train_transform,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=True,
        )

        val_ds = MVDataset(
            image_dir=val_dir,
            mask_dir=val_maskdir,
            transform=val_transform,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=False,
        )
        return train_loader, val_loader

    def load_data(self):
        train_transform = A.Compose(
            [
                A.Resize(height=self.config['image_height'], width=self.config['image_width']),
                A.Rotate(limit=35, p=1.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                ToTensorV2(),
            ],
        )

        val_transforms = A.Compose(
            [
                A.Resize(height=self.config['image_height'], width=self.config['image_width']),
                ToTensorV2(),
            ],
        )
        train_loader, val_loader = self.get_loaders(
            self.config['train_img_dir'],
            self.config['train_mask_dir'],
            self.config['val_img_dir'],
            self.config['val_mask_dir'],
            self.config['batch_size'],
            train_transform,
            val_transforms,
            self.config['num_workers'],
            self.config['pin_memory'],
        )

        self.train_loader = train_loader
        self.val_loader = val_loader

    def configure_optimizer(self, optimizer, lr):
        if optimizer == 'Adam':
            return torch.optim.Adam(self.parameters(), lr=lr)
        elif optimizer == 'SGD':
            return torch.optim.SGD(self.parameters(), lr=lr)
        else:
            raise ValueError('Optimizer not supported')

    def train_step(self):
        self.train()
        loop = tqdm(self.train_loader, leave=False)
        epoch_loss = 0
        epoch_dice = 0
        epoch_iou = 0

        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device=self.config['device'])
            targets = targets.float().unsqueeze(1).to(device=self.config['device'])

            with torch.cuda.amp.autocast():
                predictions = self.forward(data)
                loss = self.loss_fn(predictions, targets)
            
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            dice = self.dice_score(predictions, targets)
            iou = self.iou(predictions, targets)

            epoch_loss += loss.item()
            epoch_dice += dice.item()
            epoch_iou += iou.item()

            loop.set_postfix(loss=loss.item(), dice=dice.item(), iou=iou.item())

        avg_loss = epoch_loss / len(self.train_loader)
        avg_dice = epoch_dice / len(self.train_loader)
        avg_iou = epoch_iou / len(self.train_loader)

        self.metrics['train']['loss'].append(avg_loss)
        self.metrics['train']['dice_score'].append(avg_dice)
        self.metrics['train']['iou'].append(avg_iou)

    def validate_step(self):
        self.eval()
        epoch_loss = 0
        epoch_dice = 0
        epoch_iou = 0

        with torch.no_grad():
            for idx, (x, y) in enumerate(self.val_loader):
                x = x.to(device=self.config['device'])
                y = y.float().unsqueeze(1).to(device=self.config['device'])
                
                preds = torch.sigmoid(self.forward(x))
                preds = (preds > 0.5).float()
                loss = self.loss_fn(preds, y)
                
                dice = self.dice_score(preds, y)
                iou = self.iou(preds, y)
                
                epoch_loss += loss.item()
                epoch_dice += dice.item()
                epoch_iou += iou.item()
                
                if idx % 10 == 0:  # Save every 10th batch
                    torchvision.utils.save_image(preds, f"/home/valentina/pred_images/pred_{idx}.png")
                    torchvision.utils.save_image(y, f"/home/valentina/saved_images/image_{idx}.png")

        avg_loss = epoch_loss / len(self.val_loader)
        avg_dice = epoch_dice / len(self.val_loader)
        avg_iou = epoch_iou / len(self.val_loader)

        self.metrics['val']['loss'].append(avg_loss)
        self.metrics['val']['dice_score'].append(avg_dice)
        self.metrics['val']['iou'].append(avg_iou)

        self.log_metrics('val', avg_loss, avg_dice, avg_iou)

    def log_metrics(self, phase, loss, dice, iou):
        print(f"{phase} - Loss: {loss:.4f}, Dice Score: {dice:.4f}, IOU: {iou:.4f}")

    def plot_metrics(self):
        epochs = range(1, len(self.metrics['val']['loss']) + 1)

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 3, 1)
        plt.plot(epochs, self.metrics['val']['loss'], 'b', label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Validation Loss')
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(epochs, self.metrics['val']['dice_score'], 'b', label='Validation Dice Score')
        plt.xlabel('Epochs')
        plt.ylabel('Dice Score')
        plt.title('Validation Dice Score')
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.plot(epochs, self.metrics['val']['iou'], 'b', label='Validation IOU')
        plt.xlabel('Epochs')
        plt.ylabel('IOU')
        plt.title('Validation IOU')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def fit(self):
        print(self.parameters())
        self.to(self.config['device'])
        print(self.parameters())
        if self.config['load_model']:
            self.load_checkpoint(torch.load("/home/valentina/my_checkpoint.pth.tar"))
        for epoch in range(self.config['num_epochs']):
            self.train_step()
            checkpoint = {
                "state_dict": self.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            self.save_checkpoint(checkpoint, "/home/valentina/my_checkpoint.pth.tar")
            self.validate_step()

        self.plot_metrics()

def main():
    config = {
        "lr": 1e-4,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "batch_size": 16,
        "num_epochs": 32,
        "num_workers": 2,
        "image_height": 572,
        "image_width": 572,
        "pin_memory": True,
        "load_model": False,
        "train_img_dir": "/home/valentina/training_images",
        "train_mask_dir": "/home/valentina/training_binary_masks",
        "val_img_dir": "/home/valentina/validation_images",
        "val_mask_dir": "/home/valentina/validation_binary_masks",
        "optimizer": 'Adam'
    }

    unet = UNET(in_channels=3, out_channels=1, config=config)
    unet.fit()

if __name__ == "__main__":
    main()
