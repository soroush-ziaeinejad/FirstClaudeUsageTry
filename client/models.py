"""
Lightweight models for each dataset type.
All return logits of shape (batch, num_classes).
"""
import torch.nn as nn


class SmallCNN(nn.Module):
    """For CIFAR-10/100, FEMNIST, MedMNIST, ISIC (after resize to 32 or 224)."""
    def __init__(self, in_channels: int, num_classes: int, img_size: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class CharLSTM(nn.Module):
    """For Shakespeare next-character prediction."""
    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden: int = 256, layers: int = 2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden, layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        out, _ = self.lstm(self.embed(x))
        return self.fc(out[:, -1, :])


def get_model(dataset_name: str, num_classes: int) -> nn.Module:
    name = dataset_name.lower()
    if name == "shakespeare":
        return CharLSTM(vocab_size=num_classes)
    elif name in ("isic",):
        return SmallCNN(in_channels=3, num_classes=num_classes, img_size=224)
    elif name == "femnist":
        return SmallCNN(in_channels=1, num_classes=num_classes)
    else:
        in_ch = 1 if name in ("medmnist", "pathmnist", "dermamnist", "bloodmnist") else 3
        return SmallCNN(in_channels=in_ch, num_classes=num_classes)
