"""
Models for each dataset type.
All return logits of shape (batch, num_classes).
"""
import torch.nn as nn


class ResNet50FL(nn.Module):
    """ResNet-50 adapted for FL experiments across image datasets.

    Two adaptations beyond stock torchvision ResNet-50:
    - img_size < 64: replaces the 7×7 stride-2 stem with a 3×3 stride-1 conv and
      removes the maxpool (standard CIFAR fix — otherwise 32px collapses before res stages).
    - in_channels != 3: replaces conv1 to accept grayscale input.
    The FC head is always replaced to match num_classes.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 10, img_size: int = 32):
        super().__init__()
        from torchvision.models import resnet50
        net = resnet50(weights=None)

        if img_size < 64:
            # Small-image stem: 3×3, stride=1, no maxpool
            net.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
            net.maxpool = nn.Identity()
        elif in_channels != 3:
            net.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        net.fc = nn.Linear(net.fc.in_features, num_classes)
        self.net = net

    def forward(self, x):
        return self.net(x)


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
    # Grayscale datasets
    in_ch = 1 if name in ("femnist", "medmnist", "tissuemnist") else 3
    img_size = 224 if name == "isic" else 32
    return ResNet50FL(in_channels=in_ch, num_classes=num_classes, img_size=img_size)
