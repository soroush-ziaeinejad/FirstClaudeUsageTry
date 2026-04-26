"""
ISIC 2019 skin lesion classification (cross-silo setting).
Expects data pre-downloaded to data_dir/isic/ with structure:
  data_dir/isic/train/{class_name}/*.jpg
Download from: https://challenge.isic-archive.com/data/#2019
"""
import os
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import torch
from fl_datasets.partitioning import get_client_subset


def load_isic(client_id, num_clients, alpha, config):
    data_dir = os.path.join(config.get("data_dir", "./data"), "isic", "train")
    batch_size = config.get("batch_size", 32)

    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"ISIC data not found at {data_dir}. "
            "Download from https://challenge.isic-archive.com/data/#2019 "
            "and organize as data/isic/train/{class_name}/*.jpg"
        )

    tf_train = T.Compose([T.Resize(224), T.RandomHorizontalFlip(),
                          T.RandomVerticalFlip(), T.ToTensor(),
                          T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    tf_test = T.Compose([T.Resize(224), T.ToTensor(),
                         T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    full_ds = ImageFolder(data_dir, transform=tf_train)
    num_classes = len(full_ds.classes)

    n_test = int(0.15 * len(full_ds))
    n_train = len(full_ds) - n_test
    train_full, test_full = random_split(full_ds, [n_train, n_test],
                                         generator=torch.Generator().manual_seed(42))

    subset = get_client_subset(train_full, client_id, num_clients, alpha)

    # Apply test transform to test split
    test_full.dataset.transform = tf_test

    train_loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_full, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader, num_classes