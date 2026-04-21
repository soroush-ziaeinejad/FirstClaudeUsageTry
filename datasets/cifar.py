import torchvision.transforms as T
import torchvision.datasets as dsets
from torch.utils.data import DataLoader
from datasets.partitioning import get_client_subset

_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD = (0.2023, 0.1994, 0.2010)
_CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
_CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def _make_loaders(train_subset, test_dataset, batch_size):
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader


def load_cifar10(client_id, num_clients, alpha, config):
    data_dir = config.get("data_dir", "./data")
    batch_size = config.get("batch_size", 32)

    train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                          T.ToTensor(), T.Normalize(_CIFAR10_MEAN, _CIFAR10_STD)])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(_CIFAR10_MEAN, _CIFAR10_STD)])

    train_ds = dsets.CIFAR10(data_dir, train=True, download=True, transform=train_tf)
    test_ds = dsets.CIFAR10(data_dir, train=False, download=True, transform=test_tf)

    subset = get_client_subset(train_ds, client_id, num_clients, alpha)
    return *_make_loaders(subset, test_ds, batch_size), 10


def load_cifar100(client_id, num_clients, alpha, config):
    data_dir = config.get("data_dir", "./data")
    batch_size = config.get("batch_size", 32)

    train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                          T.ToTensor(), T.Normalize(_CIFAR100_MEAN, _CIFAR100_STD)])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(_CIFAR100_MEAN, _CIFAR100_STD)])

    train_ds = dsets.CIFAR100(data_dir, train=True, download=True, transform=train_tf)
    test_ds = dsets.CIFAR100(data_dir, train=False, download=True, transform=test_tf)

    subset = get_client_subset(train_ds, client_id, num_clients, alpha)
    return *_make_loaders(subset, test_ds, batch_size), 100