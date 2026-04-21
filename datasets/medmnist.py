import torchvision.transforms as T
from torch.utils.data import DataLoader
from datasets.partitioning import get_client_subset

_SUBSETS = {
    "pathmnist": 9,
    "dermamnist": 7,
    "bloodmnist": 8,
    "tissuemnist": 8,
}


def load_medmnist(client_id, num_clients, alpha, config, subset="pathmnist"):
    try:
        import medmnist
        from medmnist import INFO
    except ImportError:
        raise ImportError("Install medmnist: pip install medmnist")

    data_dir = config.get("data_dir", "./data")
    batch_size = config.get("batch_size", 32)

    info = INFO[subset]
    DataClass = getattr(medmnist, info["python_class"])
    num_classes = len(info["label"])

    tf = T.Compose([T.ToTensor(), T.Normalize(mean=[0.5], std=[0.5])])

    train_ds = DataClass(split="train", transform=tf, download=True, root=data_dir)
    test_ds = DataClass(split="test", transform=tf, download=True, root=data_dir)

    # medmnist labels are (N,1) shaped — flatten for partitioning
    class WrappedDS:
        def __init__(self, ds):
            self.ds = ds
        def __len__(self):
            return len(self.ds)
        def __getitem__(self, idx):
            x, y = self.ds[idx]
            return x, int(y.squeeze())

    train_wrapped = WrappedDS(train_ds)
    subset_ds = get_client_subset(train_wrapped, client_id, num_clients, alpha)
    test_wrapped = WrappedDS(test_ds)

    train_loader = DataLoader(subset_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_wrapped, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader, num_classes