from datasets.cifar import load_cifar10, load_cifar100
from datasets.femnist import load_femnist
from datasets.shakespeare import load_shakespeare
from datasets.medmnist import load_medmnist
from datasets.isic import load_isic


def get_dataset(name: str, client_id: int, num_clients: int, alpha: float, config: dict):
    """
    Single entry point for all datasets.
    Returns (train_loader, test_loader, num_classes).
    All partitioning is Dirichlet(alpha) unless the dataset has natural splits.
    """
    name = name.lower()
    kwargs = dict(client_id=client_id, num_clients=num_clients, alpha=alpha, config=config)

    if name == "cifar10":
        return load_cifar10(**kwargs)
    elif name == "cifar100":
        return load_cifar100(**kwargs)
    elif name == "femnist":
        return load_femnist(**kwargs)
    elif name == "shakespeare":
        return load_shakespeare(**kwargs)
    elif name in ("medmnist", "pathmnist", "dermamnist", "bloodmnist"):
        return load_medmnist(subset=name if name != "medmnist" else "pathmnist", **kwargs)
    elif name == "isic":
        return load_isic(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}")