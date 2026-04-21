import numpy as np
from torch.utils.data import Subset


def dirichlet_partition(targets, num_clients: int, alpha: float, seed: int = 42):
    """
    Partition sample indices using Dirichlet(alpha) across num_clients.
    Returns list of index arrays, one per client.
    """
    rng = np.random.default_rng(seed)
    targets = np.array(targets)
    classes = np.unique(targets)
    client_indices = [[] for _ in range(num_clients)]

    for c in classes:
        idx = np.where(targets == c)[0]
        rng.shuffle(idx)
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        proportions = (proportions * len(idx)).astype(int)
        # Fix rounding so we don't drop samples
        proportions[-1] = len(idx) - proportions[:-1].sum()
        splits = np.split(idx, np.cumsum(proportions[:-1]))
        for client_id, split in enumerate(splits):
            client_indices[client_id].extend(split.tolist())

    return client_indices


def get_client_subset(dataset, client_id: int, num_clients: int, alpha: float):
    targets = [dataset[i][1] for i in range(len(dataset))]
    all_indices = dirichlet_partition(targets, num_clients, alpha)
    return Subset(dataset, all_indices[client_id])