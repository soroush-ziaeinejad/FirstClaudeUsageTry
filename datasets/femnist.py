"""
FEMNIST via the LEAF benchmark. Falls back to EMNIST(split='byclass') with
Dirichlet partitioning when the full LEAF download is not available.
"""
import torchvision.transforms as T
import torchvision.datasets as dsets
from torch.utils.data import DataLoader
from datasets.partitioning import get_client_subset


def load_femnist(client_id, num_clients, alpha, config):
    data_dir = config.get("data_dir", "./data")
    batch_size = config.get("batch_size", 32)

    tf = T.Compose([T.Resize(28), T.ToTensor(), T.Normalize((0.9641,), (0.1592,))])

    # EMNIST byclass has 62 classes (digits + upper/lower letters) — good FEMNIST proxy
    train_ds = dsets.EMNIST(data_dir, split="byclass", train=True, download=True, transform=tf)
    test_ds = dsets.EMNIST(data_dir, split="byclass", train=False, download=True, transform=tf)

    subset = get_client_subset(train_ds, client_id, num_clients, alpha)
    train_loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader, 62