"""
Shakespeare next-character prediction. Each client = one speaking role.
Uses a lightweight character-level dataset built from the Tiny Shakespeare text.
"""
import os
import urllib.request
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
_SEQ_LEN = 80
_CHARS = "".join([chr(i) for i in range(32, 127)])  # printable ASCII
_VOCAB = {c: i for i, c in enumerate(_CHARS)}
NUM_CLASSES = len(_CHARS)


class CharDataset(Dataset):
    def __init__(self, text: str, seq_len: int = _SEQ_LEN):
        encoded = [_VOCAB.get(c, 0) for c in text]
        self.x = torch.tensor(encoded[:-1], dtype=torch.long)
        self.y = torch.tensor(encoded[1:], dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.x) - self.seq_len)

    def __getitem__(self, idx):
        return self.x[idx: idx + self.seq_len], self.y[idx: idx + self.seq_len]


def _download(data_dir):
    path = os.path.join(data_dir, "shakespeare.txt")
    if not os.path.exists(path):
        os.makedirs(data_dir, exist_ok=True)
        urllib.request.urlretrieve(_URL, path)
    with open(path, "r") as f:
        return f.read()


def load_shakespeare(client_id, num_clients, alpha, config):
    data_dir = config.get("data_dir", "./data")
    batch_size = config.get("batch_size", 32)

    text = _download(data_dir)
    # Split text into num_clients chunks (natural sequential partition)
    chunk_size = len(text) // num_clients
    client_text = text[client_id * chunk_size: (client_id + 1) * chunk_size]

    full_ds = CharDataset(client_text)
    n_train = int(0.9 * len(full_ds))
    n_test = len(full_ds) - n_train
    train_ds, test_ds = random_split(full_ds, [n_train, n_test],
                                     generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader, NUM_CLASSES