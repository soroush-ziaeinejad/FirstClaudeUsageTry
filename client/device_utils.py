import torch


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def clear_device_cache(device: torch.device):
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def dataloader_kwargs(device: torch.device) -> dict:
    """DataLoader kwargs tuned per device."""
    if device.type == "mps":
        # MPS doesn't support pinned memory; multiprocessing can cause issues
        return {"num_workers": 0, "pin_memory": False}
    if device.type == "cuda":
        return {"num_workers": 2, "pin_memory": True}
    return {"num_workers": 2, "pin_memory": False}
