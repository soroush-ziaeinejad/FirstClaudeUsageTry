"""
Lightweight MiniLM embedding module — always runs on CPU.
Sentence-transformers doesn't fully support MPS; CPU is fast enough for
embedding ~hundreds of short client descriptors per round.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, List


class EmbeddingModule:
    _instance = None  # module-level singleton to avoid reloading per round

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        # force CPU — MPS doesn't support all sentence-transformer ops
        self._model = SentenceTransformer(model_name, device="cpu")
        self._model.eval()

    @classmethod
    def get(cls, model_name: str = "all-MiniLM-L6-v2") -> "EmbeddingModule":
        if cls._instance is None:
            cls._instance = cls(model_name)
        return cls._instance

    def embed(self, descriptors: Dict[str, str]) -> Dict[str, np.ndarray]:
        """
        Embed a dict of {client_id: descriptor_text}.
        Returns {client_id: embedding_vector (384-dim)}.
        """
        if not descriptors:
            return {}
        cids = list(descriptors.keys())
        texts = [descriptors[c] for c in cids]

        import torch
        with torch.no_grad():
            vecs = self._model.encode(
                texts,
                batch_size=64,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        return {cid: vec for cid, vec in zip(cids, vecs)}
