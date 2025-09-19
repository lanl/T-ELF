# Termite/VectorInjector.py
import torch
from transformers import AutoTokenizer, AutoModel
from typing import Iterable, List

class Vectorizer:
    """Computes embeddings; does NOT talk to any DB."""

    def __init__(self, model_name: str = "malteos/scincl", device: str = None, max_length: int = 512):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval().to(self.device)
        self.max_length = max_length

    @torch.inference_mode()
    def encode(self, texts: Iterable[str]) -> List[List[float]]:
        outs = []
        for t in texts:
            tokens = self.tokenizer(
                t if isinstance(t, str) else "",
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            hidden = self.model(**tokens).last_hidden_state  # [1, L, H]
            emb = hidden.mean(dim=1).squeeze(0)              # mean pool
            outs.append(emb.detach().cpu().tolist())
        return outs
