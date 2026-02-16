from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import faiss
except ImportError:  # pragma: no cover
    faiss = None

from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class RagResult:
    snippets: list[str]
    scores: list[float]


class FAISSRetriever:
    """Simple local RAG retriever over dataset metadata + heuristics."""

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(max_features=512)
        self.index = None
        self.documents: list[str] = []
        self.doc_vectors: np.ndarray | None = None

    def build_knowledge_base(self, df: pd.DataFrame) -> None:
        self.documents = self._documents_from_dataframe(df)
        if not self.documents:
            self.documents = ["No domain context available."]

        vectors = self.vectorizer.fit_transform(self.documents).toarray().astype("float32")
        self.doc_vectors = vectors

        if faiss is not None:
            dim = vectors.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            normalized = self._normalize(vectors)
            self.index.add(normalized)

    def retrieve(self, query: str, top_k: int = 3) -> RagResult:
        if not self.documents:
            return RagResult(snippets=[], scores=[])

        query_vec = self.vectorizer.transform([query]).toarray().astype("float32")
        query_vec = self._normalize(query_vec)

        if self.index is not None:
            scores, ids = self.index.search(query_vec, min(top_k, len(self.documents)))
            ranked_ids = ids[0].tolist()
            ranked_scores = scores[0].tolist()
        else:
            sims = (self._normalize(self.doc_vectors) @ query_vec.T).ravel()
            ranked_ids = np.argsort(sims)[::-1][:top_k].tolist()
            ranked_scores = sims[ranked_ids].tolist()

        snippets = [self.documents[i] for i in ranked_ids if i >= 0]
        return RagResult(snippets=snippets, scores=[float(s) for s in ranked_scores])

    def _documents_from_dataframe(self, df: pd.DataFrame) -> list[str]:
        docs = [f"Column {col} sample values: {df[col].dropna().head(3).tolist()}" for col in df.columns]
        docs.append(f"Dataset has {len(df)} rows and {df.shape[1]} columns")
        return docs

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms
