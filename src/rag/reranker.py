"""Reranker - cross-encoder"""

import logging
from typing import Optional

from llama_index.core.schema import NodeWithScore

logger = logging.getLogger(__name__)


class Reranker:
    """Cross-encoder reranker для переранжирования результатов"""

    def __init__(self, model_name: str = "qilowoq/bge-reranker-v2-m3-en-ru", top_n: int = 5):
        self._model_name = model_name
        self._top_n = top_n
        self._model = None
        self._tokenizer = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)
            logger.info("Reranker loaded: %s", self._model_name)
        except Exception as e:
            logger.warning("Reranker not available: %s", e)

    def rerank(
        self,
        query: str,
        nodes: list[NodeWithScore],
        top_n: Optional[int] = None,
    ) -> list[NodeWithScore]:
        """Переранжирует ноды по релевантности к запросу."""
        n = top_n or self._top_n
        if len(nodes) <= n:
            return nodes

        self._load_model()
        if self._model is None:
            return nodes[:n]

        try:
            pairs = [(query, node.node.get_content()) for node in nodes]
            scores = self._model.predict(pairs)

            scored = list(zip(nodes, scores))
            scored.sort(key=lambda x: x[1], reverse=True)
            result = []
            for nws, sc in scored[:n]:
                node = nws.node
                meta = dict(node.metadata or {})
                meta["rerank_score"] = float(sc)
                node.metadata = meta
                result.append(NodeWithScore(node=node, score=float(sc)))
            return result
        except Exception as e:
            logger.warning("Rerank failed: %s", e)
            return nodes[:n]
