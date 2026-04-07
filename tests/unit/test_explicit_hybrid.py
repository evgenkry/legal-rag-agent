"""Unit tests для ExplicitHybridRetriever и fallback"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agent.schemas import AnswerVerifierResult
from src.rag.explicit_hybrid_retriever import ExplicitHybridRetriever, _fused_to_nodes
from src.rag.rrf import FusedResult


@pytest.fixture
def mock_embed():
    m = MagicMock()
    m.get_query_embedding.return_value = [0.1] * 1024
    return m


@pytest.mark.asyncio
async def test_explicit_hybrid_raises_on_sparse_error(mock_embed):
    """Explicit retriever пробрасывает исключение при ошибке sparse — pipeline делает fallback."""
    from src.rag.explicit_hybrid_retriever import ExplicitHybridRetriever

    retriever = ExplicitHybridRetriever(embed_model=mock_embed)
    with patch("src.rag.explicit_hybrid_retriever.sparse_retrieve", new_callable=AsyncMock) as mock_sparse:
        with patch("src.rag.explicit_hybrid_retriever.dense_retrieve", new_callable=AsyncMock) as mock_dense:
            mock_sparse.side_effect = RuntimeError("FTS index missing")
            mock_dense.return_value = []

            with pytest.raises(RuntimeError):
                await retriever.retrieve_async("test query")

    mock_sparse.assert_called_once()


@pytest.mark.asyncio
async def test_pipeline_fallback_on_explicit_error(mock_embed):
    """Pipeline: при ошибке explicit retriever вызывается fallback."""
    with patch("src.rag.retriever.get_retriever") as mock_get_retriever:
        from src.rag.explicit_hybrid_retriever import ExplicitHybridRetriever
        from src.rag.pipeline import RAGPipeline
        from src.rag.retriever import HybridRetriever

        with patch.object(ExplicitHybridRetriever, "retrieve_async", new_callable=AsyncMock) as mock_explicit:
            mock_explicit.side_effect = RuntimeError("DB error")
            explicit = ExplicitHybridRetriever(embed_model=mock_embed)
            mock_get_retriever.return_value = explicit

            mock_llm = MagicMock()
            pipeline = RAGPipeline(embed_model=mock_embed, llm=mock_llm, retriever=explicit)

            with patch.object(HybridRetriever, "retrieve") as mock_fallback:
                from llama_index.core.schema import NodeWithScore, TextNode
                n = TextNode(text="x", metadata={})
                n.id_ = "n1"
                mock_fallback.return_value = [NodeWithScore(node=n, score=0.5)]

                with patch.object(pipeline._query_rewriter, "rewrite", new_callable=AsyncMock, return_value="q"):
                    with patch.object(pipeline._reranker, "rerank", return_value=mock_fallback.return_value):
                        with patch.object(pipeline._context_builder, "expand", return_value=mock_fallback.return_value):
                            with patch.object(pipeline._generator, "generate", new_callable=AsyncMock, return_value=("answer", [])):
                                with patch.object(
                                    pipeline._answer_verifier,
                                    "verify",
                                    new_callable=AsyncMock,
                                    return_value=(
                                        AnswerVerifierResult(
                                            claims=[],
                                            revised_answer_body="answer",
                                            substantive=True,
                                        ),
                                        "answer",
                                    ),
                                ):
                                    result = await pipeline.query("test", use_rewrite=False, trace_id="t1")

                mock_fallback.assert_called_once()
                assert result.get("stage_timings", {}).get("fallback_used") is True
                assert "answer" in result


def test_fused_to_nodes():
    """Преобразование FusedResult в NodeWithScore."""
    fused = [
        FusedResult(
            chunk_id="c1",
            fused_score=0.5,
            provenance="both",
            text="t1",
            metadata={"a": 1},
            sparse_score=0.2,
            dense_score=0.3,
        ),
    ]
    nodes = _fused_to_nodes(fused)
    assert len(nodes) == 1
    assert nodes[0].node.get_content() == "t1"
    assert nodes[0].node.id_ == "c1"
    meta = nodes[0].node.metadata
    assert meta["a"] == 1
    assert meta["retrieval_sparse_score"] == 0.2
    assert meta["retrieval_dense_score"] == 0.3
    assert meta["retrieval_rrf_score"] == 0.5
    assert nodes[0].score == 0.5
