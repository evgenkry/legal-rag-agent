from llama_index.core.schema import NodeWithScore, TextNode

from src.rag.retrieval_policy import apply_law_explanation_policy


def _mk(score: float, source: str, idx: str):
    node = TextNode(text=f"text-{idx}", metadata={"source": source})
    node.id_ = idx
    return NodeWithScore(node=node, score=score)


def test_retrieval_policy_boost_and_diversity():
    results = [
        _mk(0.8, "rostrud", "e1"),
        _mk(0.7, "rostrud", "e2"),
        _mk(0.6, "tkrf", "l1"),
        _mk(0.55, "tkrf", "l2"),
        _mk(0.5, "tkrf", "l3"),
    ]
    out = apply_law_explanation_policy(
        results,
        top_k=4,
        law_boost=0.2,
        law_min=2,
        explanation_min=1,
    )
    ids = [x.node.id_ for x in out]
    assert "l1" in ids and "l2" in ids
    assert any(i.startswith("e") for i in ids)
    assert len(out) == 4


def test_retrieval_policy_edge_case_only_explanations():
    results = [_mk(0.6, "rostrud", "e1"), _mk(0.4, "rostrud", "e2")]
    out = apply_law_explanation_policy(
        results,
        top_k=2,
        law_boost=0.2,
        law_min=2,
        explanation_min=2,
    )
    assert [x.node.id_ for x in out] == ["e1", "e2"]
