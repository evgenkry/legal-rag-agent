"""Unit tests для RRF fusion"""

import pytest

from src.rag.rrf import FusedResult, rrf_fuse


def test_rrf_fuse_correct_sum_and_dedup():
    """Правильная RRF-сумма, дедуп по chunk_id, стабильная сортировка."""
    sparse = [
        ("a", "text a", {}, 0.9),
        ("b", "text b", {}, 0.8),
        ("c", "text c", {}, 0.7),
    ]
    dense = [
        ("b", "text b", {}, 0.95),  # b в обоих
        ("a", "text a", {}, 0.85),
        ("d", "text d", {}, 0.75),  # только dense
    ]
    fused = rrf_fuse(sparse, dense, k=60, top_n=5)
    assert len(fused) == 4
    ids = [f.chunk_id for f in fused]
    assert "a" in ids
    assert "b" in ids
    assert "c" in ids
    assert "d" in ids
    # b и a — в обоих, должны быть выше c и d
    assert ids[0] in ("a", "b")
    assert ids[1] in ("a", "b")
    # b: sparse rank 2, dense rank 1 -> 1/62 + 1/61
    b = next(f for f in fused if f.chunk_id == "b")
    expected_b = 1 / (60 + 2) + 1 / (60 + 1)  # sparse rank 2 + dense rank 1
    assert abs(b.fused_score - expected_b) < 1e-6
    assert b.sparse_score == pytest.approx(0.8)
    assert b.dense_score == pytest.approx(0.95)
    c = next(f for f in fused if f.chunk_id == "c")
    assert c.provenance == "sparse"
    d = next(f for f in fused if f.chunk_id == "d")
    assert d.provenance == "dense"


def test_rrf_fuse_provenance():
    """Merge provenance: both, sparse-only, dense-only."""
    sparse = [("s1", "t1", {}, 1.0), ("both", "tb", {}, 0.9)]
    dense = [("both", "tb", {}, 1.0), ("d1", "td", {}, 0.9)]
    fused = rrf_fuse(sparse, dense, k=60, top_n=10)
    by_id = {f.chunk_id: f for f in fused}
    assert by_id["both"].provenance == "both"
    assert by_id["s1"].provenance == "sparse"
    assert by_id["d1"].provenance == "dense"


def test_rrf_fuse_top_n():
    """top_n ограничивает результат."""
    sparse = [(f"s{i}", "t", {}, 1.0) for i in range(10)]
    dense = [(f"d{i}", "t", {}, 1.0) for i in range(10)]
    fused = rrf_fuse(sparse, dense, k=60, top_n=3)
    assert len(fused) == 3


def test_rrf_fuse_empty_inputs():
    """Пустые списки."""
    fused = rrf_fuse([], [], k=60, top_n=5)
    assert fused == []
    fused = rrf_fuse([("a", "t", {}, 1.0)], [], k=60, top_n=5)
    assert len(fused) == 1
    assert fused[0].chunk_id == "a"
    assert fused[0].provenance == "sparse"
    assert fused[0].sparse_score == pytest.approx(1.0)
    assert fused[0].dense_score is None
