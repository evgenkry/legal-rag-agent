"""Форматирование чанков для промптов LLM (нумерация [i] и node_id для верификатора)"""


def format_nodes_for_context(nodes: list) -> str:
    """Собирает текст контекста: [i] node_id=... | цитата + тело чанка"""
    parts: list[str] = []
    for i, n in enumerate(nodes, 1):
        node = n.node
        text = node.get_content() if hasattr(node, "get_content") else str(node)
        meta = getattr(node, "metadata", {}) or {}
        cit = meta.get("full_citation", "")
        nid = getattr(node, "node_id", None) or getattr(node, "id_", "") or ""
        parts.append(f"[{i}] node_id={nid} | {cit}\n{text}")
    return "\n\n".join(parts)
