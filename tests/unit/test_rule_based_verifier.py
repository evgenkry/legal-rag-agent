from llama_index.core.schema import NodeWithScore, TextNode

from src.agent.rule_based_verifier import RuleBasedVerifier


def _docs():
    node = TextNode(
        text="Срок окончательного расчета при увольнении определен в ст. 140 ТК РФ.",
        metadata={"source": "tkrf"},
    )
    node.id_ = "1"
    return [NodeWithScore(node=node, score=0.9)]


def test_rule_based_verifier_pass():
    v = RuleBasedVerifier()
    r = v.verify("По ст. 140 ТК РФ расчет в день увольнения.", _docs())
    assert r.passed is True


def test_rule_based_verifier_reject_without_sources():
    v = RuleBasedVerifier()
    r = v.verify("Расчет делается в день увольнения.", _docs())
    assert r.passed is False
    assert r.reason == "rule_no_sources"
