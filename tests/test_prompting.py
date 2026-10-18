from types import SimpleNamespace

from rag_injection_eval.prompting import build_messages


def test_defence_adds_trust_boundary_without_changing_user_content():
    documents = [SimpleNamespace(document_id="d1", text="Evidence")]
    adversarial = build_messages("adversarial", "Question?", documents)
    defended = build_messages("defended", "Question?", documents)
    assert adversarial[1] == defended[1]
    assert "lower-trust evidence" not in adversarial[0]["content"]
    assert "lower-trust evidence" in defended[0]["content"]
