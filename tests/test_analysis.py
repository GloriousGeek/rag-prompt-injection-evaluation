import pandas as pd

from rag_injection_eval.analysis import compute_metrics, wilson_interval


def test_metrics_use_declared_denominators():
    frame = pd.DataFrame(
        [
            {"condition": "clean_baseline", "legitimate_task_success": True, "retrieval_relevant": True, "attack_exposed": False, "attack_success": False, "attack_classification": "not_applicable", "needs_manual_review": False, "attack_category": "x"},
            {"condition": "adversarial", "legitimate_task_success": False, "retrieval_relevant": True, "attack_exposed": True, "attack_success": True, "attack_classification": "confirmed_success", "needs_manual_review": False, "attack_category": "x"},
            {"condition": "defended", "legitimate_task_success": True, "retrieval_relevant": True, "attack_exposed": True, "attack_success": False, "attack_classification": "confirmed_failure", "needs_manual_review": False, "attack_category": "x"},
        ]
    )
    metrics = compute_metrics(frame)
    assert metrics["conditions"]["adversarial"]["conditional_attack_success"] == {
        "successes": 1,
        "n": 1,
        "rate": 1.0,
        "ci95_wilson": list(wilson_interval(1, 1)),
    }
    assert metrics["conditions"]["adversarial"]["end_to_end_attack_success_bounds"] == {
        "confirmed_successes": 1,
        "confirmed_failures": 0,
        "unresolved": 0,
        "n": 1,
        "lower_rate": 1.0,
        "upper_rate": 1.0,
    }
    assert "defence_effectiveness" not in metrics
    assert metrics["defended_lts_change_vs_clean"] == 0.0
