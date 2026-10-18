"""Compute declared metrics and generate at most three figures."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .benchmark import load_jsonl


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if total == 0:
        return None, None
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((proportion * (1 - proportion) + z * z / (4 * total)) / total) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _rate(successes: int, total: int) -> dict:
    low, high = wilson_interval(successes, total)
    return {
        "successes": int(successes),
        "n": int(total),
        "rate": successes / total if total else None,
        "ci95_wilson": [low, high],
    }


def _attack_bounds(group: pd.DataFrame, denominator: int) -> dict:
    """Return identification bounds when unresolved judge cases remain.

    The lower bound counts only deterministic confirmed successes. The upper
    bound additionally treats every unresolved response as a success. These
    are not confidence intervals; they expose uncertainty from classification.
    """
    classifications = group["attack_classification"]
    confirmed_successes = int((classifications == "confirmed_success").sum())
    confirmed_failures = int((classifications == "confirmed_failure").sum())
    unresolved = int((classifications == "unresolved").sum())
    return {
        "confirmed_successes": confirmed_successes,
        "confirmed_failures": confirmed_failures,
        "unresolved": unresolved,
        "n": int(denominator),
        "lower_rate": confirmed_successes / denominator if denominator else None,
        "upper_rate": (confirmed_successes + unresolved) / denominator if denominator else None,
    }


def compute_metrics(frame: pd.DataFrame) -> dict:
    if "attack_classification" not in frame:
        frame = frame.copy()
        frame["attack_classification"] = frame.apply(
            lambda row: (
                "not_applicable"
                if not row["attack_exposed"]
                else "confirmed_success"
                if row["attack_success"]
                else "unresolved"
                if row["needs_manual_review"]
                else "confirmed_failure"
            ),
            axis=1,
        )
    metrics: dict = {"conditions": {}, "attack_categories": {}}
    for condition, group in frame.groupby("condition", sort=False):
        exposed = group[group["attack_exposed"]]
        classification_counts = {
            label: int((group["attack_classification"] == label).sum())
            for label in ("confirmed_success", "confirmed_failure", "unresolved", "not_applicable")
        }
        metrics["conditions"][condition] = {
            "legitimate_task_success": _rate(int(group["legitimate_task_success"].sum()), len(group)),
            "retrieval_relevance": _rate(int(group["retrieval_relevant"].sum()), len(group)),
            "attack_exposure": _rate(int(group["attack_exposed"].sum()), len(group)),
            "conditional_attack_success": _rate(int(exposed["attack_success"].sum()), len(exposed)),
            "end_to_end_attack_success": _rate(int(group["attack_success"].sum()), len(group)),
            "conditional_attack_success_bounds": _attack_bounds(exposed, len(exposed)),
            "end_to_end_attack_success_bounds": _attack_bounds(group, len(group)),
            "attack_classification_counts": classification_counts,
            "manual_review_flags": int(group["needs_manual_review"].sum()),
        }

    defended = metrics["conditions"].get("defended", {})
    baseline_lts = metrics["conditions"].get("clean_baseline", {}).get("legitimate_task_success", {}).get("rate")
    defended_lts = defended.get("legitimate_task_success", {}).get("rate")
    metrics["defended_lts_change_vs_clean"] = (
        None if baseline_lts is None or defended_lts is None else defended_lts - baseline_lts
    )

    attacked = frame[frame["condition"].isin(["adversarial", "defended"])]
    for (category, condition), group in attacked.groupby(["attack_category", "condition"]):
        exposed = group[group["attack_exposed"]]
        metrics["attack_categories"].setdefault(category, {})[condition] = {
            "conditional_attack_success": _rate(int(exposed["attack_success"].sum()), len(exposed)),
            "end_to_end_attack_success": _rate(int(group["attack_success"].sum()), len(group)),
            "conditional_attack_success_bounds": _attack_bounds(exposed, len(exposed)),
            "end_to_end_attack_success_bounds": _attack_bounds(group, len(group)),
        }
    return metrics


def _bar_chart(labels: list[str], values: list[float], ns: list[int], ylabel: str, title: str, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 4.2))
    bars = axis.bar(labels, values, color=["#5B8FF9", "#E8684A", "#5AD8A6"][: len(labels)])
    axis.set_ylim(0, 1)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    for bar, value, n in zip(bars, values, ns):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.1%}\n(n={n})", ha="center", va="bottom")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _bounds_chart(labels: list[str], lower: list[float], upper: list[float], ns: list[int], title: str, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 4.2))
    bars = axis.bar(labels, lower, color=["#E8684A", "#5AD8A6"][: len(labels)], label="confirmed lower bound")
    axis.scatter(labels, upper, marker="_", s=450, linewidths=3, color="#1F2D3D", label="possible upper bound")
    for bar, low, high, n in zip(bars, lower, upper, ns):
        axis.vlines(bar.get_x() + bar.get_width() / 2, low, high, color="#1F2D3D", linewidth=1.5)
        axis.text(bar.get_x() + bar.get_width() / 2, min(0.96, high + 0.035), f"{low:.1%}–{high:.1%}\n(n={n})", ha="center", va="bottom")
    axis.set_ylim(0, 1.08)
    axis.set_ylabel("End-to-end attack success bounds")
    axis.set_title(title)
    axis.legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def analyze(raw_path: Path, output_dir: Path | None = None) -> dict:
    rows = load_jsonl(raw_path)
    frame = pd.DataFrame(rows)
    output_dir = output_dir or raw_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = compute_metrics(frame)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    frame.to_csv(output_dir / "results.csv", index=False)

    condition_order = [item for item in ("clean_baseline", "adversarial", "defended") if item in metrics["conditions"]]
    _bar_chart(
        condition_order,
        [metrics["conditions"][item]["legitimate_task_success"]["rate"] for item in condition_order],
        [metrics["conditions"][item]["legitimate_task_success"]["n"] for item in condition_order],
        "Legitimate task success rate",
        "Task success by condition",
        output_dir / "legitimate_task_success.png",
    )
    attack_order = [item for item in ("adversarial", "defended") if item in metrics["conditions"]]
    _bounds_chart(
        attack_order,
        [metrics["conditions"][item]["end_to_end_attack_success_bounds"]["lower_rate"] for item in attack_order],
        [metrics["conditions"][item]["end_to_end_attack_success_bounds"]["upper_rate"] for item in attack_order],
        [metrics["conditions"][item]["end_to_end_attack_success_bounds"]["n"] for item in attack_order],
        "Attack success identification bounds",
        output_dir / "attack_success.png",
    )

    categories = sorted(metrics["attack_categories"])
    x_positions = list(range(len(categories)))
    width = 0.36
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for offset, condition, color in ((-width / 2, "adversarial", "#E8684A"), (width / 2, "defended", "#5AD8A6")):
        bounds = [metrics["attack_categories"][category][condition]["end_to_end_attack_success_bounds"] for category in categories]
        lower = [item["lower_rate"] for item in bounds]
        upper = [item["upper_rate"] for item in bounds]
        bars = axis.bar([position + offset for position in x_positions], lower, width, label=f"{condition}: confirmed", color=color)
        centres = [bar.get_x() + bar.get_width() / 2 for bar in bars]
        axis.vlines(centres, lower, upper, color="#1F2D3D", linewidth=1.2)
        axis.scatter(centres, upper, marker="_", s=180, linewidths=2, color="#1F2D3D")
    axis.set_xticks(x_positions, categories)
    axis.set_ylim(0, 1)
    axis.set_ylabel("End-to-end attack success bounds")
    axis.set_title("Attack success bounds by BIPIA attack category")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "attack_success_by_category.png", dpi=180)
    plt.close(figure)
    return metrics
