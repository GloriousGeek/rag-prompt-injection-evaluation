"""Cross-check raw records, pairing, and derived metrics."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from .analysis import compute_metrics
from .benchmark import load_jsonl, sha256, validate_attacks
from .config import (
    BIPIA_ATTACK_SHA256,
    BIPIA_REVISION,
    BIPIA_TABLE_SHA256,
    DATA_DIR,
    MODEL_ID,
    MODEL_REVISION,
    RAW_DIR,
    TOP_K,
)
from .runner import CONDITIONS


def validate_results(run_dir: Path) -> list[str]:
    raw_path = run_dir / "raw_outputs.jsonl"
    metrics_path = run_dir / "metrics.json"
    rows = load_jsonl(raw_path)
    if not rows:
        raise ValueError("Raw output file is empty")
    frame = pd.DataFrame(rows)
    errors: list[str] = []

    table_path = RAW_DIR / "table_test.jsonl"
    attack_path = RAW_DIR / "text_attack_test.json"
    evaluation_path = DATA_DIR / "evaluation_set.jsonl"
    if sha256(table_path) != BIPIA_TABLE_SHA256:
        errors.append("BIPIA table source checksum mismatch")
    if sha256(attack_path) != BIPIA_ATTACK_SHA256:
        errors.append("BIPIA attack source checksum mismatch")
    try:
        validate_attacks(attack_path)
    except ValueError as error:
        errors.append(str(error))

    source_rows = load_jsonl(table_path)
    evaluation_rows = load_jsonl(evaluation_path)
    evaluation_by_id = {row["evaluation_id"]: row for row in evaluation_rows}
    if len(evaluation_by_id) != len(evaluation_rows):
        errors.append("duplicate evaluation IDs in evaluation_set.jsonl")
    for example in evaluation_rows:
        source = source_rows[example["source_row_index"]]
        for field in ("question", "context", "ideal"):
            if example[field] != source[field]:
                errors.append(f"source mismatch for {example['evaluation_id']} field {field}")
        if example["source_revision"] != BIPIA_REVISION:
            errors.append(f"source revision mismatch for {example['evaluation_id']}")

    duplicates = frame.duplicated(["evaluation_id", "condition"])
    if duplicates.any():
        errors.append(f"{int(duplicates.sum())} duplicate evaluation-condition rows")
    expected_conditions = set(CONDITIONS)
    for evaluation_id, group in frame.groupby("evaluation_id"):
        if set(group["condition"]) != expected_conditions:
            errors.append(f"Incomplete condition pairing for {evaluation_id}")
            continue
        if evaluation_id not in evaluation_by_id:
            errors.append(f"unknown evaluation ID: {evaluation_id}")
            continue
        example = evaluation_by_id[evaluation_id]
        for _, row in group.iterrows():
            expected_fields = {
                "source_revision": example["source_revision"],
                "source_row_index": example["source_row_index"],
                "question": example["question"],
                "expected_answer": example["ideal"],
                "attack_category": example["attack"]["category"],
                "benchmark_attack_id": example["attack"]["benchmark_attack_id"],
                "attack_prompt": example["attack"]["prompt"],
                "attack_transform": example["attack"]["transform"],
            }
            for field, expected in expected_fields.items():
                if row[field] != expected:
                    errors.append(f"record mismatch for {evaluation_id}/{row['condition']} field {field}")
            if not isinstance(row["model_response"], str) or not row["model_response"].strip():
                errors.append(f"empty model response for {evaluation_id}/{row['condition']}")
            if float(row["latency_seconds"]) <= 0:
                errors.append(f"non-positive latency for {evaluation_id}/{row['condition']}")
            try:
                datetime.fromisoformat(row["executed_at"])
            except (TypeError, ValueError):
                errors.append(f"invalid timestamp for {evaluation_id}/{row['condition']}")
            if len(row["retrieved_document_ids"]) != TOP_K or len(row["retrieval_scores"]) != TOP_K:
                errors.append(f"retrieval width mismatch for {evaluation_id}/{row['condition']}")
            model_config = row["model_config"]
            if model_config["model_id"] != MODEL_ID or model_config["model_revision"] != MODEL_REVISION:
                errors.append(f"model identity mismatch for {evaluation_id}/{row['condition']}")

        indexed = group.set_index("condition")
        adversarial = indexed.loc["adversarial"]
        defended = indexed.loc["defended"]
        if adversarial["retrieved_document_ids"] != defended["retrieved_document_ids"]:
            errors.append(f"attacked retrieval IDs differ for {evaluation_id}")
        if adversarial["retrieval_scores"] != defended["retrieval_scores"]:
            errors.append(f"attacked retrieval scores differ for {evaluation_id}")
        if bool(indexed.loc["clean_baseline", "attack_exposed"]):
            errors.append(f"clean attack exposure for {evaluation_id}")
        if not bool(adversarial["attack_exposed"]) or not bool(defended["attack_exposed"]):
            errors.append(f"missing attacked exposure for {evaluation_id}")

    if frame["executed_at"].duplicated().any():
        errors.append("duplicate execution timestamps")
    model_configs = {json.dumps(config, sort_keys=True) for config in frame["model_config"]}
    if len(model_configs) != 1:
        errors.append("model configuration changed within the run")
    config_path = run_dir / "experiment_config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != rows[0]["model_config"]:
        errors.append("experiment_config.json does not match raw records")
    computed = compute_metrics(frame)
    saved = json.loads(metrics_path.read_text(encoding="utf-8"))
    if computed != saved:
        errors.append("metrics.json does not match recomputed metrics")
    if errors:
        raise ValueError("; ".join(errors))
    return [
        f"validated {len(frame)} raw outputs",
        f"validated {frame['evaluation_id'].nunique()} paired examples",
        "validated pinned BIPIA checksums and source-field provenance",
        "validated paired retrieval, exposure, model identity, latency, and timestamps",
        "metrics match raw outputs",
    ]
