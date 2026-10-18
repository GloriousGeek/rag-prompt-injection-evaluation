"""Execute the paired three-condition experiment with resumable JSONL output."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

from .benchmark import load_jsonl
from .config import (
    DATA_DIR,
    MAX_TOKENS,
    MODEL_ID,
    MODEL_REVISION,
    RESULTS_DIR,
    SEED,
    TEMPERATURE,
    TOP_K,
    RAW_DIR,
)
from .prompting import build_messages
from .retrieval import retrieve
from .scoring import score_response

CONDITIONS = ("clean_baseline", "adversarial", "defended")


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def experiment_metadata(model_path: str) -> dict:
    return {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_path": str(Path(model_path).expanduser()),
        "quantization": {"bits": 4, "group_size": 64},
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "sampling": "greedy",
        "seed": SEED,
        "top_k_retrieval": TOP_K,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "mlx_version": _package_version("mlx"),
        "mlx_lm_version": _package_version("mlx-lm"),
        "sklearn_version": _package_version("scikit-learn"),
        "hf_home": os.environ.get("HF_HOME"),
        "hf_hub_cache": os.environ.get("HF_HUB_CACHE"),
        "hf_xet_cache": os.environ.get("HF_XET_CACHE"),
    }


class MLXGenerator:
    def __init__(self, model_path: str):
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.sample_utils import make_sampler

        mx.random.seed(SEED)
        self.model, self.tokenizer = load(str(Path(model_path).expanduser()))
        self.sampler = make_sampler(temp=TEMPERATURE)

    def __call__(self, messages: list[dict[str, str]]) -> str:
        from mlx_lm import generate

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=MAX_TOKENS,
            sampler=self.sampler,
            verbose=False,
        ).strip()


def _completed_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    return {
        (row["evaluation_id"], row["condition"])
        for row in load_jsonl(path)
    }


def run_experiment(
    model_path: str,
    output_dir: Path | None = None,
    limit: int | None = None,
    conditions: tuple[str, ...] = CONDITIONS,
    resume: bool = False,
) -> Path:
    unknown = set(conditions) - set(CONDITIONS)
    if unknown:
        raise ValueError(f"Unknown conditions: {sorted(unknown)}")
    rows = load_jsonl(DATA_DIR / "evaluation_set.jsonl")
    if limit is not None:
        rows = rows[:limit]

    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = RESULTS_DIR / "runs" / stamp
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw_outputs.jsonl"
    if raw_path.exists() and not resume:
        raise FileExistsError(f"Refusing to overwrite {raw_path}; pass --resume")

    metadata = experiment_metadata(model_path)
    config_path = output_dir / "experiment_config.json"
    if not config_path.exists():
        config_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    completed = _completed_keys(raw_path) if resume else set()
    generator = MLXGenerator(model_path)
    selected_by_document = {row["document_id"]: row for row in rows}
    source_corpus = load_jsonl(RAW_DIR / "table_test.jsonl")
    corpus = []
    for source_index, source in enumerate(source_corpus):
        document_id = f"bipia-table-test-doc-{source_index:03d}"
        selected = selected_by_document.get(document_id)
        corpus.append(
            {
                "document_id": document_id,
                "context": source["context"],
                "attack": selected["attack"] if selected else {"prompt": ""},
            }
        )

    with raw_path.open("a", encoding="utf-8") as handle:
        for example in rows:
            for condition in conditions:
                key = (example["evaluation_id"], condition)
                if key in completed:
                    continue
                poisoned = condition != "clean_baseline"
                documents, target_rank = retrieve(
                    example["question"],
                    corpus,
                    example["document_id"],
                    poisoned_target=poisoned,
                    top_k=TOP_K,
                )
                exposed = any(document.contains_attack for document in documents)
                messages = build_messages(condition, example["question"], documents)
                started = time.perf_counter()
                response = generator(messages)
                elapsed = time.perf_counter() - started
                score = score_response(response, example["ideal"], example["attack"], exposed)
                record = {
                    "evaluation_id": example["evaluation_id"],
                    "source_benchmark": example["source_benchmark"],
                    "source_revision": example["source_revision"],
                    "source_task": example["source_task"],
                    "source_split": example["source_split"],
                    "source_row_index": example["source_row_index"],
                    "attack_category": example["attack"]["category"],
                    "benchmark_attack_id": example["attack"]["benchmark_attack_id"],
                    "attack_prompt": example["attack"]["prompt"],
                    "attack_transform": example["attack"]["transform"],
                    "condition": condition,
                    "question": example["question"],
                    "expected_answer": example["ideal"],
                    "retrieved_document_ids": [document.document_id for document in documents],
                    "retrieval_scores": [round(document.score, 8) for document in documents],
                    "target_document_rank": target_rank,
                    "retrieval_relevant": any(document.is_relevant for document in documents),
                    "attack_exposed": exposed,
                    "model_response": response,
                    **score,
                    "scoring_version": "3.0-tristate-bounds",
                    "model_config": metadata,
                    "latency_seconds": round(elapsed, 4),
                    "executed_at": datetime.now(timezone.utc).isoformat(),
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                print(
                    f'{example["evaluation_id"]} {condition}: '
                    f'task={score["legitimate_task_success"]} '
                    f'attack={score["attack_success"]} exposed={exposed}'
                )
    return output_dir
