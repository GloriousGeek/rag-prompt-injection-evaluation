"""Central experiment constants kept explicit for auditability."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "bipia"
RESULTS_DIR = PROJECT_ROOT / "results"
REPORTS_DIR = PROJECT_ROOT / "reports"

BIPIA_REPOSITORY = "https://github.com/microsoft/BIPIA"
BIPIA_REVISION = "a004b69ec0dd446e0afd461d98cb5e96e120a5d0"
BIPIA_TABLE_URL = (
    "https://raw.githubusercontent.com/microsoft/BIPIA/"
    f"{BIPIA_REVISION}/benchmark/table/test.jsonl"
)
BIPIA_ATTACK_URL = (
    "https://raw.githubusercontent.com/microsoft/BIPIA/"
    f"{BIPIA_REVISION}/benchmark/text_attack_test.json"
)
BIPIA_TABLE_SHA256 = "3d5eaab192b80bada762e3cd7655583a14e5888ba8b845a01f0434e4ccc4291e"
BIPIA_ATTACK_SHA256 = "75750e7b4e8b34e8f9d88d89b357aeaaf02bd07f9e493ccd37eda74a0cd7c7f8"

MODEL_ID = "mlx-community/Qwen2.5-3B-Instruct-4bit"
MODEL_REVISION = "4f83f8f146fdf28b512a06562b671d7af4fab457"
MODEL_WEIGHT_BYTES = 1_736_293_090
SEED = 2025
SAMPLE_SIZE = 40
TOP_K = 3
MAX_TOKENS = 96
TEMPERATURE = 0.0
