# Third-party notices

The root `LICENSE` applies only to this repository's original code and documentation. It does not
relicense third-party benchmark data, prompts, model files, or generated outputs.

## BIPIA and TableQA data

- Source: [microsoft/BIPIA](https://github.com/microsoft/BIPIA)
- Pinned revision: `a004b69ec0dd446e0afd461d98cb5e96e120a5d0`
- BIPIA code license: MIT
- TableQA provenance stated by BIPIA: WikiTableQuestions, CC BY-SA 4.0

The exact upstream source files, their SHA-256 hashes, and the deterministic adaptation are described
in `data/README.md` and `data/benchmark_manifest.json`. Users redistributing the data or adaptations
must comply with the applicable upstream attribution and share-alike terms.

## Model

- Model repository: [mlx-community/Qwen2.5-3B-Instruct-4bit](https://huggingface.co/mlx-community/Qwen2.5-3B-Instruct-4bit)
- Pinned revision: `4f83f8f146fdf28b512a06562b671d7af4fab457`
- Model-card license tag at the time of the study: `qwen-research`

No model weights are committed to this repository. Users download the snapshot directly and are
responsible for reviewing its current model card and license before use.

## Generated outputs

`results/reference/` contains model-generated responses derived from the benchmark inputs. They are
published for research auditability and are not covered by a claim that overrides upstream rights.
