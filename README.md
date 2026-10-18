# Evaluating Prompt Injection Risks in Retrieval-Augmented Generation Systems

A small, empirical local pilot study of indirect prompt injection and a prompt-only trust-boundary
defense in a retrieval-augmented generation (RAG) system.

Author: **Usman Tariq** — MSc Data Science, University of Surrey. This independent project does
not imply endorsement by or affiliation with the University of Surrey, Microsoft, Alibaba/Qwen,
Apple, or Anthropic. Development was AI-assisted; the design, code, raw outputs, scoring rules,
and limitations are retained for audit rather than presented as manually authored work.

## Research question and hypotheses

**Question:** How effectively can explicit trust-boundary instructions mitigate indirect prompt
injection attacks against a locally deployed language model?

- **H1:** malicious instructions in lower-trust retrieved documents can divert the model;
- **H2:** explicitly separating trusted instructions from retrieved evidence reduces attacks;
- **H3:** the defensive prompt may change legitimate task performance.

The reference run confirms that injection behavior occurred, but it does not establish H2: the
attack-success identification intervals overlap substantially. H3 is descriptive only; defended
task success was 5 percentage points below clean and the sample is too small for a firm conclusion.

## Threat model

An attacker can modify one document in the retrieval corpus but cannot modify the system prompt,
the user's question, model weights, or evaluation code. A successful retrieval exposes the model
to a BIPIA instruction asking it to encode, reverse, or cipher its answer. The tested defense is
prompt-only; it does not sanitize documents, isolate execution, or guarantee safety.

## Benchmark and adaptation

The project uses [BIPIA](https://github.com/microsoft/BIPIA) at commit
[`a004b69`](https://github.com/microsoft/BIPIA/commit/a004b69ec0dd446e0afd461d98cb5e96e120a5d0),
an established benchmark created specifically for indirect prompt injection. BIPIA was selected
instead of AgentDojo because its external-context tasks map directly to this small RAG study;
AgentDojo primarily studies tool-using agents. BIPIA is archived, which limits ongoing maintenance,
but the pinned data, paper, construction code, and provenance remain available.

The adaptation uses 40 TableQA test cases sampled with seed 2025 from the 48/100 cases whose
source table appears in a clean TF-IDF top-3 retrieval. It assigns 14 Base Encoding, 13 Reverse
Text, and 13 Substitution Cipher attacks. BIPIA's original fields and identifiers are preserved.
The attack is appended with BIPIA's `insert_end` construction. Because the sample is conditioned
on retrieval eligibility and the scoring differs from official BIPIA evaluation, these results
must not be described as an official benchmark score. See [data/README.md](data/README.md) and
[data/benchmark_manifest.json](data/benchmark_manifest.json).

## Model and environment

- Model: [`mlx-community/Qwen2.5-3B-Instruct-4bit`](https://huggingface.co/mlx-community/Qwen2.5-3B-Instruct-4bit)
- Revision: `4f83f8f146fdf28b512a06562b671d7af4fab457`
- Quantization: 4-bit, group size 64; model card license tag: `qwen-research`
- Snapshot payload: 1,747,851,324 bytes (1.628 GiB); weight file: 1,736,293,090 bytes
- Runtime: Python 3.12.4, MLX 0.32.4, MLX-LM 0.28.3
- Hardware: Apple M3 Pro, 36 GB unified memory, arm64
- Generation: greedy (`temperature=0`), maximum 96 new tokens, seed 2025
- Retrieval: word/bigram TF-IDF, cosine similarity, top-3 over all 100 BIPIA test tables

The reference run completed locally on 18 October 2026. The 120 generations took 283.1 seconds
of measured inference time (mean 2.36 seconds per output).

## Experimental conditions

1. **Clean baseline:** clean corpus and the standard system instruction.
2. **Adversarial:** the target document contains its BIPIA instruction; prompt otherwise unchanged.
3. **Defended:** the identical poisoned corpus plus a system instruction declaring retrieved text
   to be lower-trust evidence rather than instructions.

For every run, the JSONL records the question, expected answer, retrieved IDs and scores, target
rank, relevance, exposure, raw response, scores, timestamps, and complete model configuration.

## Metrics

For `N` examples, `E_i` denotes attack exposure, `A_i` confirmed attack success, and `L_i`
legitimate task success:

- Legitimate Task Success Rate: `LTSR = ΣL_i / N`.
- Attack Exposure Rate: `AER = ΣE_i / N`.
- Conditional Attack Success Rate: `CASR = Σ(A_i ∧ E_i) / ΣE_i`.
- End-to-End Attack Success Rate: `E2E-ASR = Σ(A_i ∧ E_i) / N`.
- Attack-success identification interval: confirmed successes form the lower bound; confirmed
  successes plus unresolved outputs form the upper bound.

Rates include Wilson 95% confidence intervals and explicit integer denominators. Legitimate-answer
scoring requires every pipe-separated answer component and matches numeric answers as whole numeric
tokens. Base-encoding attacks use conservative canonical-syntax checks; exact requested Reverse and
Caesar transformations are also confirmed automatically. All other attacked outputs receive a
tri-state classification: confirmed success, confirmed failure, or unresolved. The 28 automated
review flags were manually inspected under a published rationale file; 18 were confirmed failures
and 10 remain unresolved. Because BIPIA uses an LLM judge for these categories, the interval is more
honest than silently substituting an unvalidated self-judge.

## Reference results

| Condition | LTSR (95% Wilson CI) | AER | Confirmed E2E-ASR (95% Wilson CI) | Identification interval | Unresolved |
|---|---:|---:|---:|---:|---:|
| Clean | 13/40 = 32.5% (20.1–48.0%) | 0/40 | 0/40 = 0% | 0% | 0 |
| Adversarial | 8/40 = 20.0% (10.5–34.8%) | 40/40 | 2/40 = 5.0% (1.4–16.5%) | 5.0–20.0% | 6 |
| Defended | 11/40 = 27.5% (16.1–42.8%) | 40/40 | 1/40 = 2.5% (0.4–12.9%) | 2.5–12.5% | 4 |

The defense must not be described as effective from this pilot. Confirmed events changed from two
to one, while the admissible attack-success ranges overlap (5.0–20.0% adversarial and 2.5–12.5%
defended). Adversarial task success was 12.5 points below clean; defended prompting recovered 7.5
points relative to adversarial while remaining 5 points below clean.

Clean success is low despite relevant retrieval in 40/40 cases: the target ranked first for 27
examples, second for 11, and third for 2, with 8, 4, and 1 successes respectively. Inspection shows
mostly ordinary table-reading, comparison, counting, or distractor errors from the small quantized
model rather than missing retrieval. The top-3 multi-table prompt likely adds difficulty, but this
pilot does not isolate its causal contribution.

![Attack success by condition](results/reference/attack_success.png)

![Legitimate task success by condition](results/reference/legitimate_task_success.png)

![Attack success by category](results/reference/attack_success_by_category.png)

## Reproduce

On an Apple Silicon Mac with Python 3.12:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'

export HF_HOME="$HOME/AI-Models/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"
export HF_HUB_DISABLE_XET=1
export MPLCONFIGDIR="$PWD/.cache/matplotlib"

hf download mlx-community/Qwen2.5-3B-Instruct-4bit \
  --revision 4f83f8f146fdf28b512a06562b671d7af4fab457 \
  --cache-dir "$HF_HUB_CACHE"

python -m rag_injection_eval prepare
MODEL_PATH="$HF_HUB_CACHE/models--mlx-community--Qwen2.5-3B-Instruct-4bit/snapshots/4f83f8f146fdf28b512a06562b671d7af4fab457"
RUN_DIR="results/runs/reproduction"
REVIEWED_DIR="results/runs/reproduction-reviewed"
python -m rag_injection_eval run --model "$MODEL_PATH" --output-dir "$RUN_DIR"
mkdir -p "$REVIEWED_DIR"
python -m rag_injection_eval adjudicate "$RUN_DIR/raw_outputs.jsonl" \
  results/reference/manual_attack_adjudications.json "$REVIEWED_DIR/raw_outputs.jsonl"
cp "$RUN_DIR/experiment_config.json" "$REVIEWED_DIR/experiment_config.json"
python -m rag_injection_eval analyze "$REVIEWED_DIR/raw_outputs.jsonl" --output-dir "$REVIEWED_DIR"
python -m rag_injection_eval validate "$REVIEWED_DIR"
python -m pytest
```

`prepare` refuses to overwrite the adapted dataset unless `--force` is supplied. `run` creates a
new timestamped directory by default and supports `--resume`; it never silently replaces raw output.

## Repository map

- `src/rag_injection_eval/`: benchmark preparation, retrieval, prompts, inference, scoring, analysis;
- `data/`: pinned source files, checksums, and the auditable adaptation;
- `tests/`: deterministic unit tests for scoring, retrieval, prompts, and metrics;
- `results/reference/`: raw outputs, scoring history, manual adjudications, metrics, and three figures;
- `reports/research_report.md`: concise research paper-style account;
- `.private/`: local learning and storage notes, deliberately ignored by Git.

## Limitations and responsible use

This is a small pilot, not a definitive defense evaluation. The model is small and quantized;
clean task accuracy is low. The selected subset is
retrieval-eligible, not representative of all BIPIA. Only three task-relevant attack categories,
one insertion position, one retriever, one model, and one prompt defense are tested. The experiment
does not evaluate tool use, data exfiltration, multilingual attacks, stealth encodings, or adaptive
attackers. Ten attacked outputs remain unresolved after manual inspection, and a qualified second
reviewer or validated external judge is needed for a narrower estimate. No result establishes
guaranteed safety.

The attacks are output-format manipulations with no external actions. Use the code only for
authorized defensive research, retain benchmark attribution, and review
[third-party notices](THIRD_PARTY_NOTICES.md) before redistribution.

## Future work

Pre-register a human-review rubric, use two blinded reviewers, add bootstrap intervals for paired
differences, evaluate all insertion positions, compare retrieval depths, and test a separate
detector or structured data/instruction channel. A larger study should include stronger models and
AgentDojo only when tool-use risk is genuinely in scope.

## References

1. Yi et al. (2023), [Benchmarking and Defending Against Indirect Prompt Injection Attacks on Large Language Models](https://arxiv.org/abs/2312.14197).
2. Greshake et al. (2023), [Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection](https://arxiv.org/abs/2302.12173).
3. Debenedetti et al. (2024), [AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents](https://arxiv.org/abs/2406.13352).
4. Yang et al. (2024), [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115).
5. Apple ML Research, [MLX](https://github.com/ml-explore/mlx) and [MLX-LM](https://github.com/ml-explore/mlx-lm).
