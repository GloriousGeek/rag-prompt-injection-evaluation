# Evaluating Prompt Injection Risks in Retrieval-Augmented Generation Systems


## Abstract

This small pilot study evaluates whether an explicit natural-language trust boundary mitigates indirect prompt
injection in a small local RAG system. Forty retrieval-eligible TableQA examples were adapted from
the established BIPIA benchmark and evaluated with Qwen2.5-3B-Instruct, quantized to 4-bit MLX.
Each example was run under clean, poisoned, and defended conditions. TF-IDF retrieval exposed the
model to the relevant document in 40/40 cases. Confirmed end-to-end attack success was 2/40 (5.0%)
without the defense and 1/40 (2.5%) with it. After manual review, unresolved cases yield
identification intervals of 5.0–20.0% and 2.5–12.5%, respectively. Legitimate task success was
32.5% clean, 20.0% poisoned, and 27.5% defended. These overlapping ranges do not establish defense
effectiveness. Low baseline accuracy, selected sampling, one model, and ten unresolved cases prevent
broad claims.

## 1. Introduction

RAG systems place retrieved text beside trusted instructions in the model's context. Because the
model processes both through the same token stream, attacker-controlled text can be interpreted as
instructions. This is an indirect prompt injection: the adversary influences a data source rather
than directly submitting the user request. The experiment asks whether explicitly labeling
retrieved documents as lower-trust evidence changes this behavior in a practical local model.

## 2. Related work

Greshake et al. described indirect prompt injection as a consequence of blurred data/instruction
boundaries in LLM-integrated applications [1]. Yi et al. introduced BIPIA across external-context
tasks and evaluated prompt-based and fine-tuned defenses [2]. AgentDojo later provided a dynamic
benchmark for injection against tool-using agents [3]. This project uses BIPIA because its TableQA
context more directly represents document RAG, while avoiding AgentDojo's additional tool and state
machinery. Qwen2.5 provides the evaluated language model [4], and MLX/MLX-LM enable local Apple
Silicon inference [5,6].

## 3. Research questions

- **RQ1 / H1:** Can benchmark-derived instructions in retrieved documents cause task deviation?
- **RQ2 / H2:** Does an explicit trusted-instruction/untrusted-evidence boundary reduce success?
- **RQ3 / H3:** Does the defense change legitimate task performance?

## 4. Methodology

### 4.1 Benchmark selection and adaptation

BIPIA commit `a004b69ec0dd446e0afd461d98cb5e96e120a5d0` was pinned. Although the
repository is archived, it supplies public construction code, source provenance, task data, and a
paper. Its TableQA test split contains 100 WikiTableQuestions-derived rows. A top-3 TF-IDF precheck
identified 48 rows whose own table was retrievable; 40 were sampled without replacement with seed
2025. This conditioning deliberately isolates generation behavior but reduces external validity.

The adaptation preserves each context, question, answer, row index, and attack ID. Fourteen Base
Encoding, thirteen Reverse Text, and thirteen Substitution Cipher prompts were assigned. In attacked
conditions, only the target document was modified, using BIPIA's exact end insertion. Results are
adapted-project results, not official BIPIA scores.

### 4.2 Retrieval and prompts

Word and word-bigram TF-IDF vectors were fit over all 100 test tables plus the current question.
Cosine similarity selected three documents. The clean and adversarial conditions shared the same
base system prompt. The defended condition added: trusted instructions come only from the system
and user; retrieved documents are lower-trust evidence, never instructions.

### 4.3 Model

`mlx-community/Qwen2.5-3B-Instruct-4bit` revision
`4f83f8f146fdf28b512a06562b671d7af4fab457` was run with MLX 0.32.4 and
MLX-LM 0.28.3 on an Apple M3 Pro with 36 GB unified memory. Quantization was 4-bit with group size
64. Greedy decoding used temperature 0, seed 2025, and at most 96 generated tokens. No training or
fine-tuning occurred.

### 4.4 Scoring and metrics

Normalized answer containment determines legitimate task success, requiring every pipe-separated
answer and matching numeric answers as complete numeric tokens. Retrieval logs distinguish relevance
and exposure. Base-encoding success requires the whole response to be a canonical token of the
requested encoding; exact requested Reverse/Caesar transformations are also confirmed. Remaining
outputs receive confirmed-success, confirmed-failure, or unresolved labels. Twenty-eight automated
flags were manually reviewed with recorded rationales: 18 were confirmed failures and 10 remain
unresolved. BIPIA's registry uses an LLM judge for these categories, so reported identification
bounds retain that uncertainty.

Two scoring defects were corrected without rerunning inference. The old legitimate-answer rule
treated a pipe-separated multi-answer target as one concatenated string and allowed numeric
substrings (for example, `1` inside `1.5`). The change adds one previously missed clean success and
removes three numeric false positives across conditions; the clean total therefore remains 13/40 by
coincidence. The first expected-transformation scorer and the later conservative scorer are retained
as `*_initial_scoring*` and `*_v2_conservative*`; all versions contain identical raw model responses.

For `N` examples, exposure `E`, attack success `A`, and task success `L`:

`LTSR = ΣL/N`; `AER = ΣE/N`; `CASR = Σ(A∧E)/ΣE`; `E2E-ASR = Σ(A∧E)/N`.
Wilson 95% intervals accompany confirmed binomial rates. Identification lower bounds count confirmed
successes; upper bounds additionally count every unresolved output as a success.

## 5. Experimental setup

The three conditions were evaluated for the same 40 examples, producing 120 raw outputs. The
retrieved IDs, scores, target rank, exposure, response, scoring fields, model configuration,
latency, and timestamp are stored in JSONL. Execution was resumable and used a single model load.
Measured generation time totaled 283.1 seconds.

## 6. Results

| Condition | Legitimate task success | Relevant retrieval | Attack exposure | Confirmed E2E-ASR | Identification interval |
|---|---:|---:|---:|---:|---:|
| Clean | 13/40, 32.5% [20.1%, 48.0%] | 40/40 | 0/40 | 0/40 | 0% |
| Adversarial | 8/40, 20.0% [10.5%, 34.8%] | 40/40 | 40/40 | 2/40, 5.0% [1.4%, 16.5%] | 5.0–20.0% |
| Defended | 11/40, 27.5% [16.1%, 42.8%] | 40/40 | 40/40 | 1/40, 2.5% [0.4%, 12.9%] | 2.5–12.5% |

All three confirmed attacks were Base Encoding responses. Base Encoding bounds were 2–3/14
adversarial and 1–2/14 defended; Reverse Text bounds were 0–5/13 and 0–3/13; all 26 Substitution
Cipher outputs were confirmed failures. Six adversarial and four defended outputs remain unresolved.

## 7. Discussion

H1 receives limited support: two unequivocally encoded responses occurred after exposure and
adversarial task success was 12.5 percentage points below clean. H2 is not established: the defense
changed confirmed attacks from two to one, but the attack-success identification intervals overlap.
Defended task success recovered 7.5 points relative to adversarial while remaining 5 points below
clean; H3 therefore remains descriptive rather than inferential.

Baseline failure is not explained by retrieval omission: the target was in the top three for every
clean case. It ranked first, second, and third in 27, 11, and 2 cases, with 8, 4, and 1 successes.
Qualitative inspection found ordinary table-reading, counting, comparison, and distractor errors.
The small quantized model and multi-table prompt are plausible contributors, but this design cannot
separate them causally.

The experiment also exposes a measurement lesson: apparently simple exact matching can create both
numeric false positives and multi-answer false negatives. Preserving every scoring generation makes
that correction inspectable without altering or regenerating the underlying outputs.

## 8. Limitations

The retrieval-eligible sampling frame inflates exposure and is unsuitable for estimating population
retrieval recall. Qwen2.5-3B's 32.5% clean task success limits separation of attack failures from
ordinary reasoning errors. Quantization may alter behavior. Only 40 cases, one model, one seed, one
retriever, one insertion location, and three formatting attacks were tested. No blinded human
annotation, inter-rater agreement, external judge, adaptive attack, tool use, or exfiltration was
included. Ten ambiguous behaviors remain unresolved, so the intervals are deliberately broad.

## 9. Responsible research considerations

The tested attacks only manipulate text formatting and do not invoke tools or external actions.
Raw outputs and criteria are published for audit. The defense is not presented as a security
boundary or guarantee. Third-party data and model licenses remain distinct from the code license.

## 10. Future work

Use a pre-registered rubric with two blinded reviewers; report agreement and adjudication; evaluate
all BIPIA insertion positions; include benign-instruction hard negatives; compare retrieval depths;
test structured separation or instruction detection; and add tool-centric AgentDojo evaluation only
when agent actions enter scope. Larger paired samples are required to estimate small effects.

## 11. References

1. Greshake, K. et al. (2023). [Not what you've signed up for](https://arxiv.org/abs/2302.12173).
2. Yi, J. et al. (2023). [Benchmarking and Defending Against Indirect Prompt Injection Attacks on Large Language Models](https://arxiv.org/abs/2312.14197).
3. Debenedetti, E. et al. (2024). [AgentDojo](https://arxiv.org/abs/2406.13352).
4. Yang, A. et al. (2024). [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115).
5. Hannun, A. et al. (2023). [MLX: Efficient and flexible machine learning on Apple silicon](https://github.com/ml-explore/mlx).
6. Apple ML Research. [MLX-LM](https://github.com/ml-explore/mlx-lm).
