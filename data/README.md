# Data provenance

`raw/bipia/table_test.jsonl` and `raw/bipia/text_attack_test.json` are exact files from
[`microsoft/BIPIA`](https://github.com/microsoft/BIPIA) at commit
`a004b69ec0dd446e0afd461d98cb5e96e120a5d0`. Their SHA-256 hashes are recorded in
`benchmark_manifest.json` and checked by the preparation code.

`evaluation_set.jsonl` is this project's deterministic 40-row adaptation. It preserves the
original `context`, `question`, `ideal`, source row index, and BIPIA attack identifier. The
selection and adaptation are described in the manifest. Results from this project are **not**
official BIPIA scores.

BIPIA code is MIT licensed. Its TableQA data derives from WikiTableQuestions and is identified
by BIPIA as CC BY-SA 4.0. Consult the upstream licenses before redistributing the data. The root
MIT license covers this project's original code, not third-party benchmark data or model weights.
