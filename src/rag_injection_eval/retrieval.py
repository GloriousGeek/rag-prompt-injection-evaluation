"""Small, auditable TF-IDF retriever used by all experimental conditions."""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass(frozen=True)
class RetrievedDocument:
    document_id: str
    text: str
    score: float
    rank: int
    is_relevant: bool
    contains_attack: bool
    attack: dict | None


def poison(text: str, attack_prompt: str) -> str:
    """Match BIPIA's insert_end operation exactly."""
    return "\n".join((text, attack_prompt))


def retrieve(
    question: str,
    corpus: list[dict],
    target_document_id: str,
    poisoned_target: bool,
    top_k: int = 1,
) -> tuple[list[RetrievedDocument], int]:
    documents: list[dict] = []
    for item in corpus:
        is_target = item["document_id"] == target_document_id
        attacked = bool(poisoned_target and is_target)
        text = poison(item["context"], item["attack"]["prompt"]) if attacked else item["context"]
        documents.append({**item, "retrieval_text": text, "contains_attack": attacked})

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        sublinear_tf=True,
        token_pattern=r"(?u)\b\w+\b",
    )
    matrix = vectorizer.fit_transform([item["retrieval_text"] for item in documents] + [question])
    scores = (matrix[:-1] @ matrix[-1].T).toarray().ravel()
    order = sorted(range(len(documents)), key=lambda index: (-scores[index], documents[index]["document_id"]))
    target_rank = next(rank for rank, index in enumerate(order, start=1) if documents[index]["document_id"] == target_document_id)

    results = []
    for rank, index in enumerate(order[:top_k], start=1):
        item = documents[index]
        results.append(
            RetrievedDocument(
                document_id=item["document_id"],
                text=item["retrieval_text"],
                score=float(scores[index]),
                rank=rank,
                is_relevant=item["document_id"] == target_document_id,
                contains_attack=item["contains_attack"],
                attack=item["attack"] if item["contains_attack"] else None,
            )
        )
    return results, target_rank
