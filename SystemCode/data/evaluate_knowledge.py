"""Small retrieval acceptance set; does not measure generated-answer factuality."""
import json
import os
from pathlib import Path

from knowledge_rag import BASE_DIR, DEFAULT_DB, KnowledgeRetriever


if __name__ == "__main__":
    retriever = KnowledgeRetriever(BASE_DIR / "knowledge", DEFAULT_DB,
                                   min_similarity=float(os.getenv("RAG_MIN_SIMILARITY", "0.32")))
    rows = []
    for line in (BASE_DIR / "tests" / "retrieval_cases.jsonl").read_text(encoding="utf-8").splitlines():
        case = json.loads(line)
        result = retriever.search(case["question"], kind=case.get("kind"))
        found = {hit["source"] for hit in result["hits"]}
        passed = bool(found & set(case["expected_any_source"])) if case["expected_any_source"] else not found
        if case.get("expected_text_any"):
            passed = passed and any(term in hit["text"] for term in case["expected_text_any"] for hit in result["hits"])
        rows.append({"question": case["question"], "passed": passed, "found": sorted(found),
                     "expected_any_source": case["expected_any_source"]})
    print(json.dumps({"passed": sum(row["passed"] for row in rows), "total": len(rows),
                      "note": "A retrieval smoke check only; not proof of answer grounding or repair applicability.",
                      "cases": rows}, ensure_ascii=False, indent=2))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
