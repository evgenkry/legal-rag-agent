"""Оценка RAG: загрузка eval-кейсов, прогон через модель, формирование отчетов и шаблонов human-review."""

from __future__ import annotations

import asyncio
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REPORTS_DIR = Path(__file__).resolve().parent / "reports"

BASE_REPORT_COLUMNS = [
    "case_id",
    "trace_id",
    "dataset_name",
    "question",
    "reference_answer",
    "model_answer",
    "source_file",
    "source_url",
    "sources_found",
    "latency_ms",
    "agent_outcome",
]

HUMAN_REVIEW_COLUMNS = [
    *BASE_REPORT_COLUMNS,
    "legal_accuracy",
    "completeness",
    "citation_quality",
    "risk_level",
    "final_verdict",
    "reviewer_comment",
    "reviewer_id",
    "reviewed_at",
]


@dataclass(slots=True)
class EvalCase:
    case_id: str
    question: str
    reference_answer: str
    source_file: str = ""
    source_url: str = ""
    dataset_name: str = "default"
    trace_id: str = ""
    reference_contexts: list[str] | None = None


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def ensure_reports_dir() -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    return REPORTS_DIR


def load_jsonl_eval_cases(path: Path, dataset_name: str = "reference_qa") -> list[EvalCase]:
    cases: list[EvalCase] = []
    with open(path, encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            question = str(row.get("question", "")).strip()
            if not question:
                continue
            case_id = str(row.get("case_id") or f"{dataset_name}_{idx}")
            case = EvalCase(
                case_id=case_id,
                question=question,
                reference_answer=str(row.get("reference_answer", "")),
                source_file=str(row.get("source_file", path.name)),
                source_url=str(row.get("source_url", "")),
                dataset_name=str(row.get("dataset_name", dataset_name)),
                trace_id=str(row.get("trace_id", f"{dataset_name}_{case_id}")),
                reference_contexts=row.get("reference_contexts"),
            )
            cases.append(case)
    return cases


async def run_eval_cases(
    rag: Any,
    cases: Iterable[EvalCase],
    *,
    user_id: str,
    interaction_type: str,
    trace_prefix: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        trace_id = case.trace_id or f"{trace_prefix}_{case.case_id}"
        result = await rag.query(
            question=case.question,
            user_id=user_id,
            interaction_type=interaction_type,
            trace_id=trace_id,
        )
        rows.append(
            {
                "case_id": case.case_id,
                "trace_id": trace_id,
                "dataset_name": case.dataset_name,
                "question": case.question,
                "reference_answer": case.reference_answer,
                "model_answer": result.get("answer", ""),
                "source_file": case.source_file,
                "source_url": case.source_url,
                "sources_found": result.get("sources_found"),
                "latency_ms": result.get("latency_ms"),
                "agent_outcome": result.get("agent_outcome"),
                "context_texts": result.get("context_texts", []),
            }
        )
    return rows


def run_eval_cases_sync(
    rag: Any,
    cases: Iterable[EvalCase],
    *,
    user_id: str,
    interaction_type: str,
    trace_prefix: str,
) -> list[dict[str, Any]]:
    return asyncio.run(
        run_eval_cases(
            rag,
            cases,
            user_id=user_id,
            interaction_type=interaction_type,
            trace_prefix=trace_prefix,
        )
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def write_human_review_template(path: Path, rows: list[dict[str, Any]]) -> None:
    template_rows = []
    for row in rows:
        template_rows.append({k: row.get(k, "") for k in BASE_REPORT_COLUMNS})
    write_csv(path, template_rows, HUMAN_REVIEW_COLUMNS)


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def evaluate_ragas(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, faithfulness

    eval_set = {
        "question": [r["question"] for r in rows],
        "answer": [r["model_answer"] for r in rows],
        "contexts": [r.get("context_texts", []) for r in rows],
        "ground_truth": [r["reference_answer"] for r in rows],
    }
    ds = Dataset.from_dict(eval_set)
    scores = evaluate(ds, metrics=[faithfulness, answer_relevancy, context_precision])
    return dict(scores)
