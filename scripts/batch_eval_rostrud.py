#!/usr/bin/env python3
"""
Пакетная оценка на разъяснениях Роструда без использования Telegram.
Сохраняет JSONL с вопросом, эталонным ответом, ответом конвейера и метаданными.
Опционально: RAGAS (--ragas).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.core import (
    BASE_REPORT_COLUMNS,
    EvalCase,
    ensure_reports_dir,
    evaluate_ragas,
    run_eval_cases,
    utc_stamp,
    write_csv,
    write_human_review_template,
    write_jsonl,
    write_summary,
)
from src.api.dependencies import get_rag_service
from src.core.logging_config import setup_logging
from src.db.pool import close_pool, init_pool
from src.knowledge.rostrud_faq_parser import parse_rostrud_faq_markdown

setup_logging("INFO")
logger = logging.getLogger(__name__)

DATASET_NAME = "rostrud_faq"


async def _run(args: argparse.Namespace) -> None:
    await init_pool()
    try:
        rag = get_rag_service()
        faq_dir = Path(args.faq_dir)
        files = sorted(faq_dir.glob("faq_*.md"))
        if args.limit:
            files = files[: args.limit]

        eval_cases: list[EvalCase] = []
        for path in files:
            text = path.read_text(encoding="utf-8")
            parsed = parse_rostrud_faq_markdown(text, str(path))
            if not parsed:
                logger.warning("Skip unparsable: %s", path)
                continue
            eval_cases.append(
                EvalCase(
                    case_id=parsed.faq_id,
                    question=parsed.question,
                    reference_answer=parsed.answer_body[:20000],
                    source_file=path.name,
                    source_url=parsed.source_url,
                    dataset_name=DATASET_NAME,
                    trace_id=f"batch_{parsed.faq_id}",
                )
            )

        rows_out = await run_eval_cases(
            rag,
            eval_cases,
            user_id="batch_eval",
            interaction_type="batch_eval",
            trace_prefix="batch",
        )
        for row in rows_out:
            logger.info("Evaluated case_id=%s outcome=%s", row.get("case_id"), row.get("agent_outcome"))

        reports_dir = ensure_reports_dir()
        stamp = utc_stamp()
        base = f"rostrud_batch_{stamp}"
        jsonl_path = reports_dir / f"{base}.jsonl"
        write_jsonl(jsonl_path, rows_out)
        logger.info("Wrote %d rows to %s", len(rows_out), jsonl_path)

        csv_path = reports_dir / f"{base}.csv"
        if rows_out:
            write_csv(csv_path, rows_out, BASE_REPORT_COLUMNS)
            logger.info("Wrote CSV %s", csv_path)

        human_csv_path = reports_dir / f"{base}_human_review.csv"
        write_human_review_template(human_csv_path, rows_out)
        logger.info("Wrote human review template %s", human_csv_path)

        if args.ragas and rows_out:
            try:
                scores = evaluate_ragas(rows_out)
            except ImportError:
                logger.warning("RAGAS/datasets not installed; skip --ragas")
                return

            metrics_path = reports_dir / f"{base}_ragas.json"
            write_summary(metrics_path, scores)
            logger.info("RAGAS scores -> %s", metrics_path)
    finally:
        await close_pool()


def main() -> None:
    p = argparse.ArgumentParser(description="Batch eval on Rostrud FAQ")
    p.add_argument("--limit", type=int, default=100, help="Max FAQ files (default 100)")
    p.add_argument(
        "--faq-dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "knowledge_base" / "rostrud" / "faq"),
    )
    p.add_argument("--ragas", action="store_true", help="Run RAGAS metrics (requires ragas)")
    args = p.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
