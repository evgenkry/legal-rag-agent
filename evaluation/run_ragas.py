"""Запуск оценки на эталонном датасете."""

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BENCHMARKS = Path(__file__).parent / "benchmarks" / "reference_qa.jsonl"
DATASET_NAME = "reference_qa"


def run_ragas_eval() -> None:
    """Runs benchmark evaluation and writes report artifacts."""
    from evaluation.core import (
        BASE_REPORT_COLUMNS,
        ensure_reports_dir,
        evaluate_ragas,
        load_jsonl_eval_cases,
        run_eval_cases_sync,
        utc_stamp,
        write_csv,
        write_human_review_template,
        write_jsonl,
        write_summary,
    )
    from src.api.dependencies import get_rag_service
    from src.db.pool import close_pool, init_pool

    dataset = load_jsonl_eval_cases(BENCHMARKS, dataset_name=DATASET_NAME)
    if not dataset:
        logger.warning("No benchmark data in %s", BENCHMARKS)
        return

    try:
        asyncio.run(init_pool())
    except Exception as e:
        logger.warning("Failed to init pool for evaluation logs: %s", e)

    try:
        rag = get_rag_service()
        rows_out = run_eval_cases_sync(
            rag,
            dataset,
            user_id="eval_reference",
            interaction_type="evaluation",
            trace_prefix="eval_reference",
        )

        reports_dir = ensure_reports_dir()
        stamp = utc_stamp()
        base = f"{DATASET_NAME}_{stamp}"
        jsonl_path = reports_dir / f"{base}.jsonl"
        csv_path = reports_dir / f"{base}.csv"
        human_path = reports_dir / f"{base}_human_review.csv"

        write_jsonl(jsonl_path, rows_out)
        write_csv(csv_path, rows_out, BASE_REPORT_COLUMNS)
        write_human_review_template(human_path, rows_out)

        logger.info("Wrote %d rows to %s", len(rows_out), jsonl_path)
        logger.info("Wrote CSV %s", csv_path)
        logger.info("Wrote human review template %s", human_path)

        try:
            scores = evaluate_ragas(rows_out)
        except ImportError:
            logger.warning("RAGAS/datasets not installed; skip metrics export")
            return

        ragas_path = reports_dir / f"{base}_ragas.json"
        write_summary(ragas_path, scores)
        logger.info("RAGAS scores -> %s", ragas_path)
    finally:
        try:
            asyncio.run(close_pool())
        except Exception:
            pass


if __name__ == "__main__":
    run_ragas_eval()
