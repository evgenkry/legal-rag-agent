#!/usr/bin/env python3
"""Проверяет корректность и агрегирует CSV-аннотации экспертной оценки."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.core import HUMAN_REVIEW_COLUMNS

SCORE_FIELDS = ("legal_accuracy", "completeness", "citation_quality")
VERDICTS = {"pass", "needs_fix", "fail"}


def _to_score(value: str) -> int | None:
    value = (value or "").strip()
    if not value:
        return None
    score = int(value)
    if score < 1 or score > 5:
        raise ValueError(f"score out of range 1..5: {value}")
    return score


def _validate_row(row: dict[str, str], row_num: int) -> list[str]:
    errors: list[str] = []
    for col in HUMAN_REVIEW_COLUMNS:
        if col not in row:
            errors.append(f"row {row_num}: missing column '{col}'")

    verdict = (row.get("final_verdict") or "").strip()
    if verdict and verdict not in VERDICTS:
        errors.append(f"row {row_num}: invalid final_verdict '{verdict}'")

    for field in SCORE_FIELDS:
        try:
            _to_score(row.get(field, ""))
        except ValueError as e:
            errors.append(f"row {row_num}: {e}")
    return errors


def _aggregate(rows: list[dict[str, str]]) -> dict:
    score_values: dict[str, list[int]] = {f: [] for f in SCORE_FIELDS}
    verdict_counts: dict[str, int] = {v: 0 for v in sorted(VERDICTS)}

    for row in rows:
        for field in SCORE_FIELDS:
            score = _to_score(row.get(field, ""))
            if score is not None:
                score_values[field].append(score)
        verdict = (row.get("final_verdict") or "").strip()
        if verdict in verdict_counts:
            verdict_counts[verdict] += 1

    score_summary = {}
    for field, values in score_values.items():
        score_summary[field] = {
            "filled": len(values),
            "mean": round(statistics.fmean(values), 4) if values else None,
        }

    return {
        "rows_total": len(rows),
        "score_summary": score_summary,
        "verdict_counts": verdict_counts,
    }


def _write_summary_csv(path: Path, payload: dict) -> None:
    summary = payload.get("summary", {})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["section", "metric", "value"])
        writer.writerow(["meta", "input", payload.get("input", "")])
        writer.writerow(["meta", "valid", payload.get("valid", False)])
        writer.writerow(["meta", "errors_count", len(payload.get("errors", []))])
        writer.writerow(["summary", "rows_total", summary.get("rows_total", 0)])

        for field, field_summary in summary.get("score_summary", {}).items():
            writer.writerow(["score_summary", f"{field}.filled", field_summary.get("filled")])
            writer.writerow(["score_summary", f"{field}.mean", field_summary.get("mean")])

        for verdict, count in summary.get("verdict_counts", {}).items():
            writer.writerow(["verdict_counts", verdict, count])


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and summarize human-review CSV")
    parser.add_argument("--input", required=True, type=str, help="Path to *_human_review.csv")
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional path for JSON summary (default: <input>_summary.json)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_name(f"{input_path.stem}_summary.json")
    csv_output_path = output_path.with_suffix(".csv")

    with open(input_path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    errors: list[str] = []
    for idx, row in enumerate(rows, start=2):
        errors.extend(_validate_row(row, idx))

    summary = _aggregate(rows)
    payload = {"input": str(input_path), "valid": not errors, "errors": errors, "summary": summary}
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_summary_csv(csv_output_path, payload)

    print(f"Validation report written: {output_path}")
    print(f"CSV summary written: {csv_output_path}")
    if errors:
        print(f"Validation errors: {len(errors)}")
        raise SystemExit(1)
    print("Validation passed")


if __name__ == "__main__":
    main()
