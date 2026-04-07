#!/usr/bin/env python3
"""Запуск RAGAS оценки качества RAG."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.run_ragas import run_ragas_eval

if __name__ == "__main__":
    run_ragas_eval()
