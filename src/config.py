"""
Project-wide path constants.

Everything is resolved relative to the repository root, so the paths work no
matter which directory a script or test is launched from.

Layout
------
data/processed/     processed inputs (demand.parquet)
artifacts/          saved model files consumed downstream (best_forecaster.pkl + metadata)
reports/            benchmark tables produced by scripts/run_benchmark.py
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
DEMAND_PARQUET_PATH = DATA_DIR / "processed" / "demand.parquet"

ARTIFACTS_DIR = ROOT / "artifacts"
BEST_MODEL_PATH = ARTIFACTS_DIR / "best_forecaster.pkl"
BEST_MODEL_META_PATH = ARTIFACTS_DIR / "best_forecaster_meta.json"

REPORTS_DIR = ROOT / "reports"
BENCHMARK_WINDOWS_CSV = REPORTS_DIR / "benchmark_windows.csv"
BENCHMARK_RESULTS_CSV = REPORTS_DIR / "benchmark_results.csv"
