from pathlib import Path

from src import config


class TestPathConstants:
    def test_root_is_repository_root(self):
        assert (config.ROOT / "pyproject.toml").is_file()
        assert (config.ROOT / "src" / "config.py").is_file()

    def test_artifacts_live_in_artifacts_dir(self):
        assert config.ARTIFACTS_DIR == config.ROOT / "artifacts"
        assert config.BEST_MODEL_PATH.parent == config.ARTIFACTS_DIR
        assert config.BEST_MODEL_META_PATH.parent == config.ARTIFACTS_DIR
        assert config.BEST_MODEL_PATH.name == "best_forecaster.pkl"
        assert config.BEST_MODEL_META_PATH.name == "best_forecaster_meta.json"

    def test_benchmark_csvs_live_in_reports_dir(self):
        assert config.REPORTS_DIR == config.ROOT / "reports"
        assert config.BENCHMARK_WINDOWS_CSV.parent == config.REPORTS_DIR
        assert config.BENCHMARK_RESULTS_CSV.parent == config.REPORTS_DIR

    def test_demand_parquet_path(self):
        assert config.DEMAND_PARQUET_PATH == config.ROOT / "data" / "processed" / "demand.parquet"

    def test_all_paths_are_absolute_pathlib_objects(self):
        for name in (
            "ROOT", "DATA_DIR", "DEMAND_PARQUET_PATH", "ARTIFACTS_DIR", "BEST_MODEL_PATH",
            "BEST_MODEL_META_PATH", "REPORTS_DIR", "BENCHMARK_WINDOWS_CSV", "BENCHMARK_RESULTS_CSV",
        ):
            value = getattr(config, name)
            assert isinstance(value, Path), name
            assert value.is_absolute(), name
