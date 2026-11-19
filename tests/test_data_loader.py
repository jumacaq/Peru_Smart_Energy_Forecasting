import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import logging
from src.data_loader import (
    load_raw_data,
    parse_and_index,
    enforce_30min_grid,
    validate_series,
    save_processed_dataset,
    load_processed_dataset,
)


@pytest.fixture
def temp_excel_file():
    """Create a temporary Excel file with test data."""
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    data = {
        "FECHA": ["01/01/2024 00:00", "01/01/2024 00:30", "01/01/2024 01:00"],
        "EJECUTADO": [100.5, 101.2, 102.3],
    }
    df = pd.DataFrame(data)
    df.to_excel(tmp_path, index=False, startrow=3)
    yield tmp_path
    tmp_path.unlink()


@pytest.fixture
def temp_csv_file():
    """Create a temporary CSV file with test data."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    data = {
        "FECHA": ["01/01/2024 00:00", "01/01/2024 00:30", "01/01/2024 01:00"],
        "EJECUTADO": [100.5, 101.2, 102.3],
    }
    df = pd.DataFrame(data)
    df.to_csv(tmp_path, index=False)
    yield tmp_path
    tmp_path.unlink()


class TestLoadRawData:
    def test_load_xlsx_file(self, temp_excel_file):
        """Test loading data from Excel file."""
        df = load_raw_data(temp_excel_file)
        assert isinstance(df, pd.DataFrame)
        assert "FECHA" in df.columns
        assert "EJECUTADO" in df.columns
        assert len(df) == 3

    def test_load_csv_file(self, temp_csv_file):
        """Test loading data from CSV file."""
        df = load_raw_data(temp_csv_file)
        assert isinstance(df, pd.DataFrame)
        assert "FECHA" in df.columns
        assert "EJECUTADO" in df.columns
        assert len(df) == 3

    def test_file_not_found(self):
        """Test FileNotFoundError when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_raw_data("nonexistent_file.xlsx")

    def test_unsupported_format(self):
        """Test ValueError for unsupported file format."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                load_raw_data(tmp_path)
        finally:
            tmp_path.unlink()


class TestParseAndIndex:
    def test_timestamp_parsing(self):
        """Test that DATE column is correctly parsed and indexed."""
        data = {
            "FECHA": ["01/01/2024 00:00", "01/01/2024 00:30", "01/01/2024 01:00"],
            "EJECUTADO": [100.0, 101.0, 102.0],
        }
        df = pd.DataFrame(data)
        result = parse_and_index(df)

        assert result.index.name == "DATE"
        assert isinstance(result.index, pd.DatetimeIndex)
        assert len(result) == 3
        assert result.index[0] == pd.Timestamp("2024-01-01 00:00:00")

    def test_duplicate_timestamp_removal(self):
        """Test that duplicate timestamps are removed."""
        data = {
            "FECHA": ["01/01/2024 00:00", "01/01/2024 00:00", "01/01/2024 00:30"],
            "EJECUTADO": [100.0, 200.0, 101.0],
        }
        df = pd.DataFrame(data)
        result = parse_and_index(df)

        assert len(result) == 2
        assert result.index.duplicated().sum() == 0
        assert result.loc[pd.Timestamp("2024-01-01 00:00:00"), "Demand"] == 100.0

    def test_sorting_ascending(self):
        """Test that index is sorted in ascending order."""
        data = {
            "FECHA": ["01/01/2024 01:00", "01/01/2024 00:00", "01/01/2024 00:30"],
            "EJECUTADO": [102.0, 100.0, 101.0],
        }
        df = pd.DataFrame(data)
        result = parse_and_index(df)

        assert result.index.is_monotonic_increasing
        assert result.index[0] == pd.Timestamp("2024-01-01 00:00:00")
        assert result.index[-1] == pd.Timestamp("2024-01-01 01:00:00")

    def test_column_rename(self):
        """Test that EJECUTADO is renamed to Demand."""
        data = {
            "FECHA": ["01/01/2024 00:00"],
            "EJECUTADO": [100.0],
        }
        df = pd.DataFrame(data)
        result = parse_and_index(df)

        assert "Demand" in result.columns
        assert "EJECUTADO" not in result.columns
        assert result.shape[1] == 1


class TestEnforce30MinGrid:
    def test_reindex_creates_full_grid(self):
        """Test that missing 30-min slots are added."""
        dates = pd.date_range("2024-01-01 00:00", periods=2, freq="1h")
        df = pd.DataFrame({
            "Demand": [100.0, 102.0]
        }, index=dates)
        df.index.name = "DATE"

        result = enforce_30min_grid(df)

        # Should have 3 rows: 00:00, 00:30, 01:00
        assert len(result) == 3
        expected_freq = pd.tseries.frequencies.to_offset("30min")
        assert result.index.freq == expected_freq

    def test_interpolation_within_limit(self):
        """Test that gaps <= 4 steps (2 hours) are interpolated."""
        dates = [
            pd.Timestamp("2024-01-01 00:00"),
            pd.Timestamp("2024-01-01 02:00"),
        ]
        df = pd.DataFrame({
            "Demand": [100.0, 110.0]
        }, index=dates)
        df.index.name = "DATE"

        result = enforce_30min_grid(df)

        # Should have 5 rows with interpolated values
        assert len(result) == 5
        assert result["Demand"].isna().sum() == 0
        # Middle value should be interpolated
        assert result.iloc[2]["Demand"] == 105.0

    def test_large_gap_logged(self, caplog):
        """Gaps > 4 steps log a warning and stay NaN."""
        dates = [
            pd.Timestamp("2024-01-01 00:00"),
            pd.Timestamp("2024-01-01 04:30"),
        ]
        df = pd.DataFrame({"Demand": [100.0, 120.0]}, index=dates)
        df.index.name = "DATE"

        with caplog.at_level(logging.WARNING):
            result = enforce_30min_grid(df)

        assert len(result) == 10
        assert "Not interpolating" in caplog.text
        assert result["Demand"].iloc[1:9].isna().all()

    def test_index_name_preserved(self):
        """Test that DATE index name is preserved."""
        dates = pd.date_range("2024-01-01", periods=2, freq="30min")
        df = pd.DataFrame({"Demand": [100.0, 101.0]}, index=dates)
        df.index.name = "DATE"

        result = enforce_30min_grid(df)

        assert result.index.name == "DATE"

    def test_long_gaps_are_not_partially_filled(self):
        idx = pd.date_range("2024-01-01", periods=30, freq="30min")
        df = pd.DataFrame({"Demand": np.arange(30, dtype=float)}, index=idx)

        #3-step gap at positions 5-7 and 6-step gap at positions 15-20
        df = df.drop(idx[[5, 6, 7] + list(range(15, 21))])

        out = enforce_30min_grid(df)

        assert len(out) == 30
        # short gap (<= 4 steps): fully filled with the linear values
        assert out["Demand"].iloc[5:8].tolist() == [5.0, 6.0, 7.0]
        # long gap (> 4 steps): every step stays NaN, none half-filled
        assert out["Demand"].iloc[15:21].isna().all()


class TestValidateSeries:
    def test_valid_series_report(self):
        """Test validation report for valid data."""
        dates = pd.date_range("2024-01-01", periods=4, freq="30min")
        df = pd.DataFrame({"Demand": [100.0, 101.0, 102.0, 103.0]}, index=dates)

        report = validate_series(df)

        assert report["row_count"] == 4
        assert report["actual_count"] == 4
        assert report["min_demand"] == 100.0
        assert report["max_demand"] == 103.0
        assert report["nan_count"] == 0
        assert report["non_positive_count"] == 0

    def test_negative_demand_raises_error(self):
        """Test that negative demand values raise ValueError."""
        dates = pd.date_range("2024-01-01", periods=2, freq="30min")
        df = pd.DataFrame({"Demand": [100.0, -50.0]}, index=dates)

        with pytest.raises(ValueError, match="Negative demand values found"):
            validate_series(df)

    def test_nan_count_in_report(self):
        """Test that NaN count is correctly reported."""
        dates = pd.date_range("2024-01-01", periods=4, freq="30min")
        df = pd.DataFrame({"Demand": [100.0, np.nan, 102.0, np.nan]}, index=dates)

        report = validate_series(df)

        assert report["nan_count"] == 2

    def test_duplicate_timestamp_count(self):
        """Test duplicate timestamp detection."""
        dates = [
            pd.Timestamp("2024-01-01 00:00"),
            pd.Timestamp("2024-01-01 00:00"),
            pd.Timestamp("2024-01-01 00:30"),
        ]
        df = pd.DataFrame({"Demand": [100.0, 200.0, 101.0]}, index=dates)

        report = validate_series(df)

        assert report["duplicate_timestamp_count"] == 1

    def test_expected_count_calculation(self):
        """Test that expected count matches date range."""
        dates = pd.date_range("2024-01-01", periods=3, freq="30min")
        df = pd.DataFrame({"Demand": [100.0, 101.0, 102.0]}, index=dates)

        report = validate_series(df)

        # 3 rows with 30-min frequency should have expected_count = 3
        assert report["expected_count"] == 3


class TestProcessedDatasetIO:
    def test_save_and_load_processed_dataset(self):
        """Test saving and loading processed dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "processed" / "demand.parquet"

            dates = pd.date_range("2024-01-01", periods=4, freq="30min")
            df = pd.DataFrame({"Demand": [100.0, 101.0, 102.0, 103.0]}, index=dates)
            df.index.name = "DATE"

            save_processed_dataset(df, str(path))
            assert path.exists()

            loaded_df = load_processed_dataset(str(path))

            # Check that values are equal (parquet may change index dtype slightly)
            assert (loaded_df["Demand"] == df["Demand"]).all()
            assert loaded_df.index.name == "DATE"
            assert len(loaded_df) == len(df)

    def test_load_nonexistent_processed_dataset(self):
        """Test that loading nonexistent dataset raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_processed_dataset("/nonexistent/path/demand.parquet")

    def test_index_name_restored_on_load(self):
        """Test that index name 'DATE' is restored when loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "demand.parquet"

            dates = pd.date_range("2024-01-01", periods=2, freq="30min")
            df = pd.DataFrame({"Demand": [100.0, 101.0]}, index=dates)
            df.index.name = "DATE"

            save_processed_dataset(df, str(path))
            loaded_df = load_processed_dataset(str(path))

            assert loaded_df.index.name == "DATE"
