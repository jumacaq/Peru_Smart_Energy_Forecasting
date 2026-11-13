import pytest
import pandas as pd
import numpy as np
from src.metrics import (
    mae,
    rmse,
    mape,
    smape,
    evaluate_all,
    evaluation_table,
)


class TestMAE:
    def test_mae_perfect_prediction(self):
        """Test MAE when predictions are perfect."""
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        y_pred = np.array([10.0, 20.0, 30.0, 40.0])

        result = mae(y_true, y_pred)

        assert result == 0.0

    def test_mae_hand_calculated(self):
        """Test MAE with hand-calculated value."""
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        y_pred = np.array([11.0, 19.0, 32.0, 38.0])

        # Errors: |10-11| = 1, |20-19| = 1, |30-32| = 2, |40-38| = 2
        # MAE = (1 + 1 + 2 + 2) / 4 = 6 / 4 = 1.5
        expected = 1.5
        result = mae(y_true, y_pred)

        assert np.isclose(result, expected, rtol=1e-10)

    def test_mae_with_pandas_series(self):
        """Test MAE with pandas Series input."""
        y_true = pd.Series([10.0, 20.0, 30.0])
        y_pred = pd.Series([12.0, 18.0, 32.0])

        result = mae(y_true, y_pred)

        # Errors: 2, 2, 2 -> MAE = 2.0
        assert result == 2.0

    def test_mae_return_type(self):
        """Test that MAE returns a float."""
        y_true = np.array([10.0, 20.0])
        y_pred = np.array([11.0, 21.0])

        result = mae(y_true, y_pred)

        assert isinstance(result, float)


class TestRMSE:
    def test_rmse_perfect_prediction(self):
        """Test RMSE when predictions are perfect."""
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        y_pred = np.array([10.0, 20.0, 30.0, 40.0])

        result = rmse(y_true, y_pred)

        assert result == 0.0

    def test_rmse_hand_calculated(self):
        """Test RMSE with hand-calculated value."""
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        y_pred = np.array([11.0, 21.0, 29.0, 40.0])

        # Squared errors: 1, 1, 1, 0
        # MSE = (1 + 1 + 1 + 0) / 4 = 0.75
        # RMSE = sqrt(0.75) ≈ 0.866025...
        expected = np.sqrt(0.75)
        result = rmse(y_true, y_pred)

        assert np.isclose(result, expected)

    def test_rmse_larger_errors_penalized(self):
        """Test that RMSE penalizes larger errors more than MAE."""
        y_true = np.array([10.0, 20.0])
        y_pred_a = np.array([11.0, 21.0])  # Errors: 1, 1
        y_pred_b = np.array([12.0, 20.0])  # Errors: 2, 0

        mae_a = mae(y_true, y_pred_a)
        mae_b = mae(y_true, y_pred_b)
        rmse_a = rmse(y_true, y_pred_a)
        rmse_b = rmse(y_true, y_pred_b)

        # Both have same MAE = 1.0
        assert mae_a == mae_b == 1.0

        # But RMSE_b > RMSE_a because of the larger single error
        assert rmse_b > rmse_a

    def test_rmse_return_type(self):
        """Test that RMSE returns a float."""
        y_true = np.array([10.0, 20.0])
        y_pred = np.array([11.0, 21.0])

        result = rmse(y_true, y_pred)

        assert isinstance(result, float)


class TestMAPE:
    def test_mape_perfect_prediction(self):
        """Test MAPE when predictions are perfect."""
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        y_pred = np.array([10.0, 20.0, 30.0, 40.0])

        result = mape(y_true, y_pred)

        assert result == 0.0

    def test_mape_hand_calculated(self):
        """Test MAPE with hand-calculated value."""
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 180.0])

        # MAPE = mean(|100-110|/100, |200-180|/200) * 100
        # = mean(0.1, 0.1) * 100 = 10.0
        expected = 10.0
        result = mape(y_true, y_pred)

        assert np.isclose(result, expected, rtol=1e-10)

    def test_mape_epsilon_prevents_division_by_zero(self):
        """Test that epsilon prevents division by zero when y_true is 0."""
        y_true = np.array([0.0, 100.0])
        y_pred = np.array([10.0, 100.0])

        # Should not raise division by zero error
        result = mape(y_true, y_pred)

        assert isinstance(result, float)
        assert not np.isnan(result)
        assert not np.isinf(result)

    def test_mape_return_type_and_percentage(self):
        """Test that MAPE returns a float and is expressed as percentage."""
        y_true = np.array([100.0, 100.0])
        y_pred = np.array([110.0, 110.0])

        result = mape(y_true, y_pred)

        # Should be ~10 (10%), not 0.1
        assert result > 1.0


class TestSMAPE:
    def test_smape_perfect_prediction(self):
        """Test sMAPE when predictions are perfect."""
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        y_pred = np.array([10.0, 20.0, 30.0, 40.0])

        result = smape(y_true, y_pred)

        assert result == 0.0

    def test_smape_hand_calculated(self):
        """Test sMAPE with hand-calculated value."""
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([120.0, 180.0])

        # sMAPE = mean(2*|y_true - y_pred| / (|y_true| + |y_pred|)) * 100
        # Row 1: 2*|100-120| / (100+120) = 40/220 ≈ 0.1818
        # Row 2: 2*|200-180| / (200+180) = 40/380 ≈ 0.1053
        # sMAPE ≈ mean(0.1818, 0.1053) * 100 ≈ 14.35
        expected_value = (2 * 20 / 220 + 2 * 20 / 380) / 2 * 100
        result = smape(y_true, y_pred)

        assert np.isclose(result, expected_value)

    def test_smape_symmetric(self):
        """Test that sMAPE is symmetric (swapping y_true and y_pred gives same result)."""
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 190.0, 320.0])

        result1 = smape(y_true, y_pred)
        result2 = smape(y_pred, y_true)

        assert np.isclose(result1, result2)

    def test_smape_epsilon_prevents_division_by_zero(self):
        """Test that epsilon prevents division by zero when both are 0."""
        y_true = np.array([0.0, 100.0])
        y_pred = np.array([0.0, 100.0])

        result = smape(y_true, y_pred)

        assert isinstance(result, float)
        assert not np.isnan(result)
        assert not np.isinf(result)

    def test_smape_range_0_to_200(self):
        """Test that sMAPE is in the range [0, 200]."""
        y_true = np.array([100.0, 100.0, 100.0])
        y_pred = np.array([200.0, 0.0, 50.0])

        result = smape(y_true, y_pred)

        assert 0.0 <= result <= 200.0

    def test_smape_return_type(self):
        """Test that sMAPE returns a float."""
        y_true = np.array([10.0, 20.0])
        y_pred = np.array([11.0, 21.0])

        result = smape(y_true, y_pred)

        assert isinstance(result, float)


class TestEvaluateAll:
    def test_evaluate_all_returns_dict(self):
        """Test that evaluate_all returns a dictionary with all metrics."""
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([11.0, 21.0, 29.0])

        result = evaluate_all(y_true, y_pred)

        assert isinstance(result, dict)
        assert "MAE" in result
        assert "RMSE" in result
        assert "MAPE" in result
        assert "sMAPE" in result

    def test_evaluate_all_values_are_floats(self):
        """Test that all metric values are floats."""
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([11.0, 21.0, 29.0])

        result = evaluate_all(y_true, y_pred)

        for metric_name, metric_value in result.items():
            assert isinstance(metric_value, float)

    def test_evaluate_all_consistency(self):
        """Test that evaluate_all results match individual function calls."""
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        y_pred = np.array([11.0, 19.0, 32.0, 38.0])

        result = evaluate_all(y_true, y_pred)

        assert result["MAE"] == mae(y_true, y_pred)
        assert result["RMSE"] == rmse(y_true, y_pred)
        assert result["MAPE"] == mape(y_true, y_pred)
        assert result["sMAPE"] == smape(y_true, y_pred)

    def test_evaluate_all_perfect_predictions(self):
        """Test evaluate_all with perfect predictions."""
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([10.0, 20.0, 30.0])

        result = evaluate_all(y_true, y_pred)

        assert result["MAE"] == 0.0
        assert result["RMSE"] == 0.0
        assert result["MAPE"] == 0.0
        assert result["sMAPE"] == 0.0


class TestEvaluationTable:
    def test_evaluation_table_basic(self):
        """Test that evaluation_table creates a DataFrame with correct structure."""
        models_metrics = {
            "Model_A": {"MAE": 10.0, "RMSE": 15.0, "MAPE": 5.0, "sMAPE": 4.0},
            "Model_B": {"MAE": 8.0, "RMSE": 12.0, "MAPE": 4.0, "sMAPE": 3.5},
        }

        result = evaluation_table(models_metrics)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == (2, 4)
        assert list(result.columns) == ["MAE", "RMSE", "MAPE", "sMAPE"]

    def test_evaluation_table_sorting_by_rmse(self):
        """Test that table is sorted by RMSE in ascending order."""
        models_metrics = {
            "Model_A": {"MAE": 10.0, "RMSE": 15.0, "MAPE": 5.0, "sMAPE": 4.0},
            "Model_B": {"MAE": 8.0, "RMSE": 12.0, "MAPE": 4.0, "sMAPE": 3.5},
            "Model_C": {"MAE": 12.0, "RMSE": 18.0, "MAPE": 6.0, "sMAPE": 5.0},
        }

        result = evaluation_table(models_metrics, sort_by="RMSE", ascending=True)

        # Should be sorted: Model_B (12.0), Model_A (15.0), Model_C (18.0)
        assert result.index.tolist() == ["Model_B", "Model_A", "Model_C"]
        assert result["RMSE"].tolist() == [12.0, 15.0, 18.0]

    def test_evaluation_table_sorting_by_mae(self):
        """Test that table can be sorted by MAE."""
        models_metrics = {
            "Model_A": {"MAE": 10.0, "RMSE": 15.0},
            "Model_B": {"MAE": 8.0, "RMSE": 12.0},
        }

        result = evaluation_table(models_metrics, sort_by="MAE", ascending=True)

        # Should be sorted: Model_B (8.0), Model_A (10.0)
        assert result.index.tolist() == ["Model_B", "Model_A"]

    def test_evaluation_table_descending_sort(self):
        """Test that table can be sorted in descending order."""
        models_metrics = {
            "Model_A": {"MAE": 10.0, "RMSE": 15.0},
            "Model_B": {"MAE": 8.0, "RMSE": 12.0},
        }

        result = evaluation_table(models_metrics, sort_by="RMSE", ascending=False)

        # Should be sorted descending: Model_A (15.0), Model_B (12.0)
        assert result.index.tolist() == ["Model_A", "Model_B"]
        assert result["RMSE"].tolist() == [15.0, 12.0]

    def test_evaluation_table_invalid_sort_column(self):
        """Test that table handles invalid sort_by column gracefully."""
        models_metrics = {
            "Model_A": {"MAE": 10.0, "RMSE": 15.0},
            "Model_B": {"MAE": 8.0, "RMSE": 12.0},
        }

        # Should not raise error, falls back to first column
        result = evaluation_table(models_metrics, sort_by="INVALID_METRIC")

        assert isinstance(result, pd.DataFrame)

    def test_evaluation_table_preserves_all_metrics(self):
        """Test that all metrics are preserved in the table."""
        models_metrics = {
            "Model_A": {"MAE": 10.0, "RMSE": 15.0, "MAPE": 5.0, "sMAPE": 4.0},
            "Model_B": {"MAE": 8.0, "RMSE": 12.0, "MAPE": 4.0, "sMAPE": 3.5},
        }

        result = evaluation_table(models_metrics)

        # Check that all metrics are present
        assert result.loc["Model_A", "MAE"] == 10.0
        assert result.loc["Model_A", "RMSE"] == 15.0
        assert result.loc["Model_A", "MAPE"] == 5.0
        assert result.loc["Model_A", "sMAPE"] == 4.0

        assert result.loc["Model_B", "MAE"] == 8.0
        assert result.loc["Model_B", "RMSE"] == 12.0
        assert result.loc["Model_B", "MAPE"] == 4.0
        assert result.loc["Model_B", "sMAPE"] == 3.5

    def test_evaluation_table_single_model(self):
        """Test evaluation_table with a single model."""
        models_metrics = {
            "Model_A": {"MAE": 10.0, "RMSE": 15.0, "MAPE": 5.0, "sMAPE": 4.0},
        }

        result = evaluation_table(models_metrics)

        assert result.shape == (1, 4)
        assert result.index.tolist() == ["Model_A"]

    def test_evaluation_table_row_index_is_model_names(self):
        """Test that row index contains model names."""
        models_metrics = {
            "Model_A": {"MAE": 10.0, "RMSE": 15.0},
            "Model_B": {"MAE": 8.0, "RMSE": 12.0},
        }

        result = evaluation_table(models_metrics)

        assert "Model_A" in result.index
        assert "Model_B" in result.index
