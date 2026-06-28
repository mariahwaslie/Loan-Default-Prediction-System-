import pandas as pd
import pytest

from src.preprocessing import validate_features, convert_categorical_columns


def test_validate_features_reorders_columns():
    df = pd.DataFrame({
        "b": [1],
        "a": [2]
    })

    expected = ["a", "b"]

    result = validate_features(df, expected)

    assert list(result.columns) == expected


def test_validate_features_missing_column_raises():
    df = pd.DataFrame({
        "a": [1]
    })

    with pytest.raises(ValueError):
        validate_features(df, ["a", "b"])


def test_convert_categorical_columns():
    df = pd.DataFrame({
        "grade": ["A", "B"],
        "loan_amnt": [1000, 2000]
    })

    result = convert_categorical_columns(df, ["grade"])

    assert str(result["grade"].dtype) == "category"