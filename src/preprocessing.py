import pandas as pd 
import pandas as pd
from src.default_features import BASE_LOAN

def load_input(data):
    """Convert a single loan application (Python dictionary)
    into a one-row pandas DataFrame.

    Example input:
    {
        "loan_amnt": 10000,
        "annual_inc": 65000,
        "grade": "B"
    }
    """

    # Put the dictionary inside a list so pandas creates
    # one row instead of treating each key as its own column.
    return pd.DataFrame([data])

def validate_features(df, expected_features):
    """
    Make sure every feature required by the trained model exists.
    If a required feature is missing, raise an error so the user
    knows exactly what needs to be provided.
    """
    # Find every feature the model expects that is missing
    missing_features = [
        feature for feature in expected_features
        if feature not in df.columns
    ]

    # Stop immediately if anything is missing.
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")

    # Keep ONLY the columns used during training,
    # and put them in the exact same order.
    return df[expected_features]




def prepare_input(data: dict, features: list, cat_features: list) -> pd.DataFrame:
    """
    Build a model-ready dataframe from raw API input.

    RULES:
    - Always start from BASE_LOAN defaults
    - Override with user-provided values
    - Ensure all required features exist
    - Enforce correct feature order
    - Cast categorical columns properly
    """

    # 1. start with defaults
    record = BASE_LOAN.copy()

    # 2. override with incoming API data
    record.update(data)

    # 3. convert to dataframe
    df = pd.DataFrame([record])

    # 4. enforce feature order (VERY IMPORTANT)
    df = df[features]

    # 5. cast categorical columns
    for col in cat_features:
        df[col] = df[col].astype("category")

    return df

def convert_categorical_columns(
    df: pd.DataFrame,
    categorical_columns: list
) -> pd.DataFrame:

    """
    Convert categorical columns into pandas 'category' dtype.
    CatBoost expects these columns to be treated as categorical,
    just like they were during training.
    """

    # Loop through every categorical column saved
    # inside the trained model artifact.
    for column in categorical_columns:
        # Only convert columns that actually exist.
        if column in df.columns:
            df[column] = df[column].astype("category")
    return df
