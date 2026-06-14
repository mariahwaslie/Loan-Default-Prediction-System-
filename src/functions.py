import pandas as pd

def drop_cols_by_missing(df, threshold=0.5, return_dropped=False, verbose=True):
    """
    Efficiently drop columns exceeding missing value threshold.
    
    Parameters
    ----------
    df : pandas.DataFrame
    threshold : float
        Fraction of missing values (0–1)
    return_dropped : bool
        If True, also returns list of dropped columns
    verbose : bool
        Print summary
    
    Returns
    -------
    DataFrame (and optionally list of dropped columns)
    """
    
    # Compute missing fraction ONCE (fast vectorized op)
    missing_frac = df.isna().mean()

    # Boolean mask (faster than building lists first)
    drop_mask = missing_frac > threshold

    # Select columns to drop
    cols_to_drop = missing_frac.index[drop_mask]

    if verbose:
        print(f"Dropping {len(cols_to_drop)} columns > {threshold:.0%} missing")

    # Drop in one operation (fastest path in pandas)
    df_clean = df.drop(columns=cols_to_drop)

    if return_dropped:
        return df_clean, cols_to_drop.tolist()

    return df_clean

import pandas as pd

def drop_rows_by_missing(df, max_missing_frac=0.5, verbose=True):
    """
    Drop rows where missing fraction exceeds threshold.

    Parameters
    ----------
    df : pandas.DataFrame
    max_missing_frac : float
        Max allowed fraction of missing values per row.
        Example: 0.5 = drop rows with >50% missing
    verbose : bool
        Print how many rows were removed

    Returns
    -------
    pandas.DataFrame
    """

    # Compute row-wise missing fraction (FAST vectorized op)
    row_missing_frac = df.isna().mean(axis=1)

    # Boolean mask for rows to keep
    keep_mask = row_missing_frac <= max_missing_frac

    if verbose:
        removed = (~keep_mask).sum()
        print(f"Removing {removed} rows with > {max_missing_frac:.0%} missing values")

    return df.loc[keep_mask].copy()