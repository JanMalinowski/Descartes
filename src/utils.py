import pandas as pd
from typing import List


def change_money(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Function for changing money format from $xyz,xyz to xyzxyz.
    """
    assert isinstance(df, pd.DataFrame)
    assert isinstance(cols, list)

    for col in cols:
        df[col] = df[col].map(
            lambda x: x.replace("$", "").replace(",", "") if isinstance(x, str) else x
        )
        df[col] = df[col].astype(float)
    return df
