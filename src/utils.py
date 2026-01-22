"""
Utility functions
"""

import pandas as pd
import os
from typing import Any, Literal

MergeHow = Literal["left", "right", "outer", "inner", "cross"]


def _find_key(d: dict, key: str) -> Any:
    """
    Find key in nested dictionary recursively
    Used for extract tables from data vault based on table name
    """
    if key in d.keys():
        return d[key]

    for v in d.values():
        if not isinstance(v, dict):
            continue
        val = _find_key(v, key)
        if val is not None:
            return val


def find_key(d: dict, keys: str | list[str]) -> Any:
    """
    Same as find_key but generalizes for a list of keys
    to return a dictionary
    """
    if isinstance(keys, str):
        return _find_key(d, keys)
    return {k: _find_key(d, k) for k in keys}


def remove_key(d: dict, key: str | list[str]) -> dict[str, Any]:
    if isinstance(key, str):
        return {k: v for k, v in d.items() if k != key}
    return {k: v for k, v in d.items() if k not in key}


def add_val(og_d: dict, key: str, val: Any) -> dict[str, Any]:
    d = og_d.copy()
    if key not in d:
        d[key] = val
    elif isinstance(d[key], list):
        d[key].append(val)
    else:
        d[key] = [d[key], val]
    return d


def create_filter_query(col: str, val: Any, operator: str = "==") -> str:
    if isinstance(val, str):
        return f"{col} {operator} '{val}'"
    return f"{col} {operator} {val}"


def unique_vals(df: pd.DataFrame, col: str | list[str], dropna: bool = True) -> list:
    base_df = df.dropna() if dropna else df
    if isinstance(col, str):
        return list(base_df[col].unique())
    return list(base_df[col].drop_duplicates().values)


def join_tables(
    df: pd.DataFrame,
    join_table: pd.DataFrame | dict[str, pd.DataFrame],
    join_key: str | list[str] | dict[str, str] | dict[str, list[str]],
    how: MergeHow = "left",
) -> pd.DataFrame:
    result: pd.DataFrame
    if isinstance(join_table, dict):
        if not isinstance(join_key, dict):
            raise ValueError(
                "join_key must also be dictionary if join_table is dictionary."
            )

        diff = set(join_table.keys()) - set(join_key.keys())
        if len(diff) > 0:
            raise ValueError(
                f"join_key must contain the same keys as join_table. Missing keys: {diff}"
            )
        result = df.copy()
        for table_name, table in join_table.items():
            result = pd.merge(result, table, on=join_key[table_name], how=how)
    elif isinstance(join_table, pd.DataFrame) and isinstance(join_key, (str, list)):
        result = pd.merge(df, join_table, on=join_key, how=how)

    return result


def filter_top_n(
    data: pd.DataFrame,
    group_col: str,
    y_col: str,
    n_largest: int,
    x_col: str | None = None,
    x_val_n_largest: Any | None = None,
) -> pd.DataFrame:
    base_data = data[data[x_col] == x_val_n_largest] if x_val_n_largest else data
    top_n = base_data.groupby(group_col)[y_col].sum().nlargest(n_largest).index
    return data[data[group_col].isin(top_n)]


def get_branch(branch_env_var: str | None) -> str:
    if branch_env_var:
        branch = os.getenv(branch_env_var)
        if branch is not None:
            return branch
    try:
        return f"local/{os.getlogin()}"
    except OSError:
        branch = os.getenv("USER", os.getenv("USERNAME"))
        return f"local/{branch}" if branch is not None else "local"
