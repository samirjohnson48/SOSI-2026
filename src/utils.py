"""
Utility functions
"""

import pandas as pd
import os
from typing import Any


def find_key(d: dict, key: str) -> Any:
    """
    Find key in nested dictionary recursively
    Used for extract tables from data vault based on table name
    """
    if key in d.keys():
        return d[key]

    for v in d.values():
        if not isinstance(v, dict):
            continue
        val = find_key(v, key)
        if val is not None:
            return val


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
