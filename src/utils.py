"""
Utility functions
"""

import pandas as pd
import os
import argparse
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


def find_table(
    tables: dict[str, dict[str, pd.DataFrame]],
    args: dict,
    table_key: str,
    keys_to_remove: str | list[str] | None = None,
    pop_from_args: bool = False,
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    tables_to_search = (
        remove_key(tables, keys_to_remove)
        if keys_to_remove is not None
        else tables.copy()
    )

    table_name = args.pop(table_key) if pop_from_args else args.get(table_key)
    if table_name is None:
        raise KeyError(
            f"Incorrect table key: {table_key} from args {args}. Did you mean 'input_table' or 'join_table'?"
        )

    match table_name:
        case str():
            return find_key(tables_to_search, table_name)
        case list():
            return {tn: find_key(tables_to_search, tn) for tn in table_name}
        case _:
            raise TypeError(
                f"Unknown type for table name(s) {table_name} found by key {table_key} specified in args {args}"
            )


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
            return branch.replace("/", "_")
    try:
        return f"local/{os.getlogin().replace('/', '_')}"
    except OSError:
        branch = os.getenv("USER", os.getenv("USERNAME"))
        return f"local/{branch.replace('/', '_')}" if branch is not None else "local"


def parse_args_config(
    args_config: dict,
    all_flag_default: str = "ALL",
    all_flag_key: str = "all_flag",
    description: str = "SOSI 2026",
) -> tuple:
    parser = argparse.ArgumentParser(description=description)

    all_flag = (
        args_config.pop(all_flag_key)
        if all_flag_key in args_config
        else all_flag_default
    )

    for arg, arg_info in args_config.items():
        required = arg_info.pop("required", False)
        abbr = arg_info.pop("abbreviation", None)
        arg_name = arg if required else f"--{arg}"

        if abbr:
            parser.add_argument(arg_name, f"-{abbr}", **arg_info)
        else:
            parser.add_argument(arg_name, **arg_info)

    args = parser.parse_args()

    for k, v in vars(args).items():
        if v == []:
            vars(args)[k] = all_flag

    return (args, all_flag)


def is_step_enabled(
    step_id: str,
    selected_steps: str | list[str] | None,
    run_all_keyword: str,
) -> bool:
    if selected_steps is None:
        return False

    return selected_steps == run_all_keyword or step_id in selected_steps
