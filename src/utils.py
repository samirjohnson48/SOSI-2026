"""
Utility functions
"""

import pandas as pd
import numpy as np
import os
import argparse
from typing import Any, Literal, TypeVar
from textwrap import wrap

type MergeHow = Literal["left", "right", "outer", "inner", "cross"]
K = TypeVar("K")
V = TypeVar("V")


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


def sort_dict(d: dict[K, V], order: list[K]) -> dict[K, V]:
    priority = {key: i for i, key in enumerate(order)}
    default = len(priority)
    return {k: d[k] for k in sorted(d, key=lambda x: priority.get(x, default))}


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


def unique_vals(
    df: pd.DataFrame,
    col: str | list[str],
    dropna: bool = True,
    vals_to_exclude: Any | list[Any] | None = None,
) -> list:
    base_df = df.dropna() if dropna else df
    if isinstance(col, str):
        u_vals = list(base_df[col].unique())
        if vals_to_exclude is not None:
            vte = set(vals_to_exclude)
            uv = set(u_vals)
            return list(uv - vte)
    return list(base_df[col].drop_duplicates().values)


def join_tables(
    df: pd.DataFrame,
    join_table: pd.DataFrame | dict[str, pd.DataFrame],
    join_key: str | list[str] | dict[str, list[str] | str],
    how: MergeHow = "left",
    suffixes: tuple[str, str] = ("_x", "_y"),
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
            result = pd.merge(
                result, table, on=join_key[table_name], how=how, suffixes=suffixes
            )
    elif isinstance(join_table, pd.DataFrame) and isinstance(join_key, (str, list)):
        result = pd.merge(df, join_table, on=join_key, how=how, suffixes=suffixes)

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

    args, _ = parser.parse_known_args()

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


def resolve_config_variables(obj: Any, ax_args: dict) -> dict:
    for key, value in ax_args.items():
        if isinstance(value, str) and value.startswith("@self."):
            attr_name = value.replace("@self.", "")
            ax_args[key] = getattr(obj, attr_name, value)
        elif isinstance(value, dict):
            ax_args[key] = resolve_config_variables(obj, value)
    return ax_args


def wrap_text(text: str, width: int, wrap_char: str = "\n") -> str:
    return wrap_char.join(wrap(text, width=width))


def broadcast_df(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    level: int | str,
    operation: Literal["add", "subtract"],
    set_level_vals: dict[str, str] | None = None,
    fill_value: int | float = 0,
) -> pd.DataFrame:
    temp = df2.groupby(level=level).sum()
    temp_broadcasted = temp.reindex(df1.index, level=level)

    result = getattr(df1, operation)(temp_broadcasted, fill_value=fill_value)

    if set_level_vals and isinstance(result.index, pd.MultiIndex):
        idx_df = result.index.to_frame()
        for lvl_name, new_val in set_level_vals.items():
            if lvl_name in idx_df.columns:
                idx_df[lvl_name] = new_val
        result.index = pd.MultiIndex.from_frame(idx_df)

    return result


def _get_fill_val(s: pd.Series) -> int | str:
    return 0 if pd.api.types.is_numeric_dtype(s.dtype) else ""


def sort_df(
    df: pd.DataFrame,
    order: dict[Any, int],
    level: int | str | None = None,
    col: Any | None = None,
    sort_by: str | list[str] | None = None,
) -> pd.DataFrame:
    if level is None and col is None:
        raise ValueError("Must specify either level or col to identify 'order' rows.")

    label_values = df.index.get_level_values(level) if level is not None else df[col]

    primary_key = pd.Series(label_values).map(lambda x: order.get(x, 0)).to_numpy()

    keys_to_sort = []
    if sort_by is not None:
        if isinstance(sort_by, str):
            sort_by = [sort_by]
        for name in reversed(sort_by):
            if name in df.index.names:
                v = df.index.get_level_values(name)
            elif name in df.columns:
                v = df[name]
            else:
                raise KeyError(f"'{name}' not found in index levels or columns.")

            v_series = pd.Series(v)
            keys_to_sort.append(pd.Series(v).fillna(_get_fill_val(v_series)).to_numpy())
    else:
        v_series = pd.Series(label_values)
        keys_to_sort.append(v_series.fillna(_get_fill_val(v_series)).to_numpy())

    keys_to_sort.append(primary_key)

    indexer = np.lexsort(keys_to_sort)
    return df.iloc[indexer]


def split_by_suffix(
    data_map: dict[str, pd.DataFrame], suffix: str
) -> (
    tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]
    | tuple[pd.DataFrame, pd.DataFrame]
):
    """
    Splits a dictionary into two based on a suffix,
    sharing keys that do not have a suffixed counterpart.
    """
    suffixed_data = {}
    base_data = {}

    suffixed_keys = {k for k in data_map if k.endswith(suffix)}
    suffix_mapping = {k.removesuffix(suffix): k for k in suffixed_keys}

    for key, df in data_map.items():
        if key in suffixed_keys:
            suffixed_data[key] = df
        elif key in suffix_mapping:
            base_data[key] = df
        else:
            base_data[key] = df
            suffixed_data[key] = df

    if len(base_data) == 1 and len(suffixed_data) == 1:
        return list(base_data.values())[0], list(suffixed_data.values())[0]

    return base_data, suffixed_data


def make_series_unique(s: pd.Series) -> pd.Series:
    """
    Appends a count to duplicate strings (e.g., 'name', 'name 1', 'name 2').
    """
    counts = s.groupby(s).cumcount()
    suffix = counts.map(lambda x: f" {x}" if x > 0 else "")
    return s.astype(str) + suffix


def order_columns(
    df: pd.DataFrame, value_order: list[Any] | None = None, total_label: str = "Total"
) -> pd.DataFrame:
    """
    Sets the innermost column level to a Categorical type to enforce
    total_label first, followed by value_order.
    """
    assert isinstance(df.columns, pd.MultiIndex)

    inner_level_values = df.columns.get_level_values(-1).unique()
    if value_order is None:
        value_order = [v for v in inner_level_values if v != total_label]

    full_order = [total_label] + list(value_order)

    new_levels = [
        pd.Categorical(level, categories=full_order, ordered=True)
        if i == len(df.columns.levels) - 1
        else level
        for i, level in enumerate(df.columns.levels)
    ]

    df.columns = df.columns.set_levels(new_levels)

    return df.sort_index(axis=1)


def find_by_attribute(
    module: Any, attr_name: str, get_attr_name: bool = False
) -> list[str]:
    return [
        getattr(obj, attr_name) if get_attr_name else name
        for name, obj in vars(module).items()
        if hasattr(obj, attr_name)
    ]
