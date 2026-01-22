"""
Class used to create plots for SOSI 2026
"""

import pandas as pd
import logging
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import Any, Literal

from .utils import (
    find_key,
    remove_key,
    add_val,
    unique_vals,
    create_filter_query,
    join_tables,
    filter_top_n,
)

logger = logging.getLogger(__file__)

PlotKind = Literal[
    "line",
    "bar",
    "barh",
    "hist",
    "box",
    "kde",
    "density",
    "area",
    "pie",
    "scatter",
    "hexbin",
]


class SOSIPlotter:
    def __init__(
        self,
        tables: dict[str, dict[str, pd.DataFrame]],
        assessment_year: int,
        isscaap_to_exclude: list[int],
        species_to_exclude: list[str],
        show_figure: bool = False,
    ):
        self.tables = tables
        self.ass_year = assessment_year
        self.isscaap_to_exclude = isscaap_to_exclude
        self.species_to_exclude = species_to_exclude
        self.show_figure = show_figure

    def _create_subplot(
        self,
        ax: Axes,
        input_table: pd.DataFrame,
        x_col: str,
        y_col: str,
        kind: PlotKind = "line",
        join_table: pd.DataFrame | dict[str, pd.DataFrame] | None = None,
        join_key: str | list[str] | None = None,
        filter_query: str | list[str] | None = None,
        y_scale: float | None = None,
        group_col: str | None = None,
        n_largest: int | None = None,
        x_val_n_largest: Any | None = None,
        legend_args: dict = {},
        plot_args: dict = {},
        grid_args: dict = {"visible": False},
        sort_ascending: bool = True,
    ):
        if join_table is not None and join_key is not None:
            data = join_tables(input_table, join_table, join_key)
        else:
            data = input_table

        if filter_query is not None:
            if isinstance(filter_query, str):
                filter_query = [filter_query]
            for fq in filter_query:
                data = data.query(fq)

        if y_scale is not None:
            if isinstance(y_scale, str):
                y_scale = float(y_scale)
            data[y_col] *= y_scale

        if group_col is not None:
            if n_largest is not None:
                if x_val_n_largest == "assessment_year":
                    x_val_n_largest = self.ass_year
                data = filter_top_n(
                    data=data,
                    group_col=group_col,
                    y_col=y_col,
                    n_largest=n_largest,
                    x_col=x_col,
                    x_val_n_largest=x_val_n_largest,
                )
            grouped = data.groupby([group_col, x_col])[y_col].sum()
            pivoted = grouped.unstack(level=group_col)
            sorted_columns = pivoted.sum().sort_values(ascending=sort_ascending).index
            pivoted = pivoted[sorted_columns]
            pivoted.plot(ax=ax, kind=kind, **plot_args)

            if "map_col" in legend_args:
                handles, labels = ax.get_legend_handles_labels()
                label_map = {
                    k: v
                    for k, v in zip(data[group_col], data[legend_args.pop("map_col")])
                }
                new_labels = [label_map[l] for l in labels]
                legend_args["handles"] = handles
                legend_args["labels"] = new_labels

            ax.legend(**legend_args)
        else:
            grouped = data.groupby(x_col)[y_col].sum()
            grouped.plot(ax=ax, kind=kind, **plot_args)

        ax.grid(**grid_args)

        return ax

    def create_figure(
        self,
        params: dict,
    ) -> Figure | dict[str, Figure]:
        if "groupby" in params:
            figs = {}
            table, col = params["groupby"]["table"], params["groupby"]["col"]
            for val in unique_vals(
                find_key(self.tables, table),
                col,
            ):
                val_filter = create_filter_query(col=col, val=val)
                fig, axs = plt.subplots(**params.get("subplots_args", {}))
                suptitle_args = params.get("suptitle")
                if suptitle_args is not None:
                    st_args = suptitle_args.copy()
                    if "t" in suptitle_args:
                        st_args["t"] += f" for {val}"
                    fig.suptitle(**st_args)
                for i, ax_args in enumerate(params["axs_params"].values()):
                    input_table = find_key(self.tables, ax_args["input_table"])
                    join_table = (
                        find_key(self.tables, ax_args["join_table"])
                        if "join_table" in ax_args
                        else None
                    )
                    ax_args = add_val(ax_args, "filter_query", val_filter)
                    self._create_subplot(
                        ax=axs[i],
                        input_table=input_table,
                        join_table=join_table,
                        **remove_key(ax_args, ["input_table", "join_table"]),
                    )
                fig.set_tight_layout(params.get("tight_layout", False))
                if self.show_figure:
                    plt.show()
                figs[val] = fig
            return figs

        fig, axs = plt.subplots(**params.get("subplots_args", {}))
        for i, ax_args in enumerate(params["axs_params"].values()):
            input_table_name = ax_args.pop("input_table")
            input_table = find_key(self.tables, input_table_name)
            join_table = (
                find_key(self.tables, ax_args["join_table"])
                if "join_table" in ax_args
                else None
            )
            self._create_subplot(
                ax=axs[i], input_table=input_table, join_table=join_table, **ax_args
            )
        return fig
