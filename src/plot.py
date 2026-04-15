"""
Class used to create plots for SOSI 2026
"""

import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from typing import Any, Literal
from pathlib import Path

from .utils import (
    find_key,
    remove_key,
    add_val,
    unique_vals,
    create_filter_query,
    join_tables,
    filter_top_n,
    is_step_enabled,
    wrap_text,
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
    SHOW_ALL_FLAG_DEFAULT = "ALL"
    CAPTION_LENGTH = 50
    CAPTION_FONT_SIZE = 7
    LEGEND_FONT_SIZE = 7

    FONTS = {
        "suptitle": {"size": 7.5, "name": "HelveticaNeue-CondensedBlack"},
        "title": {"size": 9, "name": "HelveticaNeue-CondensedBold"},
        "legend": {"size": 7, "name": "HelveticaNeue-CondensedBold"},
        "species_legend": {"size": 7, "name": "HelveticaNeue-Italic"},
        "default": {"size": 7, "name": "HelveticaNeue"},
        "ylabel": {"size": 7, "name": "HelveticaNeue-CondensedBold"},
        "caption": {"size": 9, "name": "HelveticaNeue-Condensed"},
        "averages": {"size": 7, "name": "HelveticaNeue-Condensed"},
    }

    FAO_AREAS = [21, 27, 31, 34, 37, 41, 47, 48, 51, 57, 58, 61, 67, 71, 77, 81, 87, 88]

    def __init__(
        self,
        tables: dict[str, dict[str, pd.DataFrame]],
        assessment_year: int,
        isscaap_to_exclude: list[int],
        species_to_exclude: list[str],
        fao_areas: list[int] = FAO_AREAS,
        fonts_path: Path | str | None = None,
    ):
        self.tables = tables
        self.ass_year = assessment_year
        self.isscaap_to_exclude = isscaap_to_exclude
        self.species_to_exclude = species_to_exclude

        if fonts_path is None:
            root = Path(__file__).resolve().parent.parent
            fonts_path = root / "fonts"
            self.fonts = self._load_fonts(fonts_path)

    def _load_fonts(
        self, fonts_path: Path | str, fonts_config: dict[str, dict[str, Any]] = FONTS
    ):
        if isinstance(fonts_path, str):
            fonts_path = Path(fonts_path).resolve()

        if not fonts_path.is_dir():
            raise FileNotFoundError(
                f"Please specify a valid directory for the fonts folder. {str(fonts_path)} is not a valid directory."
            )

        if "default" not in fonts_config:
            raise ValueError("Please specify 'default' font in font configuration")

        fonts = {}
        for font_name, font_info in fonts_config.items():
            fn = fonts_path / (
                font_info["name"] + "." + font_info.get("extension", "ttf")
            )
            if not fn.is_file():
                raise FileNotFoundError(
                    f"Cannot find {font_info['name']} for {font_name} in specified directory {str(fonts_path)}"
                )
            font = FontProperties(fname=fn, size=font_info.get("size", 7))
            fonts[font_name] = font

            if font_name == "default":
                self._configure_default_font(font)

        return fonts

    def _configure_default_font(self, default_font: FontProperties) -> None:
        plt.rcParams["font.family"] = default_font.get_family()
        plt.rcParams["font.style"] = default_font.get_style()
        plt.rcParams["font.weight"] = default_font.get_weight()
        plt.rcParams["font.size"] = default_font.get_size()

    def _get_font(self, font_name: str) -> FontProperties:
        font = self.fonts.get(font_name)
        return font if font is not None else self.fonts["default"]

    def _create_subplot(
        self,
        ax: Axes,
        input_table: pd.DataFrame,
        x_col: str,
        y_col: str,
        kind: PlotKind = "line",
        label: Any | None = None,
        join_table: pd.DataFrame | dict[str, pd.DataFrame] | None = None,
        join_key: str | list[str] | None = None,
        title: str | None = None,
        ylabel: str | None = None,
        filter_query: str | list[str] | None = None,
        y_scale: float | None = None,
        group_col: str | None = None,
        n_largest: int | None = None,
        x_val_n_largest: Any | None = None,
        legend_args: dict | None = None,
        plot_args: dict | None = None,
        grid_args: dict | None = None,
        sort_ascending: bool = True,
        label_peak: bool = False,
        xtick_interval: int | None = None,
        average_args: dict | None = None,
        caption_type: Literal["total_production", "top10_species"] | None = None,
    ):
        legend_args = legend_args or {}
        plot_args = plot_args or {}
        grid_args = grid_args or {"visible": False}
        average_args = average_args or {}

        base_data = input_table.copy()
        if join_table is not None and join_key is not None:
            base_data = join_tables(
                base_data, join_table, join_key, suffixes=("", "_x")
            )

        if filter_query:
            queries = [filter_query] if isinstance(filter_query, str) else filter_query
            for fq in queries:
                base_data = base_data.query(fq)

        if y_scale is not None:
            base_data[y_col] = base_data[y_col] * float(y_scale)

        if xtick_interval is not None:
            ax.set_xticks(
                range(min(base_data[x_col]), max(base_data[x_col]), xtick_interval)
            )

        if group_col is not None:
            if n_largest is not None:
                x_val = (
                    self.ass_year
                    if x_val_n_largest == "assessment_year"
                    else x_val_n_largest
                )
                data = filter_top_n(
                    base_data, group_col, y_col, n_largest, x_col, x_val
                )
            else:
                data = base_data.copy()

            pivoted = (
                data.groupby([x_col, group_col])[y_col].sum().unstack(fill_value=0)
            )

            sorted_cols = pivoted.sum().sort_values(ascending=sort_ascending).index
            pivoted = pivoted[sorted_cols]
            pivoted.plot(ax=ax, kind=kind, **plot_args)

            if "map_col" in legend_args:
                map_col = legend_args["map_col"]
                label_map = dict(zip(data[group_col], data[map_col]))
                handles, labels = ax.get_legend_handles_labels()
                new_labels = [wrap_text(label_map.get(l, l), 20) for l in labels]
                ax.legend(
                    handles=handles,
                    labels=new_labels,
                    prop=self._get_font("species_legend"),
                    **remove_key(legend_args, "map_col"),
                )
            else:
                ax.legend(prop=self._get_font("legend"), **legend_args)

            plot_data = pivoted
        else:
            plot_data = base_data.groupby(x_col)[y_col].sum()
            plot_data.plot(ax=ax, kind=kind, **plot_args)

        if label_peak:
            peak_x, peak_y = self._add_peak_label(ax, plot_data)

        if average_args:
            max_delta, mdi1, mdi2 = self._add_averages(ax, plot_data, **average_args)
            ax.legend(prop=self._get_font("legend"), **legend_args)

        if caption_type is not None:
            match caption_type:
                case "total_production":
                    assert label is not None
                    self._add_total_production_caption(
                        ax,
                        label,
                        peak_x,
                        peak_y,
                        max_delta,
                        (mdi1, mdi2),
                    )
                case "top10_species":
                    assert isinstance(plot_data, pd.DataFrame)
                    assert group_col is not None
                    self._add_top10_species_caption(
                        ax=ax,
                        plot_data=plot_data,
                        area_capture=base_data,
                        species_col=group_col,
                        production_col=y_col,
                        year_col=x_col,
                    )

        if title is not None:
            ax.set_title(title, fontproperties=self._get_font("title"))
        else:
            ax.set_title(ax.get_title(), fontproperties=self._get_font("title"))

        if ylabel is not None:
            ax.set_ylabel(ylabel, fontproperties=self._get_font("ylabel"))
        else:
            ax.set_ylabel(ax.get_ylabel(), fontproperties=self._get_font("ylabel"))

        ax.grid(**grid_args)
        return ax

    def _add_peak_label(
        self, ax: Axes, data: pd.Series | pd.DataFrame
    ) -> tuple[Any, float]:
        if isinstance(data, pd.DataFrame):
            total_series = data.sum(axis=1)
            peak_x = total_series.idxmax()
            peak_y = total_series.max()
        else:
            peak_x = data.idxmax()
            peak_y = data.max()

        ax.annotate(
            str(peak_x),
            xy=(float(peak_x), peak_y * 1.1),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            arrowprops=dict(arrowstyle="->", color="black", lw=1),
        )

        return peak_x, peak_y

    def _add_averages(
        self,
        ax: Axes,
        data: pd.Series | pd.DataFrame,
        window: int,
        legend_title: str | None = None,
        unit: str = "MT",
    ) -> tuple[float, tuple[int, int], tuple[int, int]]:
        l, h = min(data.index), max(data.index)
        aves = []
        intervals = [(i, min(i + window, h)) for i in range(l, h, window)]
        max_delta, max_delta_idx = 0, 0
        for idx, (i, j) in enumerate(intervals):
            ave = data.loc[i:j].mean()
            aves.append(ave)

            if idx > 0:
                if aves[idx - 1] == 0:
                    max_delta = np.inf
                    max_delta_idx = idx
                else:
                    p_delta = 100 * abs(ave - aves[idx - 1]) / aves[idx - 1]
                    if p_delta > max_delta:
                        max_delta = p_delta
                        max_delta_idx = idx

            label = legend_title if (legend_title is not None and i == l) else ""
            ax.hlines(
                y=ave,
                xmin=i,
                xmax=j,
                color="grey",
                linestyles="--",
                alpha=0.9,
                linewidths=0.8,
                label=label,
            )
        if legend_title is not None:
            ave_label = ", ".join(
                [f"{d[0]}-{d[1] - 1}: {p:.2f} {unit}" for d, p in zip(intervals, aves)]
            )
            ax.text(
                0.12,
                -0.18,
                s=wrap_text(ave_label, 42),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontproperties=self._get_font("averages"),
            )

        return max_delta, intervals[max_delta_idx - 1], intervals[max_delta_idx]

    def _add_total_production_caption(
        self,
        ax: Axes,
        val: str,
        peak_x: int,
        peak_y: float,
        max_delta: float,
        max_delta_interval: tuple[tuple[int, int], tuple[int, int]],
    ) -> None:
        indicator = "a decrease" if max_delta < 0 else "an increase"
        caption = (
            f"Capture production in {val} peaked in {peak_x}, "
            f"with total landings of {peak_y:.2f} million tonnes. "
            f"The greatest change in mean production for the decade "
            f"occurred between {max_delta_interval[0][0]} - {max_delta_interval[0][1]} and "
            f"{max_delta_interval[1][0]} - {max_delta_interval[1][1]}, with {indicator} of "
            f"{max_delta:.2f} percent."
        )
        ax.text(
            0,
            -0.35,
            s=wrap_text(caption, self.CAPTION_LENGTH),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontproperties=self._get_font("caption"),
        )

    def _calculate_coverage(
        self,
        area_capture: pd.DataFrame,
        top10_species: list[str],
        species_col: str,
        production_col: str,
    ) -> float:
        mask = area_capture[species_col].isin(top10_species)

        p_cov = (
            100
            * area_capture.loc[mask, production_col].sum()
            / area_capture[production_col].sum()
        )
        return p_cov

    def _calculate_diversity(
        self,
        area_capture: pd.DataFrame,
        species_col: str,
        production_col: str,
        percent_coverage: int = 75,
    ):
        total_capture = area_capture[production_col].sum()

        species_ranked = (
            area_capture.groupby(species_col)
            .agg({production_col: "sum"})
            .reset_index()
            .sort_values(by=production_col, ascending=False)
        )

        cumulative_sum = species_ranked[production_col].cumsum()
        species_needed = (
            cumulative_sum <= total_capture * (percent_coverage / 100)
        ).sum() + 1

        return species_needed

    def _add_top10_species_caption(
        self,
        ax: Axes,
        plot_data: pd.DataFrame,
        area_capture: pd.DataFrame,
        species_col: str = "asfis_code",
        production_col: str = "production",
        year_col: str = "year",
    ) -> None:
        top10_species = list(plot_data.columns)
        p_cov = self._calculate_coverage(
            area_capture, top10_species, species_col, production_col
        )
        assessment_year = max(plot_data.index)
        year_mask = area_capture[year_col].eq(assessment_year)
        diversity = self._calculate_diversity(
            area_capture[year_mask], species_col, production_col
        )

        caption = (
            f"The top ten species accounted for {p_cov:.2f} percent "
            f"of the total capture production in {assessment_year}. "
            f"Seventy-five percent of the total capture production is covered "
            f"by the top {diversity} species. "
        )
        ax.text(
            0,
            -0.4,
            s=wrap_text(caption, self.CAPTION_LENGTH),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontproperties=self._get_font("caption"),
        )

    def _configure_subplot(
        self, ax: Axes, ax_args: dict, val: str | None = None, col: str | None = None
    ) -> None:
        input_table = find_key(self.tables, ax_args["input_table"])
        join_table = (
            find_key(self.tables, ax_args["join_table"])
            if "join_table" in ax_args
            else None
        )
        if col is not None and val is not None:
            val_filter = create_filter_query(col, val)
            ax_args = add_val(ax_args, "filter_query", val_filter)
        caption_type = (
            ax_args.pop("caption_type") if "caption_type" in ax_args else None
        )
        self._create_subplot(
            ax=ax,
            input_table=input_table,
            join_table=join_table,
            label=val,
            caption_type=caption_type,
            **remove_key(ax_args, ["input_table", "join_table"]),
        )

    def _scale_figsize(
        self,
        args: dict,
        unit_key: str = "figsize_unit",
        size_key: str = "figsize",
    ) -> dict:
        scales = {"mm": 25.4}
        if unit_key in args:
            unit = args.pop(unit_key)
            try:
                args[size_key] = tuple(s / scales[unit] for s in args[size_key])
            except KeyError:
                raise KeyError(
                    f"Unknown unit {unit} for figsize scaling. Acceptable units: {', '.join(scales.keys())}"
                )
        return args

    def _configure_suptitle(self, fig: Figure, args: dict | None, val: Any) -> None:
        if args is not None:
            st_args = args.copy()
            if "t" in args:
                st_args["t"] += f" for {val}"
            fig.suptitle(fontproperties=self._get_font("suptitle"), **st_args)

    def _setup_figure(
        self,
        params: dict,
        val: Any | None = None,
        col: str | None = None,
    ) -> Figure:
        subplots_args = params.get("subplots", {})
        subplots_args = self._scale_figsize(subplots_args)
        fig, axs = plt.subplots(**subplots_args)
        axs_flat = axs.flatten()
        suptitle_args = params.get("suptitle")
        self._configure_suptitle(fig, suptitle_args, val)
        for i, ax_args in enumerate(params["axs_params"].values()):
            self._configure_subplot(axs_flat[i], ax_args, val, col)
        fig.set_tight_layout(params.get("tight_layout", False))
        if "subplots_adjust" in params:
            fig.subplots_adjust(**params["subplots_adjust"])

        return fig

    def _show_figure(
        self, figures_to_show: str | list[str], figure_name: str, show_all_flag: str
    ) -> None:
        if is_step_enabled(figure_name, figures_to_show, show_all_flag):
            plt.show()

    def create_figure(
        self,
        figure_name: str,
        params: dict,
        figures_to_show: str | list[str] | None = None,
        show_all_flag: str = SHOW_ALL_FLAG_DEFAULT,
    ) -> Figure | dict[str, Figure]:
        if "groupby" not in params:
            fig = self._setup_figure(params)
            if figures_to_show is not None:
                self._show_figure(figures_to_show, figure_name, show_all_flag)

        figs = {}
        try:
            table, col = params["groupby"]["table"], params["groupby"]["col"]
        except KeyError:
            raise KeyError(
                f"Specify both 'table' and 'col' in 'groupby' section of figure {figure_name} params."
            )
        for val in unique_vals(find_key(self.tables, table), col, dropna=True):
            fig = self._setup_figure(params, val, col)
            figs[val] = fig
            if figures_to_show is not None:
                self._show_figure(figures_to_show, figure_name, show_all_flag)
            plt.close(fig)

        return figs
