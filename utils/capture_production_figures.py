"""

"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import inflect
from textwrap import fill

def compute_top_species_by_area(fishstat, file_path, year=2023, num_species=10):
    # Identify top 10 species for each area by production
    top_species = (
        fishstat.groupby(["Area", "ASFIS Scientific Name"])
        .sum()
        .reset_index()
        .sort_values(by=year, ascending=False)
        .groupby("Area")[["Area", "ASFIS Scientific Name", year]]
    )
    
    # Save data to file_path
    # Save as one file with an individual sheet per area
    with pd.ExcelWriter(file_path) as writer:
        for area, group in top_species:
            group = group.drop(columns="Area").reset_index(drop=True).head(num_species)
            if isinstance(area, float):
                area = int(area)
            group.to_excel(writer, sheet_name=str(area), index=False)
    
    print(f"Top 10 species data saved to {file_path}")
    
    # Convert to lists
    top_species_by_area = (
        top_species
        .apply(lambda group: group.nlargest(num_species, year)["ASFIS Scientific Name"].tolist())
        .reset_index(name="top_species")
    )
    
    return top_species_by_area

def list_to_text(values, round=None):
    if len(values) == 0:
        return ""

    # If rounding is needed, apply rounding to each value in the list
    if round is not None:
        values = [f"{value:.{round}f}" for value in values]
    else:
        values = [str(value) for value in values]

    if len(values) == 1:
        return values[0]
    elif len(values) == 2:
        return f"{values[0]} and {values[1]}"
    else:
        return f"{', '.join(values[:-1])}, and {values[-1]}"
    
def calculate_diversity(area, fishstat, year=2023, percent_coverage=75):
    total_capture = fishstat[fishstat["Area"]==area][year].sum()
    
    species_ranked = fishstat[fishstat["Area"]==area].groupby("ASFIS Scientific Name").agg(
        {year: "sum"}
    ).reset_index().sort_values(by=year, ascending=False)
    
    cumulative_sum = species_ranked[year].cumsum()
    species_needed = (cumulative_sum <= total_capture * (percent_coverage / 100)).sum() + 1
    
    return species_needed


def figure_summary(area, production_aves, capture_peaks, species_percentage, species_needed, first_year=1950, last_year=2023, last_year_dec=2020, percent_coverage=75):    
    p = inflect.engine()
    species_needed_text = p.number_to_words(species_needed)
    species_summary = f"The top ten species accounted for {species_percentage:.2f} percent of total capture production in {last_year}. "
    pc_text = p.number_to_words(percent_coverage)
    pc_text = pc_text[0].upper() + pc_text[1:]
    species_summary += f"{pc_text} percent of the total capture production is covered by the top {species_needed_text} species."

    # Document the peaks with capture production levels in millions of tonnes
    total_summary = f"Area {area} had its peak " + \
    f"in capture production in {list_to_text(capture_peaks.keys())}," + \
    f"with total landings of {list_to_text(capture_peaks.values(), round=2)} million tonnes. "
    
    percent_changes = [(production_aves[i+1] - production_aves[i]) / (production_aves[i]) * 100  if production_aves[i] > 0 else 0 for i in range(len(production_aves)-1)]
    decades = [(y, y+9) for y in range(first_year, last_year_dec, 10)] + [(last_year_dec, last_year)]
    max_pc = percent_changes.index(max([abs(p) for p in percent_changes]))
    total_summary += f"The greatest change in mean production for the decade occured between {decades[max_pc][0]}-{decades[max_pc][1]} " + \
    f"and {decades[max_pc+1][0]}-{decades[max_pc+1][1]}, with a{" decrease" if percent_changes[max_pc] < 0 else "n increase"} of {percent_changes[max_pc]:.2f} percent."

    return total_summary, species_summary
    
def create_capture_production_figure(area, capture_by_area, top_species, fishstat, output_dir, first_year=1950, last_year=2023, last_year_dec=2020, percent_coverage=75):
    # Set dimensions for figure
    mm_to_inches = 25.4
    width_to_height = 1
    w = 154 / mm_to_inches
    h = w / width_to_height
    
    font_dir = os.path.join(os.getcwd(), "input", "fonts")
    
    # Create two side-by-side plots
    fig, axs = plt.subplots(1, 2, figsize=(w, h), gridspec_kw={"width_ratios": [1,1]})
    title_s = "S" if isinstance(area, str) else ""
    suptitle_text = f"CAPTURE PRODUCTION ANALYSIS FOR AREA{title_s} {area}"
    suptitle_font = FontProperties(fname=os.path.join(font_dir, "HelveticaNeue-CondensedBlack.ttf"), size=7.5)
    
    fig.suptitle(
        suptitle_text,
        x=0,
        fontproperties=suptitle_font
    )

    # --- Plot 1: Total Production Over Time ---
    production_ts = (
        capture_by_area[capture_by_area["Area"] == area].drop("Area", axis=1).values[0]
        / 1e6
    ) # Total area production in Mt

    cp_area = {}
    cp_area["total"] = production_ts
    dec_ave = {}
    
    years = list(range(first_year, last_year + 1))
    
    cps = {}
    
    # Plot total production
    axs[0].plot(
        years, production_ts, color="black", linewidth=1, label="Total Production"
    )
    
    # Add decade averages to total production plot
    decades = [(y, y+10) for y in range(first_year, last_year_dec, 10)] + [(last_year_dec, last_year + 1)]
    production_aves = []
    
    for i, decade in enumerate(decades):
        # Note: list slice [i:j] retrieves values i, ..., j-1
        # Thus, data from decade is taken from 1950-1959, 1960-1969,..., i.e. not double counted
        production_ave = production_ts[decade[0]-first_year:decade[1]-first_year].mean()
        production_aves.append(production_ave)
        dec_ave[f"{decade[0]}-{decade[1]-1}"] = production_ave
        xmax = decade[1] - 1 if decade[1] == last_year + 1 else decade[1]
        label = "Mean Production for Decade" if i == 0 else None
        axs[0].hlines(y=production_ave, xmin=decade[0], xmax=xmax, 
                      color="grey", linestyles="--", alpha=0.9, linewidths=0.8, label=label)
    mean_prod_label = ", ".join([f"{d[0]}-{d[1]-1}: {p:.2f} Mt" for d,p in zip(decades, production_aves)])
    mean_prod_label = fill(mean_prod_label, 42)
    
    # Label max value peak
    max_idx = np.argmax(production_ts)
    max_value = production_ts[max_idx]
    
    default_font = FontProperties(fname=os.path.join(font_dir, "HelveticaNeue.ttf"), size=7)

    axs[0].annotate(
        f"{years[max_idx]}",
        xy=(years[max_idx], max_value),
        xytext=(years[max_idx], max_value * 1.09),
        arrowprops=dict(arrowstyle="->", color="black", lw=1),
        fontproperties=default_font,
        ha="center",
        color="black",
    )
    cps[years[max_idx]] = max_value
    
    title_font = FontProperties(fname=os.path.join(font_dir, "HelveticaNeue-CondensedBold.ttf"), size=9)

    axs[0].set_title("Total Capture Production", x=0.3, fontproperties=title_font)  
    y_min, y_max = axs[0].get_ylim()
    axs[0].set_ylim(y_min, y_max * 1.1)
    
    yaxis_font = FontProperties(fname=os.path.join(font_dir, "HelveticaNeue-CondensedBold.ttf"), size=7)
    
    axs[0].set_ylabel("PRODUCTION (MILLION TONNES)", fontproperties=yaxis_font)

    for label in axs[0].get_xticklabels() + axs[0].get_yticklabels():
        label.set_fontproperties(default_font)
    
    axs[0].set_xticks(np.arange(first_year, last_year, 10))
    
    leg0_font = yaxis_font   
    axs[0].legend(fontsize=7, bbox_to_anchor=(0, -0.3), loc="lower left", frameon=False, labelspacing=1, handlelength=1.5, handleheight=0.25, prop=leg0_font)
    
    axs[0].grid(True, linestyle="--", alpha=0.4)
    axs[0].set_box_aspect(1)

    # --- Plot 2: Top Ten Species Production ---
    area_mask = fishstat["Area"] == area
    
    for species in top_species:
        species_production_ts = (
            fishstat[
                (fishstat["ASFIS Scientific Name"] == species) & area_mask
            ][["ASFIS Scientific Name"] + years]
            .groupby("ASFIS Scientific Name")
            .sum()
            .values[0]
            / 1e6
        )
        cp_area[species] = species_production_ts
        
    stackplot_dict = {k:v for k, v in sorted(cp_area.items(), key=lambda item: np.sum(item[1])) if k!="total" and "-" not in k}
    labels = [fill(l, 20) for l in stackplot_dict.keys()]
    
    axs[1].set_title("Top Ten Species Production", x=0.35, fontproperties=title_font)
    axs[1].stackplot(years, stackplot_dict.values(), labels=labels)
    axs[1].set_ylabel("TOTAL PRODUCTION (MILLION TONNES)", fontproperties=yaxis_font)
    
    for label in axs[1].get_xticklabels() + axs[1].get_yticklabels():
        label.set_fontproperties(default_font)
    
    axs[1].set_xticks(np.arange(first_year, last_year, 10))
    
    sp_legend_font = FontProperties(fname=os.path.join(font_dir, "HelveticaNeue-Italic.ttf"), size=7)
    
    axs[1].legend(loc="upper left", ncol=2, bbox_to_anchor=(-0.085, -0.1), frameon=False, labelspacing=0.75, handlelength=0.6, prop=sp_legend_font)
    
    axs[1].grid(True, linestyle="--", alpha=0.4)
    axs[1].set_box_aspect(1)
        
    # Calculate the percent coverage of top ten species
    total_landings = production_ts[last_year - first_year]
    top_species_mask = fishstat["ASFIS Scientific Name"].isin(top_species)
    species_percentage = (fishstat[top_species_mask & area_mask][last_year].sum() / 1e6) / total_landings * 100
    
    # Calculate species needed for 75% coverage
    species_needed = calculate_diversity(area, fishstat, year=last_year, percent_coverage=percent_coverage)
    
    # Attach figure summary
    total_summary, species_summary = figure_summary(area, production_aves, cps, species_percentage, species_needed, percent_coverage=percent_coverage)
    
    mean_prod_font = FontProperties(fname=os.path.join(font_dir, "HelveticaNeue-Condensed.ttf"), size=7)
    
    fig.text(0.15, -0.32, mean_prod_label, ha="left", va="top", transform=axs[0].transAxes, fontproperties=mean_prod_font, linespacing=2)
    
    caption_font = FontProperties(fname=os.path.join(font_dir, "HelveticaNeue-Condensed.ttf"), size=9)
    
    fig.text(-0.1, -0.7, fill(total_summary, 50), ha="left", va="top", transform=axs[0].transAxes, fontproperties=caption_font)
    fig.text(1.25, -0.7, fill(species_summary, 50), ha="left", va="top", transform=axs[0].transAxes, fontproperties=caption_font)
    
    box0 = axs[0].get_position()
    box1 = axs[1].get_position()
    axs[0].set_position([box0.x0 - 0.25, box0.y0 + 0.25, box0.width, box0.height])
    axs[1].set_position([box1.x0 - 0.2, box1.y0 + 0.25, box1.width, box1.height])
    
        
    # Save figure
    file_path = os.path.join(output_dir, f"capture_production_species_area_{area}.pdf")
    fig.savefig(file_path, bbox_inches="tight", dpi=300)
        
    return cp_area, dec_ave, cps