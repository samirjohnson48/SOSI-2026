/** @type {import('d3')} */
const d3 = window.d3;

import { getStatusByNumber, getStatusByLandings, getCaptureProduction, getTopSpeciesProduction } from "./stats.js";

const elements = {
  tooltip: null,
  statsContainer: null,
  statsToggle: null,
  statsHeader: null
}

const mako10 = [
  "#0b0405", // 0: Abyssal Black/Purple
  "#1f102a", // 1: Deep Plum
  "#35193e", // 2: Dark Grape
  "#4d2b67", // 3: Deep Indigo
  "#4c5b96", // 4: Ocean Blue
  "#3f87a6", // 5: Steel Teal
  "#35b7ab", // 6: Mako Teal
  "#46cea1", // 7: Bright Aquamarine
  "#80d681", // 8: Shallow Sea Green
  "#def49c"  // 9: Luminous Seafoam
];

const unitMap = {
  1: "T",
  1e-3: "KT",
  1e-6: "MT"
};

const formatNum = (n, d) => Number.isInteger(n) ? String(n) : n.toFixed(d);

/**
 * @param {object} data
 * @param {string} containerLabel
 * @param {string} title
*/
const checkDataAndFlag = (data, containerLabel, title) => {
  if (!data || Object.keys(data).length === 0) {
    d3.select(containerLabel)
      .append("div")
      .text(`No data available for ${title}.`)
      .attr("class", "charts-not-available");
    return false;
  }
  return true;
}

export function initUI() {
  elements.tooltip = d3.select("body").append("div").attr("class", "tooltip");
  elements.statsContainer = document.querySelector('#stats-container');
  elements.statsToggle = document.querySelector("#stats-toggle");
  elements.statsHeader = document.querySelector("#stats-header");
  elements.charts = document.querySelector("#charts");

  elements.statsToggle.classList.add("hidden");

  elements.statsToggle?.addEventListener("click", () => {
    elements.statsContainer.classList.remove("active");
    elements.statsToggle.classList.add("hidden");
    
    d3.select("#charts").selectAll(":scope > *").html("");
  })
}

/**
 * @param {string} body
 * @param {number} x
 * @param {number} y
*/
export function showTooltip(body, x, y) {
  elements.tooltip.style("opacity", 1)
    .html(body) 
    .style("left", (x + 10) + "px")
    .style("top", (y - 28) + "px");
}

export function hideTooltip() {
  elements.tooltip?.style("opacity", 0);
}


export async function populateStats(props) {
  if (!elements.statsContainer) {
    console.error("UI Error: #stats-container not found.")
    return;
  }
  if ("code" in props && props.code === "18") {
    // No data for Area 18
    return;
  }

  // Clear charts
  d3.select("#charts").selectAll(":scope > *").html("");

  // Move into view
  elements.statsToggle.classList.remove("hidden");
  elements.statsContainer.classList.add("active");
  elements.statsHeader.textContent = props.header;

  // Render status charts
  const sbn = await getStatusByNumber(props.sosi_grouping);
  const sbl = await getStatusByLandings(props.sosi_grouping);
  const statusContainer = "#status-charts";
  renderBarChart(sbn, statusContainer, "Status by Number", "Count");
  renderBarChart(sbl, statusContainer, "Status by Landings", "Landings", "MT");

  // Render capture production figure
  const prodScale = 1e-6;
  const cap = await getCaptureProduction(props.sosi_grouping, prodScale);
  const capContainer = "#capture-charts";
  renderLinePlot(cap, capContainer, "Total Capture Production", "Year", "Landings", unitMap[prodScale]);

  // Render top species production figure
  const n_species = 10;
  const speciesScale = 1e-3;
  const topSpeciesProd = await getTopSpeciesProduction(props.sosi_grouping, n_species, speciesScale);
  const topSpeciesContainer = "#top-species-charts"
  renderAreaChart(topSpeciesProd, topSpeciesContainer, `Top Species Capture Production`, "Year", "Landings", unitMap[speciesScale])
}

/**
 * Renders a D3 bar chart based on data
 * @param {Object} data - Dictionary of metric: count pairs
 * @param {string} containerLabel - container to add charts to
 * @param {string} title - Title of chart
 * @param {string} metric - metric being plotted
*/
function renderBarChart(data, containerLabel, title, metric, unit = null) {
  const dataIsValid = checkDataAndFlag(data, containerLabel, title);
  if (!dataIsValid) { return; }

  const container = d3.select(containerLabel);

  const totalValue = d3.sum(data, d => d.y);

  const width = 300;
  const height = 400;
  const margin = {
    t: 50,
    r: 50,
    b: 30,
    l: 50,
  };
  
  const x = d3.scaleBand()
    .domain(data.map(d => d.x))
    .range([margin.l, width - margin.r])
    .padding(0.1);

  const y = d3.scaleLinear()
    .domain([0, d3.max(data, d => d.y)])
    .range([height - margin.b, margin.t]);

  const yPercent = d3.scaleLinear()
    .domain([0, d3.max(data, d => (d.y / totalValue) * 100)])
    .range([height - margin.b, margin.t]);

  const svg = container.append("svg")
    .attr("class", "bar-chart")
    .attr("width", width)
    .attr("height", height)
    .attr("viewBox", [0, 0, width, height])

  // X axis
  svg.append("g")
    .attr("transform", `translate(0,${height - margin.b})`)
    .call(d3.axisBottom(x).tickSizeOuter(0));

  // Y axis (count)
  svg.append("g")
    .attr("transform", `translate(${margin.l},0)`)
    .call(d3.axisLeft(y).ticks(5));

  // Y axis (percent)
  svg.append("g")
    .attr("transform", `translate(${width - margin.r},0)`)
    .call(d3.axisRight(yPercent)
      .ticks(5)
      .tickFormat(d => d + "%"));

  // Title
  svg.append("text")
    .attr("class", "chart-title")
    .attr("x", width / 2)
    .attr("y", margin.t / 4)
    .text(title);

  // Add y-axis label
  const yLabel = unit === null ? metric : `${metric} (${unit})`
  svg.append("g")
    .attr("transform", `translate(${margin.l},0)`)
    .call(g => g.append("text")
      .attr("x", 0)
      .attr("y", margin.t * 0.8)
      .attr("fill", "white")
      .attr("text-anchor", "middle")
      .attr("alignment-baseline", "middle")
      .attr("font-size", "0.8em")
      .text(yLabel));

  // Add bars
  svg.append("g")
    .selectAll()
    .data(data)
    .join("rect")
      .attr("class", "stats-bar")
      .attr("x", (d) => x(d.x))
      .attr("y", (_) => y(0))
      .attr("height", 0)
      .attr("width", x.bandwidth());

  // Transition
  svg.selectAll("rect")
    .transition()
    .duration(1000)
    .attr("y", (d) => y(d.y))
    .attr("height", (d) => y(0) - y(d.y))
    .delay((_, i) => i*100);

  // Tooltip interactions
  svg.selectAll("rect")
    .on("mouseover", (event, d) => {
      const pc = d.y / totalValue * 100;
      var tooltipBody = `<strong>${metric}:</strong> ${formatNum(d.y, 2)}`;
      if (unit) { tooltipBody += ` ${unit}`; }
      tooltipBody += ` (${pc.toFixed(2)}%)`;
      showTooltip(tooltipBody, event.clientX, event.clientY);
    }) 
    .on("mouseleave", hideTooltip)
}

/**
 *
 * @param {string} title
 * @param {string} xLabel
 * @param {string} yLabel
*/
function addTitleAndAxisLabels(svg, innerWidth, innerHeight, margin, title, xLabel, yLabel) {
  svg.append("text")
    .attr("class", "chart-title")
    .attr("x", innerWidth / 2)
    .attr("y", -margin.t / 2)
    .text(title);
  svg.append("text")
    .attr("class", "chart-axis-label")
    .attr("x", innerWidth / 2)
    .attr("y", innerHeight + margin.b)
    .text(xLabel);
  svg.append("text")
    .attr("class", "chart-axis-label")
    .attr("transform", "rotate(-90)")
    .attr("y", -2 * margin.l / 3)
    .attr("x", -innerHeight / 2 - margin.t)
    .text(yLabel);
}

/**
 * Renders a D3 line plot
 * @param {Array} data
 * @param {string} containerLabel - name of container for plot to be placed in
 * @param {string} title
 * @param {string} xLabel
 * @param {string} yLabel
 * @param {string} unit
*/
function renderLinePlot(data, containerLabel, title, xLabel, yLabel, unit = null) {
  const dataIsValid = checkDataAndFlag(data, containerLabel, title);
  if (!dataIsValid) { return; }

  const container = d3.select(containerLabel);
  const width = 800; 
  const height = 450;
  const margin = { t: 40, r: 30, b: 40, l: 60 };
  const innerWidth = width - margin.l - margin.r;
  const innerHeight = height - margin.t - margin.b;

  const svg = container.append("svg")
    .attr("width", width)
    .attr("height", height)
    .attr("viewBox", `0 0 ${width} ${height}`)
    .attr("preserveAspectRatio", "xMidYMid meet")
    .append("g")
    .attr("transform", `translate(${margin.l},${margin.t})`);

  const xScale = d3.scaleLinear()
    .domain(d3.extent(data, d => d.x))
    .range([0, innerWidth]);
  svg.append("g")
    .attr("transform", `translate(0,${innerHeight})`)
    .call(d3.axisBottom(xScale).tickFormat(d3.format("d")));

  const yScale = d3.scaleLinear()
    .domain([0, d3.max(data, d => d.y) * 1.1])
    .range([innerHeight, 0]);
  svg.append("g").call(d3.axisLeft(yScale).ticks(6));

  addTitleAndAxisLabels(svg, innerWidth, innerHeight, margin, title, xLabel, `${yLabel} (${unit})`);

  // Add data with transition
  const flatLineGenerator = d3.line()
    .x(d => xScale(d.x))
    .y(_ => innerHeight);

  const lineGenerator = d3.line()
    .x(d => xScale(d.x))
    .y(d => yScale(d.y));

  svg.append("path")
    .datum(data)
    .attr("class", "data-line")
    .attr("fill", "none")
    .attr("stroke", "steelblue")
    .attr("stroke-width", 2)
    .attr("d", flatLineGenerator)

  svg.selectAll(".data-line")
    .datum(data)
    .transition()
    .duration(1000)
    .ease(d3.easeCubicOut)
    .attr("d", lineGenerator)

  // Create tooltip interaction
  const overlay = svg.append("rect")
    .attr("class", "overlay")
    .attr("width", width)
    .attr("height", height)
    .style("fill", "none")
    .style("pointer-events", "all");

  const bisect = d3.bisector(d => d.x).left;

  const focusCircle = svg.append("g")
  .append("circle")
    .attr("class", "focus-circle")

  overlay
    .on("mousemove", (event) => {
      const [xCoord] = d3.pointer(event);
      const x0 = xScale.invert(xCoord);
      
      const i = bisect(data, x0, 1);
      const d0 = data[i - 1];
      const d1 = data[i];
      const d = d1 - x0 > x0 - d0 ? d1 : d0;

      let tooltipBody = `<strong>${xLabel}:</strong> ${d.x}<br/>`;
      tooltipBody += `<strong>${yLabel}:</strong> ${formatNum(d.y, 2)}`;
      if (unit) { tooltipBody += ` ${unit}`; };
      
      const xPos = xScale(d.x);
      const yPos = yScale(d.y);

      const matrix = svg.node().getScreenCTM();
      const globalX = xPos + margin.l + window.scrollX + matrix.e;
      const globalY = yPos + margin.t + window.scrollY + matrix.f;

      showTooltip(tooltipBody, globalX, globalY);
 
      focusCircle
        .attr("cx", xScale(d.x))
        .attr("cy", yScale(d.y))
        .style("opacity", 1);

    })
    .on("mouseleave", () => {
      hideTooltip();
      focusCircle.style("opacity", 0);
    });
}

/**
 * @param {Array} data
 * @param {string} containerLabel - name of container for plot to be placed in
 * @param {string} title
 * @param {string} xLabel
 * @param {string} yLabel
 * @param {string} unit
*/
function renderAreaChart(data, containerLabel, title, xLabel, yLabel, unit = null) {
  const dataIsValid = checkDataAndFlag(data, containerLabel, title);
  if (!dataIsValid) { return; }

  const container = d3.select(containerLabel);
  const width = 800; 
  const height = 450;
  const margin = { t: 40, r: 30, b: 40, l: 60 };
  const innerWidth = width - margin.l - margin.r;
  const innerHeight = height - margin.t - margin.b;

  const lastPoint = data[data.length - 1];
  const groups = Object.keys(lastPoint)
    .filter(k => k != "x")
    .sort((a, b) => lastPoint[b] - lastPoint[a]);
  const stackGenerator = d3.stack().keys(groups)
  const series = stackGenerator(data);

  const svg = container.append("svg")
    .attr("width", width)
    .attr("height", height)
    .attr("viewBox", `0 0 ${width} ${height}`)
    .attr("preserveAspectRatio", "xMidYMid meet")
    .append("g")
    .attr("transform", `translate(${margin.l},${margin.t})`);

  const xScale = d3.scaleLinear()
    .domain(d3.extent(data, d => d.x))
    .range([0, innerWidth]);
  svg.append("g")
    .attr("transform", `translate(0, ${innerHeight})`)
    .call(d3.axisBottom(xScale).tickFormat(d3.format("d")));

  const maxStackedValue = d3.max(series, layer => d3.max(layer, d => d[1]));
  const yScale = d3.scaleLinear()
    .domain([0, maxStackedValue * 1.1])
    .range([innerHeight, 0])
  svg.append("g").call(d3.axisLeft(yScale));

  addTitleAndAxisLabels(svg, innerWidth, innerHeight, margin, title, xLabel, `${yLabel} (${unit})`);

  const areaGenerator = d3.area()
    .x(d => xScale(d.data.x))
    .y0(d => yScale(d[0]))
    .y1(d => yScale(d[1]));

  const colorScale = d3.scaleOrdinal()
    .domain(groups)
    .range(mako10)
  
  const layerLabel = "stacked-area-layer";
  const layerSelector = "." + layerLabel;
  svg
    .selectAll(layerSelector)
    .data(series)
    .enter()
    .append("path")
      .style("fill", d => colorScale(d.key))
      .attr("class", layerLabel)
      .attr("d", areaGenerator);

  // Legend
  const legend = container.append("div")
    .attr("class", "area-chart-legend");
  groups.forEach(g => {
    const legendItem = legend.append("div")
      .attr("class", "legend-item");

    legendItem.append("div")
      .attr("class", "legend-swatch")
      .style("background-color", colorScale(g));

    legendItem.append("span")
      .text(g);

    legendItem
      .on("mouseenter", () => {
        svg.selectAll(layerSelector)
          .classed("active", d => d.key === g)
          .classed("inactive", d => d.key !== g);
        legendItem.classed("active", true);
      })
      .on("mouseleave", () => {
        svg.selectAll(layerSelector)
          .classed("active", false)
          .classed("inactive", false);
        legendItem.classed("active", false);
      });
  })


  // Add tooltip interactions
  const overlay = svg.append("rect")
    .attr("class", "overlay")
    .attr("width", width)
    .attr("height", height)
    .style("fill", "none")
    .style("pointer-events", "all");

  const bisect = d3.bisector(d => d.x).left;

  const lineBisector = svg.append("rect")
    .attr("class", "line-bisector")
    .attr("height", innerHeight);

  overlay
    .on("mousemove", (event) => {
      const [xCoord] = d3.pointer(event);
      const x0 = xScale.invert(xCoord);
      
      const i = bisect(data, x0, 1);
      const d0 = data[i - 1];
      const d1 = data[i];
      const d = x0 - d0.x > d1.x - x0 ? d1 : d0;

      lineBisector
        .attr("x", xScale(d.x) - 1)
        .attr("y", 0)
        .style("opacity", 1);

      let tooltipBody = `<strong>${xLabel}:</strong> ${d.x}<br/>`;
      groups.forEach(group => {
        const y = d[group];
        const yStr = isNaN(y) ? "NA" : formatNum(y, 2);
        tooltipBody += `<strong>${group}:</strong> ${yStr}`;

        if (unit && !isNaN(y)) { tooltipBody += ` ${unit}`; }

        tooltipBody += "<br/>";
      });

      const rootSvg = d3.select(containerLabel).select("svg").node();

      let pt = rootSvg.createSVGPoint();
      pt.x = xScale(d.x);
      pt.y = 0;

      let globalPT = pt.matrixTransform(svg.node().getScreenCTM());

      showTooltip(tooltipBody, globalPT.x + window.scrollX, globalPT.y + window.scrollY);

    })
    .on("mouseleave", () => {
      hideTooltip();
      lineBisector.style("opacity", 0);
    });
}
