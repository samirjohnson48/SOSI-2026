/** @type {import('d3')} */
const d3 = window.d3;

import { showTooltip, hideTooltip, populateStats } from "./ui.js";

const faoAreasMapPath = "/static/assets/geojson/fao_areas_mod.json";
const countriesMapPath = "/static/assets/geojson/countries.json";

const specialGroups = {
  "Atlantic Salmon": "Atl.Salmon",
  "Billfish": "Billfish",
  "Pacific Salmon": "Pac.Salmon",
  "Pelagic Sharks": "Pel.Sharks",
  "Tuna": "Tuna"
};

const mapContainer = document.querySelector('#map-container');
const width = mapContainer.clientWidth;
const height = mapContainer.clientHeight;

const loader = document.querySelector(".loader");

const svg = d3.select("#map")
  .attr("viewBox", [0, 0, width, height])
  .attr("preserveAspectRatio", "xMidYMid meet");
const mapGroup = svg.append("g");
const projection = d3.geoNaturalEarth1()
    .scale(width / 2 / Math.PI)
    .translate([width / 2, height / 2]);
const path = d3.geoPath().projection(projection);
const zoom = d3.zoom()
  .scaleExtent([1,40])
  .translateExtent([[0, 0], [width, height]])
  .on("zoom", (event) => {
    mapGroup.attr("transform", event.transform);
  });


export async function loadMap() {
  try {
    /** @type {[any, any]} */
    const [countriesData, faoAreasData] = await Promise.all(
      [
        d3.json(countriesMapPath),
        d3.json(faoAreasMapPath),
      ]
    );
    
    /** @type {import('geojson').FeatureCollection} */
    const countries = countriesData;
    /** @type {import('geojson').FeatureCollection} */
    const faoAreas = faoAreasData;

    mapGroup.selectAll(".country")
      .data(countries.features).enter()
      .append("path")
      .attr("d", path)
      .attr("class", "country")
      .on("mouseover", (event, d) => {
        const tooltipBody = `<strong>${d.properties.TERR_NAME}</strong>`;
        showTooltip(tooltipBody, event.clientX, event.clientY);
      })
      .on("mouseleave", hideTooltip);

    mapGroup.selectAll(".fao-area")
      .data(faoAreas.features).enter()
      .append("path")
      .attr("d", path)
      .attr("class", "fao-area")
      .on("mouseover", (event, d) => {
        const tooltipBody = `<strong>${d.properties.F_NAME}</strong> (Area ${d.properties.F_CODE})`;
        showTooltip(tooltipBody, event.clientX, event.clientY)
      }) 
      .on("mouseleave", hideTooltip)
      .on("click", (_, d) => {
        const code = d.properties.F_CODE;
        const name = d.properties.F_NAME;
        const props = {
          "code": code,
          "header": `Area ${code}: ${name}`,
          "sosi_grouping": `Area ${code}`
        }
        populateStats(props);
      });

    loader.classList.add("hidden");

    addSpecialGroups();

    svg.call(zoom);
  }
  catch (error) {
    console.error("An error occurred when loading the map data", error.message);
  }
}

function addSpecialGroups() {
  const names = Object.keys(specialGroups);

  names.forEach((name) => {
    const props = {
      "header": name,
      "sosi_grouping": specialGroups[name]
    };

    d3.select("#special-groups-container")
      .append("div")
      .attr("class", "special-groups-button")
      .text(name)
      .on("click", (_, __) => populateStats(props));
  })
}

