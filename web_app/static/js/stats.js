import { getChartData } from "./api.js";

const stocksFlag = "stocks";
const captureFlag = "capture";
const validStatus = ["U", "M", "O"];

/**
 * Reformats the data for the charts in ui.js 
 * @param {Array} data - data received from API 
 * @param {string} metric - name of metric of query, e.g. 'status'
 * @param {string} value - name of value of query, e.g. 'count' or 'landings'
 * @param {number} scale - scale the value, default 1
*/
const formatData = (data, metric, value, scale = 1) =>
  data?.map(d => ({x: d[metric], y: d[value] * scale}));


/**
 * Reformats the data for the charts in ui.js -- used for grouped data
 * @param {Object[]} data - data received from API 
 * @param {string} grouping - column by which the query is grouped, e.g. 'asfis_code', 'sosi_edition'
 * @param {string} metric - name of metric of query, e.g. 'status'
 * @param {string} value - name of value of query, e.g. 'count' or 'landings'
 * @param {number} scale - scale the value, default 1
*/
const formatGroupedData = (data, grouping, metric, value, scale = 1) => {
  if (!data) return {};

  const groupedData = {};
  data.forEach(d => {
    const key = d[metric];
    const groupKey = d[grouping];

    if (!groupedData[key]) { groupedData[key] = {}; };

    groupedData[key][groupKey] = d[value] * scale;
  });

  return Object.entries(groupedData)
    .map(([k, v]) => ({x: isNaN(k) ? k : +k, ...v}))
    .sort((a, b) => a.x - b.x);
}

/**
 * Computes the status counts for a sosi grouping
 * @param {string} sosi_grouping - e.g. 'Area 21' or 'Tuna'
*/
export async function getStatusByNumber(sosi_grouping) {
  const params = {
    metric: "status",
    filters: {
      sosi_edition: 2026,
      sosi_record_type: "SoSIndex",
      sosi_grouping: sosi_grouping,
      "status": validStatus,
    }
  }
  const sbn = await getChartData(stocksFlag, params);
  return formatData(sbn, params.metric, "count");
}

/**
 * Computes the status weighted by landings for a sosi grouping
 * @param {string} sosi_grouping - e.g. 'Area 21' or 'Tuna'
*/
export async function getStatusByLandings(sosi_grouping) {
  const params = {
    metric: "status",
    filters: {
      sosi_edition: 2026,
      sosi_record_type: "SoSIndex",
      sosi_grouping: sosi_grouping,
      "status": validStatus,
    },
    weight_by: "landings"
  }
  const sbl = await getChartData(stocksFlag, params);
  return formatData(sbl, params.metric, params.weight_by, 1e-6);
}

/**
 * @param {string} sosi_grouping
 * @param {number} scale
*/
export async function getCaptureProduction(sosi_grouping, scale = 1e-6) {
  const params = {
    sosi_grouping: sosi_grouping
  }

  const cap = await getChartData(captureFlag, params);
  return formatData(cap, "year", "production", scale);
}

/**
 * @param {string} sosi_grouping
 * @param {number} n_species
 * @param {number} scale
*/
export async function getTopSpeciesProduction(sosi_grouping, n_species = 10, scale = 1e-6) {
  const params = {
    sosi_grouping: sosi_grouping,
    n_species: n_species
  }

  const cap = await getChartData(captureFlag, params);
  return formatGroupedData(cap, "common_name", "year", "production", scale)
}
