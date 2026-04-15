const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

let currentControllers = {
  stocks: null,
  capture: null
};

export function buildWFSEndpoint({ requestType, mapName, column = null, value = null, operator = "="}) {
  const faoBaseUrl = "https://www.fao.org/fishery/geoserver/fifao/ows";
  const params = new URLSearchParams({
      service: "WFS",
      version: "1.0.0",
      request: requestType,
      typeName: mapName,
      outputFormat: "json",
  });

  if (column && value) {
    params.append("CQL_FILTER", `${column}${operator}'${value}'`);
  }

  return `${faoBaseUrl}?${params.toString()}`;
}

/**
 * Builds a URL for the SoSI Stocks API
 * @param {Object} options
 * @param {string} options.metric - e.g., 'status' or 'tier'
 * @param {Object} options.filters - Dictionary of key-value pairs
 * @param {string} [options.weight_by=null] - Optional weighting flag
 */
function buildStockDataEndpoint({ metric, filters = {}, weight_by = null }) {
  const params = new URLSearchParams();

  if (weight_by) {
    params.append("weight_by", weight_by);
  }

  Object.entries(filters).forEach(([key, value]) => {
    if (Array.isArray(value)) {
      value.forEach(val => params.append(key, val));
    } else if (value !== undefined && value !== null) {
      params.append(key, value);
    }
  });

  return `/api/stocks/query/${metric}?${params.toString()}`;
}

/**
 * Builds a URL for the Capture API
 * @param {Object} options 
 * @param {string} [options.sosi_grouping]
 * @param {number} [options.n_species]
 * @param {boolean} [options.exclude_isscaap]
 */
function buildCaptureDataEndpoint({ sosi_grouping, n_species, exclude_isscaap = false }) {
  const params = new URLSearchParams();

  if (sosi_grouping) params.append("sosi_grouping", sosi_grouping);
  if (n_species) params.append("n_species", String(n_species));
  if (exclude_isscaap) params.append("exclude_isscaap", "true");

  return `/api/capture?${params.toString()}`;
}
/**
 * @param {string} dataType - valid types: 'stocks', 'capture'
 * @param {object} endpointParams
 * @param {number} maxRetries
*/
export async function getChartData(dataType, endpointParams, maxRetries = 5) {
  const dataEndpoints = {
    stocks: buildStockDataEndpoint,
    capture: buildCaptureDataEndpoint
  };

  if (currentControllers[dataType]) {
    currentControllers[dataType].abort();
  }

  currentControllers[dataType] = new AbortController();
  const { signal } = currentControllers[dataType];

  const endpoint = dataEndpoints[dataType](endpointParams);

  let retries = 0;
  while (retries < maxRetries) {
    try {
      const response = await fetch(endpoint, { signal });

      if (response.ok) { return await response.json() };

      if (response.status < 500) break; 
    } catch (error) {
      if (error.name === "AbortError") {
        console.log(`Request for ${dataType} data cancelled due to user action.`);
        return null;
      }
      console.error(`Attemped ${retries + 1} retries of ${maxRetries}.`)
    }

    retries++;
    if (retries < maxRetries) {
      const waitTime = Math.pow(2, retries - 1) * 1000;
      await sleep(waitTime);
    }
  }

  console.error(`Error in fetching ${dataType} data from endpoint ${endpoint} after ${maxRetries} retries.`);
  return null;
}

