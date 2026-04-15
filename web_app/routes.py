from flask import Blueprint, jsonify, request, abort, render_template
from sqlalchemy import select, func, Select
from werkzeug.datastructures import MultiDict
from .models import db, StockReference, StockLandings, Capture, Asfis
from pathlib import Path
import yaml

# Configuration Loading
root = Path(__file__).resolve().parent.parent
with open(root / "config" / "pipeline.yaml", "r") as file:
    pipeline_config = yaml.safe_load(file)

api_bp = Blueprint("api", __name__)

# Constants & Mappings
ALLOWED_FILTERS = {
    "sosi_grouping": StockReference.sosi_grouping,
    "isscaap_code": StockReference.isscaap_code,
    "status_year": StockReference.status_year,
    "tier": StockReference.tier,
    "status": StockReference.status,
    "sosi_edition": StockReference.sosi_edition,
    "ocean": StockReference.ocean,
}

METRIC_MAP = {
    "status": StockReference.status,
    "tier": StockReference.tier,
}

WEIGHT_MAP = {"landings": StockLandings.landings}

ASSESSMENT_YEAR = pipeline_config.get("assessment_year")
ISSCAAP_TO_EXCLUDE = pipeline_config.get("transformation", {}).get(
    "isscaap_to_exclude", []
)
SPECIES_TO_EXCLUDE = pipeline_config.get("plotting", {}).get("species_to_exclude", [])

# --- Routes ---


@api_bp.route("/")
def index():
    return render_template("index.html")


@api_bp.route("/api/stocks/query/<string:metric_key>", methods=["GET"])
def query_stock_assessments(metric_key: str):
    """Endpoint for Stock Reference data with optional weighting."""
    weight_key = request.args.get("weight_by")

    if metric_key not in METRIC_MAP:
        abort(
            400,
            description=f"Invalid or missing metric. Choices: {list(METRIC_MAP.keys())}",
        )
    if weight_key and weight_key not in WEIGHT_MAP:
        abort(
            400,
            description=f"Invalid weight key: {weight_key}. Choices: {list(WEIGHT_MAP.keys())}",
        )

    filters = _parse_filters(request.args)
    stmt = _build_stock_query(metric_key, weight_key, filters)

    return _execute_and_format(stmt)


@api_bp.route("/api/capture", methods=["GET"])
def query_capture():
    """Endpoint for Capture production time-series."""
    # Extract capture-specific parameters
    sosi_grouping = request.args.get("sosi_grouping")
    n_species = request.args.get("n_species", type=int)
    exclude_isscaap = request.args.get("exclude_isscaap", "false").lower() == "true"

    stmt = _build_capture_query(sosi_grouping, n_species, exclude_isscaap)

    return _execute_and_format(stmt)


# --- Query Builders ---


def _build_stock_query(
    metric_key: str, weight_key: str | None, filters: dict
) -> Select:
    metric_col = METRIC_MAP[metric_key]
    joined_tables = {metric_col.parent.entity}

    if weight_key:
        weight_col = WEIGHT_MAP[weight_key]
        agg_expr = func.sum(weight_col).label(weight_key)
        target_table = weight_col.parent.entity
    else:
        agg_expr = func.count(StockReference.uid).label("count")
        target_table = StockReference

    stmt = select(metric_col.label(metric_key), agg_expr)

    if target_table not in joined_tables:
        stmt = stmt.join(target_table)
        joined_tables.add(target_table)

    if filters:
        stmt = _apply_filters(stmt, filters, joined_tables)

    return stmt.group_by(metric_col).order_by(metric_col)


def _build_capture_query(
    sosi_grouping: str | None = None,
    n_species: int | None = None,
    exclude_isscaap: bool = False,
) -> Select:
    agg_expr = func.sum(Capture.production).label("production")
    stmt = select(Capture.year, agg_expr).group_by(Capture.year)

    if sosi_grouping:
        stmt = stmt.where(Capture.sosi_grouping == sosi_grouping)

    if exclude_isscaap:
        stmt = stmt.join(Asfis, Capture.asfis_code == Asfis.asfis_code)
        stmt = stmt.where(~Asfis.isscaap_code.in_(ISSCAAP_TO_EXCLUDE))

    if n_species:
        ts_stmt = (
            select(Capture.asfis_code, agg_expr)
            .where(Capture.year == ASSESSMENT_YEAR)
            .where(Capture.sosi_grouping == sosi_grouping)
            .group_by(Capture.asfis_code)
            .order_by(agg_expr.desc())
            .limit(n_species)
        )

        if SPECIES_TO_EXCLUDE:
            ts_stmt = ts_stmt.where(~Capture.asfis_code.in_(SPECIES_TO_EXCLUDE))

        ts_subq = ts_stmt.subquery("top_species")

        stmt = (
            stmt.add_columns(Capture.asfis_code, Asfis.common_name)
            .join(Asfis, Capture.asfis_code == Asfis.asfis_code)
            .where(Capture.asfis_code.in_(select(ts_subq.c.asfis_code)))
            .group_by(Capture.asfis_code, Asfis.common_name)
            .order_by(Capture.asfis_code)
        )

    stmt = stmt.order_by(Capture.year)

    return stmt


# --- Utilities ---


def _apply_filters(stmt: Select, filters: dict, joined_tables: set) -> Select:
    for key, value in filters.items():
        col = ALLOWED_FILTERS[key]
        if col.parent.entity not in joined_tables:
            stmt = stmt.join(col.parent.entity)
            joined_tables.add(col.parent.entity)

        stmt = stmt.where(col.in_(value) if isinstance(value, list) else col == value)

    return stmt


def _parse_filters(args: MultiDict[str, str]) -> dict:
    """Extracts valid filters from request arguments."""
    active_filters = {}
    for key in args.keys():
        if key in ALLOWED_FILTERS:
            vals = args.getlist(key)
            parsed_vals = [int(v) if v.isdigit() else v for v in vals]
            active_filters[key] = (
                parsed_vals if len(parsed_vals) > 1 else parsed_vals[0]
            )
    return active_filters


def _execute_and_format(stmt: Select, retries: int = 0, max_retries: int = 5):
    """Executes query and returns a D3-friendly list of dicts."""
    try:
        results = db.session.execute(stmt).mappings().all()
        return jsonify([dict(row) for row in results])
    except Exception as e:
        db.session.rollback()
        if retries < max_retries - 1:
            return _execute_and_format(stmt, retries + 1, max_retries)
        else:
            abort(500, description=f"Database error: {str(e)}")
