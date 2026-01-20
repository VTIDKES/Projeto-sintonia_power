"""Power System Studio - Data handler.

Centralized import/export routines for network configuration and results.

The Streamlit app stores everything as pandas DataFrames inside st.session_state.
This module provides:
- Default empty tables (with required columns)
- JSON export/import (single file)
- CSV export helpers

JSON schema:
{
  "meta": {
    "app": "Power System Studio",
    "version": "0.1",
    "timestamp": "ISO-8601",
    "notes": "optional"
  },
  "tables": {
    "buses":  [ {row}, ... ],
    "gens":   [ {row}, ... ],
    "loads":  [ {row}, ... ],
    "lines":  [ {row}, ... ],
    "trafos": [ {row}, ... ]
  },
  "results": {
    "newton": {...},
    "simplified": {...}
  }
}

All table rows are stored as JSON-serializable primitives.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import pandas as pd

APP_NAME = "Power System Studio"
APP_VERSION = "0.1"


def default_tables() -> Dict[str, pd.DataFrame]:
    """Return default empty DataFrames for all network element tables."""
    buses = pd.DataFrame(
        columns=["name", "vn_kv", "type", "x", "y"],
        data=[
            {"name": "Bus 1", "vn_kv": 13.8, "type": "Slack", "x": 0.0, "y": 0.0},
            {"name": "Bus 2", "vn_kv": 13.8, "type": "PQ", "x": 1.5, "y": 0.0},
            {"name": "Bus 3", "vn_kv": 0.69, "type": "PQ", "x": 3.0, "y": -0.6},
            {"name": "Bus 4", "vn_kv": 0.69, "type": "PQ", "x": 3.0, "y": 0.6},
        ],
    )

    gens = pd.DataFrame(
        columns=["name", "bus", "p_mw", "vm_pu", "min_q_mvar", "max_q_mvar", "slack"],
        data=[
            {
                "name": "G1",
                "bus": "Bus 1",
                "p_mw": 5.0,
                "vm_pu": 1.02,
                "min_q_mvar": -5.0,
                "max_q_mvar": 5.0,
                "slack": True,
            }
        ],
    )

    loads = pd.DataFrame(
        columns=["name", "bus", "p_mw", "q_mvar"],
        data=[
            {"name": "L1", "bus": "Bus 2", "p_mw": 2.5, "q_mvar": 1.2},
            {"name": "L2", "bus": "Bus 4", "p_mw": 1.0, "q_mvar": 0.4},
        ],
    )

    # Lines: allow either std_type or explicit parameters.
    lines = pd.DataFrame(
        columns=[
            "name",
            "from_bus",
            "to_bus",
            "length_km",
            "r_ohm_per_km",
            "x_ohm_per_km",
            "c_nf_per_km",
            "max_i_ka",
            "std_type",
        ],
        data=[
            {
                "name": "Line 1",
                "from_bus": "Bus 1",
                "to_bus": "Bus 2",
                "length_km": 5.0,
                "r_ohm_per_km": 0.3,
                "x_ohm_per_km": 0.4,
                "c_nf_per_km": 10.0,
                "max_i_ka": 0.4,
                "std_type": "",
            },
            {
                "name": "Line 2",
                "from_bus": "Bus 3",
                "to_bus": "Bus 4",
                "length_km": 1.0,
                "r_ohm_per_km": 0.08,
                "x_ohm_per_km": 0.12,
                "c_nf_per_km": 0.0,
                "max_i_ka": 0.8,
                "std_type": "",
            },
        ],
    )

    trafos = pd.DataFrame(
        columns=[
            "name",
            "hv_bus",
            "lv_bus",
            "sn_mva",
            "vn_hv_kv",
            "vn_lv_kv",
            "vk_percent",
            "vkr_percent",
            "pfe_kw",
            "i0_percent",
            "tap_side",
            "tap_pos",
            "tap_neutral",
            "tap_min",
            "tap_max",
            "tap_step_percent",
        ],
        data=[
            {
                "name": "T1",
                "hv_bus": "Bus 2",
                "lv_bus": "Bus 3",
                "sn_mva": 5.0,
                "vn_hv_kv": 13.8,
                "vn_lv_kv": 0.69,
                "vk_percent": 6.0,
                "vkr_percent": 1.0,
                "pfe_kw": 2.0,
                "i0_percent": 0.2,
                "tap_side": "hv",
                "tap_pos": 0,
                "tap_neutral": 0,
                "tap_min": -5,
                "tap_max": 5,
                "tap_step_percent": 1.25,
            }
        ],
    )

    return {"buses": buses, "gens": gens, "loads": loads, "lines": lines, "trafos": trafos}


def df_to_records(df: pd.DataFrame) -> list[dict]:
    """Convert DataFrame to list of JSON-serializable row dicts."""
    if df is None or df.empty:
        return []
    clean = df.copy()
    # Replace NaN with None
    clean = clean.where(pd.notnull(clean), None)
    return clean.to_dict(orient="records")


def records_to_df(records: list[dict], columns: list[str]) -> pd.DataFrame:
    """Build a DataFrame from records while enforcing column order."""
    if not records:
        return pd.DataFrame(columns=columns)
    df = pd.DataFrame.from_records(records)
    for c in columns:
        if c not in df.columns:
            df[c] = None
    df = df[columns]
    return df


def export_project_json(
    tables: Dict[str, pd.DataFrame],
    results: Optional[Dict[str, Any]] = None,
    notes: str = "",
) -> str:
    """Serialize project tables/results into a JSON string."""
    payload: Dict[str, Any] = {
        "meta": {
            "app": APP_NAME,
            "version": APP_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "notes": notes,
        },
        "tables": {k: df_to_records(v) for k, v in tables.items()},
        "results": results or {},
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def import_project_json(json_str: str) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any], Dict[str, Any]]:
    """Load project from JSON string.

    Returns (tables, results, meta).
    """
    payload = json.loads(json_str)
    meta = payload.get("meta", {})
    tables_in = payload.get("tables", {})
    results = payload.get("results", {}) or {}

    # Build using default schemas
    defaults = default_tables()
    tables_out: Dict[str, pd.DataFrame] = {}
    for name, default_df in defaults.items():
        cols = list(default_df.columns)
        records = tables_in.get(name, []) or []
        tables_out[name] = records_to_df(records, cols)

    return tables_out, results, meta


def export_table_csv(df: pd.DataFrame) -> str:
    """Export a table to CSV string."""
    if df is None:
        return ""
    return df.to_csv(index=False)


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return default
        return float(x)
    except Exception:
        return default
