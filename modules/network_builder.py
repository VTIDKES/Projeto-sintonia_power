"""Power System Studio - Network builder.

Build a pandapower network from app tables (pandas DataFrames).

The app uses bus *names* (strings) to reference buses. This builder maps those
names to pandapower indices.

Validation focuses on:
- duplicate bus names
- references to non-existent buses
- disconnected / islanded networks

No Streamlit dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd

import pandapower as pp
import pandapower.topology as top


@dataclass
class BuildWarnings:
    warnings: List[str]


class NetworkBuildError(RuntimeError):
    pass


def _require_columns(df: pd.DataFrame, required: List[str], table_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise NetworkBuildError(f"Table '{table_name}' is missing columns: {missing}")


def validate_tables(
    buses: pd.DataFrame,
    gens: pd.DataFrame,
    loads: pd.DataFrame,
    lines: pd.DataFrame,
    trafos: pd.DataFrame,
) -> BuildWarnings:
    """Validate the user tables. Raises NetworkBuildError on hard errors.

    Returns warnings for non-fatal issues.
    """
    warnings: List[str] = []

    _require_columns(buses, ["name", "vn_kv", "type"], "buses")

    # Duplicate buses
    if buses["name"].isna().any():
        raise NetworkBuildError("'buses.name' cannot contain empty values.")
    if buses["name"].duplicated().any():
        dups = buses.loc[buses["name"].duplicated(), "name"].tolist()
        raise NetworkBuildError(f"Duplicate bus names found: {dups}")

    bus_set = set(buses["name"].astype(str).tolist())

    # Referenced buses
    def _check_ref(df: pd.DataFrame, col: str, table: str) -> None:
        if df is None or df.empty:
            return
        if col not in df.columns:
            return
        bad = sorted(set(df[col].astype(str)) - bus_set)
        if bad:
            raise NetworkBuildError(f"Table '{table}' references unknown buses in '{col}': {bad}")

    _check_ref(gens, "bus", "gens")
    _check_ref(loads, "bus", "loads")
    _check_ref(lines, "from_bus", "lines")
    _check_ref(lines, "to_bus", "lines")
    _check_ref(trafos, "hv_bus", "trafos")
    _check_ref(trafos, "lv_bus", "trafos")

    # Slack existence
    slack_buses = buses[buses["type"].astype(str).str.upper() == "SLACK"]
    if slack_buses.empty:
        # Accept generator marked slack as well
        if gens is None or gens.empty or ("slack" not in gens.columns) or (not gens["slack"].fillna(False).any()):
            warnings.append("No Slack bus found. The app will create an ext_grid at the first bus.")

    return BuildWarnings(warnings=warnings)


def build_pandapower_net(
    buses: pd.DataFrame,
    gens: pd.DataFrame,
    loads: pd.DataFrame,
    lines: pd.DataFrame,
    trafos: pd.DataFrame,
    f_hz: float = 60.0,
) -> Tuple[pp.pandapowerNet, Dict[str, int]]:
    """Create a pandapower network from input tables.

    Returns (net, bus_name_to_idx).
    """
    validate_tables(buses, gens, loads, lines, trafos)

    net = pp.create_empty_network(f_hz=f_hz)

    # -----------------
    # Buses
    # -----------------
    bus_map: Dict[str, int] = {}
    for _, row in buses.iterrows():
        name = str(row.get("name"))
        vn_kv = float(row.get("vn_kv"))
        idx = pp.create_bus(net, name=name, vn_kv=vn_kv)
        bus_map[name] = idx

    # Geodata (optional)
    if "x" in buses.columns and "y" in buses.columns:
        for _, row in buses.iterrows():
            name = str(row.get("name"))
            x = row.get("x")
            y = row.get("y")
            if pd.notna(x) and pd.notna(y):
                pp.create_bus_geodata(net, bus=bus_map[name], x=float(x), y=float(y))

    # -----------------
    # Slack / PV
    # -----------------
    # Strategy:
    # - If any generator has slack=True OR bus type Slack, create ext_grid there.
    # - If there are PV buses, create gen(s) with vm_pu.

    # bus type slack
    slack_bus_names = buses[buses["type"].astype(str).str.upper() == "SLACK"]["name"].astype(str).tolist()

    ext_created = False
    if slack_bus_names:
        bname = slack_bus_names[0]
        pp.create_ext_grid(net, bus_map[bname], vm_pu=1.02, name=f"Slack@{bname}")
        ext_created = True

    if gens is not None and not gens.empty:
        for _, row in gens.iterrows():
            bname = str(row.get("bus"))
            p_mw = float(row.get("p_mw", 0.0))
            vm_pu = float(row.get("vm_pu", 1.0))
            slack = bool(row.get("slack", False))
            name = str(row.get("name", "Gen"))
            min_q = float(row.get("min_q_mvar", -999.0))
            max_q = float(row.get("max_q_mvar", 999.0))

            if slack and not ext_created:
                pp.create_ext_grid(net, bus_map[bname], vm_pu=vm_pu, name=name)
                ext_created = True
            else:
                # Use gen as PV-like element
                pp.create_gen(net, bus_map[bname], p_mw=p_mw, vm_pu=vm_pu, name=name, min_q_mvar=min_q, max_q_mvar=max_q)

    if not ext_created:
        # Fallback
        first_bus = list(bus_map.values())[0]
        pp.create_ext_grid(net, first_bus, vm_pu=1.02, name="Slack")

    # -----------------
    # Loads
    # -----------------
    if loads is not None and not loads.empty:
        for _, row in loads.iterrows():
            bname = str(row.get("bus"))
            name = str(row.get("name", "Load"))
            p_mw = float(row.get("p_mw", 0.0))
            q_mvar = float(row.get("q_mvar", 0.0))
            pp.create_load(net, bus_map[bname], p_mw=p_mw, q_mvar=q_mvar, name=name)

    # -----------------
    # Lines
    # -----------------
    if lines is not None and not lines.empty:
        required = ["from_bus", "to_bus", "length_km"]
        _require_columns(lines, required, "lines")

        for _, row in lines.iterrows():
            fb = bus_map[str(row.get("from_bus"))]
            tb = bus_map[str(row.get("to_bus"))]
            name = str(row.get("name", "Line"))
            length_km = float(row.get("length_km", 1.0))
            std_type = str(row.get("std_type", "")).strip()

            if std_type:
                pp.create_line(net, fb, tb, length_km=length_km, std_type=std_type, name=name)
            else:
                r = float(row.get("r_ohm_per_km", 0.0))
                x = float(row.get("x_ohm_per_km", 0.0))
                c = float(row.get("c_nf_per_km", 0.0))
                max_i_ka = float(row.get("max_i_ka", 1.0))
                pp.create_line_from_parameters(
                    net,
                    from_bus=fb,
                    to_bus=tb,
                    length_km=length_km,
                    r_ohm_per_km=r,
                    x_ohm_per_km=x,
                    c_nf_per_km=c,
                    max_i_ka=max_i_ka,
                    name=name,
                )

    # -----------------
    # Transformers
    # -----------------
    if trafos is not None and not trafos.empty:
        required = ["hv_bus", "lv_bus", "sn_mva", "vn_hv_kv", "vn_lv_kv", "vk_percent", "vkr_percent"]
        _require_columns(trafos, required, "trafos")

        for _, row in trafos.iterrows():
            hv = bus_map[str(row.get("hv_bus"))]
            lv = bus_map[str(row.get("lv_bus"))]
            name = str(row.get("name", "Trafo"))

            pp.create_transformer_from_parameters(
                net,
                hv_bus=hv,
                lv_bus=lv,
                sn_mva=float(row.get("sn_mva")),
                vn_hv_kv=float(row.get("vn_hv_kv")),
                vn_lv_kv=float(row.get("vn_lv_kv")),
                vk_percent=float(row.get("vk_percent")),
                vkr_percent=float(row.get("vkr_percent")),
                pfe_kw=float(row.get("pfe_kw", 0.0)),
                i0_percent=float(row.get("i0_percent", 0.0)),
                tap_side=str(row.get("tap_side", "hv")),
                tap_pos=int(row.get("tap_pos", 0)),
                tap_neutral=int(row.get("tap_neutral", 0)),
                tap_min=int(row.get("tap_min", -10)),
                tap_max=int(row.get("tap_max", 10)),
                tap_step_percent=float(row.get("tap_step_percent", 1.25)),
                name=name,
            )

    return net, bus_map


def check_connectivity(net: pp.pandapowerNet) -> Tuple[bool, List[List[int]]]:
    """Return (is_connected, islands) where islands is list of bus index lists."""
    try:
        g = top.create_nxgraph(net, include_trafos=True)
        comps = [list(c) for c in top.connected_components(g)]
        if len(comps) <= 1:
            return True, []
        # Largest component is considered main; others are islands
        comps_sorted = sorted(comps, key=len, reverse=True)
        islands = comps_sorted[1:]
        return False, islands
    except Exception:
        # If topology fails, don't hard fail.
        return True, []
