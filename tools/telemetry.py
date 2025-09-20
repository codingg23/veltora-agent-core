"""
telemetry.py

DuckDB-backed telemetry query layer.

Agents never touch SQL directly. This module translates structured tool inputs
into DuckDB queries over Parquet files, then returns clean Python dicts.

Schema:
  thermal:  timestamp, rack_id, row_id, zone_id, metric, value_c
  power:    timestamp, rack_id, pdu_id, row_id, zone_id, kw, pue
  crac:     timestamp, unit_id, zone_id, load_fraction, supply_temp_c, return_temp_c
  capacity: rack_id, row_id, zone_id, total_u, used_u, rated_kw, current_kw
"""

import logging
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
import duckdb

logger = logging.getLogger(__name__)


def _scope_clause(scope: str, alias: str = "t") -> str:
    if not scope or scope in ("facility", "all"):
        return "1=1"
    if scope.startswith("rack:"):
        return f"{alias}.rack_id = '{scope[5:]}'"
    if scope.startswith("row:"):
        return f"{alias}.row_id = '{scope[4:]}'"
    if scope.startswith("zone:"):
        return f"{alias}.zone_id = '{scope[5:]}'"
    return "1=1"


class TelemetryTools:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self._con = duckdb.connect(":memory:")
        self._thermal = os.path.join(data_path, "thermal/**/*.parquet")
        self._power = os.path.join(data_path, "power/**/*.parquet")
        self._crac = os.path.join(data_path, "crac/**/*.parquet")
        self._capacity = os.path.join(data_path, "capacity/**/*.parquet")

    def _q(self, sql: str) -> list[dict]:
        try:
            return self._con.execute(sql).fetchdf().to_dict(orient="records")
        except duckdb.IOException:
            return []
        except Exception as e:
            logger.warning(f"Query failed: {e}")
            return []

    def query_thermal(self, scope, start, end, aggregation="avg", metric="inlet_temp_c", **kw):
        sc = _scope_clause(scope)
        agg = {"avg":"AVG","max":"MAX","min":"MIN","p95":"PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY value_c)"}.get(aggregation,"AVG")
        if aggregation == "timeseries":
            sql = f"SELECT date_trunc('hour',timestamp) ts, AVG(value_c) value_c FROM read_parquet('{self._thermal}') t WHERE {sc} AND metric='{metric}' AND timestamp BETWEEN '{start}' AND '{end}' GROUP BY 1 ORDER BY 1"
        else:
            sql = f"SELECT {agg}(value_c) value_c, COUNT(*) n FROM read_parquet('{self._thermal}') t WHERE {sc} AND metric='{metric}' AND timestamp BETWEEN '{start}' AND '{end}'"
        return {"scope":scope,"metric":metric,"aggregation":aggregation,"data":self._q(sql)}

    def get_hot_spots(self, threshold_c=25.0, facility="all"):
        sql = f"SELECT rack_id,row_id,zone_id,AVG(value_c) avg_c,MAX(value_c) max_c FROM read_parquet('{self._thermal}') t WHERE metric='inlet_temp_c' AND timestamp>=NOW()-INTERVAL '15 minutes' AND value_c>{threshold_c} GROUP BY 1,2,3 ORDER BY avg_c DESC LIMIT 20"
        rows = self._q(sql)
        return {"threshold_c":threshold_c,"hot_spots":rows,"count":len(rows)}

    def query_crac(self, zone, start, end):
        zc = f"zone_id='{zone}'" if zone and zone!="all" else "1=1"
        sql = f"SELECT unit_id,zone_id,AVG(load_fraction) avg_load,MAX(load_fraction) max_load,AVG(supply_temp_c) avg_supply_c,AVG(return_temp_c) avg_return_c FROM read_parquet('{self._crac}') t WHERE {zc} AND timestamp BETWEEN '{start}' AND '{end}' GROUP BY 1,2 ORDER BY avg_load DESC"
        return {"zone":zone,"crac_units":self._q(sql)}

    def forecast_thermal(self, scope, load_delta_kw, horizon_hours=4):
        return {"scope":scope,"load_delta_kw":load_delta_kw,"horizon_hours":horizon_hours,"estimated_inlet_delta_c":round(load_delta_kw*0.15,2),"note":"Linear approx only - use digital twin for accuracy"}

    def query_power(self, scope, start, end, aggregation="avg", metric="pdu_kw", **kw):
        sc = _scope_clause(scope)
        agg = {"avg":"AVG","max":"MAX","sum":"SUM"}.get(aggregation,"AVG")
        sql = f"SELECT {agg}(kw) kw, COUNT(*) n FROM read_parquet('{self._power}') t WHERE {sc} AND timestamp BETWEEN '{start}' AND '{end}'"
        return {"scope":scope,"aggregation":aggregation,"data":self._q(sql)}

    def query_pue(self, facility, start, end, granularity="hourly"):
        trunc = {"hourly":"hour","daily":"day"}.get(granularity,"hour")
        sql = f"SELECT date_trunc('{trunc}',timestamp) ts, AVG(pue) pue FROM read_parquet('{self._power}') t WHERE timestamp BETWEEN '{start}' AND '{end}' GROUP BY 1 ORDER BY 1"
        return {"facility":facility,"granularity":granularity,"data":self._q(sql)}

    def query_ups(self, ups_id):
        return {"ups_id":ups_id,"load_pct":None,"note":"Requires live SNMP feed"}

    def forecast_power(self, scope, horizon_hours=24):
        return {"scope":scope,"horizon_hours":horizon_hours,"note":"Use dc-power-forecaster for LSTM forecast"}

    def get_capacity(self, scope, include_reserved=True):
        sc = _scope_clause(scope)
        sql = f"SELECT zone_id,row_id,SUM(total_u-used_u) free_u,SUM(rated_kw-current_kw) free_kw,SUM(rated_kw) total_kw,COUNT(*) racks FROM read_parquet('{self._capacity}') t WHERE {sc} GROUP BY 1,2 ORDER BY free_kw"
        return {"scope":scope,"capacity":self._q(sql)}

    def estimate_load_impact(self, location, new_load_kw, rack_units):
        return {"location":location,"requested_kw":new_load_kw,"requested_u":rack_units,"current_capacity":self.get_capacity(location),"note":"Run digital twin for thermal impact"}

    def find_placement(self, required_kw, required_u, prefer_zone=None, min_cooling_headroom_pct=15.0):
        sql = f"SELECT zone_id,row_id,SUM(total_u-used_u) free_u,SUM(rated_kw-current_kw) free_kw FROM read_parquet('{self._capacity}') GROUP BY 1,2 HAVING SUM(total_u-used_u)>={required_u} AND SUM(rated_kw-current_kw)>={required_kw} ORDER BY free_kw DESC LIMIT 5"
        candidates = self._q(sql)
        if prefer_zone:
            candidates.sort(key=lambda r: 0 if r.get("zone_id")==prefer_zone else 1)
        return {"required_kw":required_kw,"required_u":required_u,"candidates":candidates}

    def get_underutilised(self, scope="facility", threshold_pct=30.0):
        sql = f"SELECT rack_id,row_id,zone_id,current_kw,rated_kw,ROUND(100.0*current_kw/NULLIF(rated_kw,0),1) utilisation_pct FROM read_parquet('{self._capacity}') WHERE (100.0*current_kw/NULLIF(rated_kw,0))<{threshold_pct} ORDER BY utilisation_pct LIMIT 20"
        return {"threshold_pct":threshold_pct,"racks":self._q(sql)}

    def resolve_time(self, reference: str) -> dict:
        now = datetime.now(tz=timezone.utc)
        r = reference.lower().strip()
        table = {
            "now": (now, now), "today": (now.replace(hour=0,minute=0,second=0), now),
            "yesterday": ((now-timedelta(days=1)).replace(hour=0,minute=0,second=0),(now-timedelta(days=1)).replace(hour=23,minute=59,second=59)),
            "last hour": (now-timedelta(hours=1), now), "past hour": (now-timedelta(hours=1), now),
            "past 4 hours": (now-timedelta(hours=4), now), "past 24 hours": (now-timedelta(hours=24), now),
            "this morning": (now.replace(hour=6,minute=0,second=0), now.replace(hour=12,minute=0,second=0)),
            "this afternoon": (now.replace(hour=12,minute=0,second=0), now.replace(hour=18,minute=0,second=0)),
            "last week": (now-timedelta(days=7), now),
        }
        if r in table:
            s, e = table[r]
            return {"start": s.isoformat(), "end": e.isoformat(), "reference": reference}
        m = re.match(r"(?:last|past)\s+(\d+)\s+(hour|hours|day|days|minute|minutes)", r)
        if m:
            n, unit = int(m.group(1)), m.group(2)
            delta = timedelta(hours=n) if "hour" in unit else timedelta(days=n) if "day" in unit else timedelta(minutes=n)
            return {"start": (now-delta).isoformat(), "end": now.isoformat(), "reference": reference}
        return {"error": f"Cannot parse: '{reference}'"}
