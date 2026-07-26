"""
TD-Risk-CMCLP 최종 관제 API
============================
팀원 정본 데이터 + 우리 고속 솔버 (0.3초) + 실시간 시각화
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import time
import json
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Set
from dataclasses import dataclass, field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pulp import LpProblem, LpMaximize, LpVariable, LpBinary, LpStatus, lpSum, PULP_CBC_CMD, value

app = FastAPI(title="대전 트램 물류 관제 시스템", version="2.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

INPUT_DIR = PROJECT_ROOT / "data" / "mclp 입력 데이터"


# ─────────────────────────────────────────────
# 캐시 + 사전 로딩
# ─────────────────────────────────────────────
class Cache:
    demand: pd.DataFrame = None
    candidates: pd.DataFrame = None
    coverage: pd.DataFrame = None
    stations: pd.DataFrame = None
    tram_corridor: dict = None
    boundary: dict = None
    road_network: dict = None
    # 솔버용 사전 계산
    demand_ids: np.ndarray = None
    cand_ids: np.ndarray = None
    cand_h: np.ndarray = None
    cand_coords: np.ndarray = None
    # 반경별 연결 캐시
    connections_2500: list = None
    risks_2500: np.ndarray = None
    connections_2000: list = None
    risks_2000: np.ndarray = None
    connections_3000: list = None
    risks_3000: np.ndarray = None
    # 가중치
    weights_base: np.ndarray = None
    weights_conservative: np.ndarray = None
    weights_enhanced: np.ndarray = None
    demand_districts: np.ndarray = None
    loaded: bool = False

C = Cache()


def load_all():
    if C.loaded:
        return

    # 정본 데이터 로딩
    C.demand = pd.read_csv(INPUT_DIR / "demand_points.csv")
    C.candidates = pd.read_csv(INPUT_DIR / "candidate_sites.csv")
    C.coverage = pd.read_csv(INPUT_DIR / "coverage_matrix.csv")

    # 스크리닝 통과 수요만
    dem = C.demand[C.demand["screening_pass"] == 1].reset_index(drop=True)
    # 허용 후보만
    cands = C.candidates[C.candidates["model_review_candidate"] == 1].reset_index(drop=True)

    C.demand_ids = dem["demand_id"].astype(str).values
    C.cand_ids = cands["candidate_id"].astype(str).values
    C.cand_h = np.ones(len(cands), dtype=int)
    C.cand_coords = cands[["latitude", "longitude"]].values.astype(float)
    C.demand_districts = dem["district_name"].fillna("UNKNOWN").values

    # 가중치
    C.weights_base = dem["weight_base"].values.astype(float)
    C.weights_conservative = dem["weight_conservative"].values.astype(float)
    C.weights_enhanced = dem["weight_enhanced"].values.astype(float)

    # 연결 사전 계산 (반경별)
    cov = C.coverage[
        (C.coverage["optimization_eligible"] == 1) &
        (C.coverage["route_feasible"] == 1) &
        (C.coverage["model_review_candidate_snapshot"] == 1)
    ].copy()
    cov["candidate_id"] = cov["candidate_id"].astype(str)
    cov["demand_id"] = cov["demand_id"].astype(str)

    dem_idx = {did: i for i, did in enumerate(C.demand_ids)}
    cand_idx = {cid: i for i, cid in enumerate(C.cand_ids)}

    for radius, flag in [(2000, "within_2000m"), (2500, "within_2500m"), (3000, "within_3000m")]:
        subset = cov[(cov[flag] == 1) & cov["candidate_id"].isin(cand_idx) & cov["demand_id"].isin(dem_idx)]
        conns = []
        risks = []
        for _, row in subset.iterrows():
            di = dem_idx.get(row["demand_id"])
            ci = cand_idx.get(row["candidate_id"])
            if di is not None and ci is not None:
                conns.append((di, ci))
                risks.append(float(row["route_risk"]))
        setattr(C, f"connections_{radius}", conns)
        setattr(C, f"risks_{radius}", np.array(risks))

    # 지도 데이터
    st_path = PROJECT_ROOT / "data" / "network" / "역_좌표.csv"
    if st_path.exists():
        C.stations = pd.read_csv(st_path)

    corridor_path = PROJECT_ROOT / "outputs" / "tram_corridor.geojson"
    if corridor_path.exists():
        with open(corridor_path, "r", encoding="utf-8") as f:
            C.tram_corridor = json.load(f)

    boundary_path = PROJECT_ROOT / "data" / "network" / "daejeon_boundary_wgs84.geojson"
    if boundary_path.exists():
        with open(boundary_path, "r", encoding="utf-8") as f:
            C.boundary = json.load(f)

    road_path = PROJECT_ROOT / "data" / "network" / "daejeon_link.geojson"
    if road_path.exists():
        with open(road_path, "r", encoding="utf-8") as f:
            roads = json.load(f)
        C.road_network = {
            "type": "FeatureCollection",
            "features": [f for f in roads["features"] if str(f["properties"].get("ROAD_RANK","")) in ("101","102","103")]
        }

    C.loaded = True


# ─────────────────────────────────────────────
# 고속 솔버 (Stage1: LP cover + Stage2: greedy assign)
# ─────────────────────────────────────────────
def solve_fast(p, radius_m, beta, weight_mode, d_min_m=0, incident_lat=None, incident_lon=None, incident_radius_m=500):
    conns = getattr(C, f"connections_{radius_m}", C.connections_2500)
    risks = getattr(C, f"risks_{radius_m}", C.risks_2500)

    if conns is None or len(conns) == 0:
        return {"status": "NoRoutes", "anchors": [], "covered": [], "uncovered": [], "routes": [], "wcr": 0, "ucr": 0, "obj": 0, "time_ms": 0, "districts": {}}

    weights = {"기준": C.weights_base, "보수": C.weights_conservative, "강화": C.weights_enhanced}.get(weight_mode, C.weights_base)

    n_dem = len(C.demand_ids)
    n_cand = len(C.cand_ids)
    h = C.cand_h.copy()

    # 돌발 사고
    if incident_lat and incident_lon:
        for j in range(n_cand):
            dlat = np.radians(C.cand_coords[j, 0] - incident_lat)
            dlon = np.radians(C.cand_coords[j, 1] - incident_lon)
            a = np.sin(dlat/2)**2 + np.cos(np.radians(incident_lat)) * np.cos(np.radians(C.cand_coords[j,0])) * np.sin(dlon/2)**2
            if 2 * 6371000 * np.arcsin(np.sqrt(a)) <= incident_radius_m:
                h[j] = 0

    # 수요별 커버 거점
    demand_covered_by = {}
    for idx, (i, j) in enumerate(conns):
        if h[j] == 0:
            continue
        demand_covered_by.setdefault(i, []).append((j, idx, risks[idx]))

    start = time.time()

    # LP
    prob = LpProblem("FastMCLP", LpMaximize)
    x = {j: LpVariable(f"x{j}", cat=LpBinary) for j in range(n_cand)}
    z = {i: LpVariable(f"z{i}", cat=LpBinary) for i in range(n_dem)}

    # obj: max Σ w_i(1-β*min_risk)*z_i
    obj = []
    for i in range(n_dem):
        covers = demand_covered_by.get(i, [])
        min_r = min((r for _, _, r in covers), default=0)
        obj.append(weights[i] * (1.0 - beta * min_r) * z[i])
    prob += lpSum(obj)

    # 제약
    for j in range(n_cand):
        if h[j] == 0:
            prob += x[j] == 0
    for i in range(n_dem):
        covers = demand_covered_by.get(i, [])
        if covers:
            prob += z[i] <= lpSum(x[j] for j, _, _ in covers)
        else:
            prob += z[i] == 0
    prob += lpSum(x[j] for j in range(n_cand)) <= p

    # 이격
    if d_min_m > 0:
        for j1 in range(n_cand):
            for j2 in range(j1+1, n_cand):
                dlat = C.cand_coords[j1,0] - C.cand_coords[j2,0]
                dlon = C.cand_coords[j1,1] - C.cand_coords[j2,1]
                dist_m = np.sqrt(dlat**2 + dlon**2) * 111000
                if dist_m < d_min_m:
                    prob += x[j1] + x[j2] <= 1

    solver = PULP_CBC_CMD(msg=0, timeLimit=30)
    prob.solve(solver)

    solve_ms = (time.time() - start) * 1000
    status = LpStatus[prob.status]

    if status not in ("Optimal",):
        return {"status": status, "anchors": [], "covered": [], "uncovered": [], "routes": [], "wcr": 0, "ucr": 0, "obj": 0, "time_ms": solve_ms, "districts": {}}

    selected = [j for j in range(n_cand) if value(x[j]) and value(x[j]) > 0.5]
    covered = [i for i in range(n_dem) if value(z[i]) and value(z[i]) > 0.5]

    # Greedy assign
    selected_set = set(selected)
    assignments = {}
    for i in covered:
        covers = demand_covered_by.get(i, [])
        best_j = None
        best_r = 999
        for j, idx, r in covers:
            if j in selected_set and r < best_r:
                best_j = j
                best_r = r
        if best_j is not None:
            assignments[i] = best_j

    wcr = weights[covered].sum() / weights.sum() if weights.sum() > 0 else 0
    ucr = len(covered) / n_dem if n_dem > 0 else 0

    # District coverage
    districts = {}
    for dist in set(C.demand_districts):
        members = np.where(C.demand_districts == dist)[0]
        cov_count = sum(1 for m in members if m in set(covered))
        districts[dist] = cov_count / len(members) if len(members) > 0 else 0

    # 앵커 정보
    cand_df = C.candidates[C.candidates["model_review_candidate"]==1].reset_index(drop=True)
    dem_df = C.demand[C.demand["screening_pass"]==1].reset_index(drop=True)

    anchors = []
    for j in selected:
        row = cand_df.iloc[j]
        assigned = sum(1 for fj in assignments.values() if fj == j)
        anchors.append({
            "candidate_id": str(row["candidate_id"]),
            "name": str(row["candidate_name"]),
            "lat": float(row["latitude"]),
            "lon": float(row["longitude"]),
            "district": str(row["district_name"]),
            "parking_type": str(row.get("parking_type", "")),
            "assigned_demands": assigned,
        })

    # 좌표
    covered_coords = [[float(dem_df.iloc[i]["latitude"]), float(dem_df.iloc[i]["longitude"])] for i in covered[:800]]
    uncov_ids = set(range(n_dem)) - set(covered)
    uncovered_coords = [[float(dem_df.iloc[i]["latitude"]), float(dem_df.iloc[i]["longitude"])] for i in list(uncov_ids)[:300]]

    # AMR routes
    routes = []
    for i, j in assignments.items():
        if i < len(dem_df) and j < len(cand_df):
            routes.append({
                "from": [float(cand_df.iloc[j]["longitude"]), float(cand_df.iloc[j]["latitude"])],
                "to": [float(dem_df.iloc[i]["longitude"]), float(dem_df.iloc[i]["latitude"])],
            })

    # Truck routes: 각 창고에서 선택된 거점으로 연결
    wh_path = PROJECT_ROOT / "data" / "택배창고_좌표.csv"
    truck_routes = []
    if wh_path.exists():
        wh = pd.read_csv(wh_path)
        selected_cands = cand_df.iloc[selected] if selected else pd.DataFrame()
        for _, w in wh.iterrows():
            wlat, wlon = float(w["lat"]), float(w["lon"])
            if len(selected_cands) > 0:
                # 선택된 거점 중 가장 가까운 2개에 연결
                dists = np.sqrt(
                    (selected_cands["latitude"].astype(float) - wlat)**2 +
                    (selected_cands["longitude"].astype(float) - wlon)**2
                )
                nearest = dists.nsmallest(min(2, len(dists))).index
                for ci in nearest:
                    c = selected_cands.loc[ci]
                    truck_routes.append({
                        "from": [wlon, wlat],
                        "to": [float(c["longitude"]), float(c["latitude"])],
                        "warehouse": w["name"],
                    })

    return {
        "status": status, "anchors": anchors,
        "covered": covered_coords, "uncovered": uncovered_coords,
        "routes": routes, "truck_routes": truck_routes,
        "wcr": wcr, "ucr": ucr,
        "obj": float(value(prob.objective) or 0),
        "time_ms": solve_ms, "districts": districts,
        "p_selected": len(selected), "total": n_dem, "covered_n": len(covered),
    }


# ─────────────────────────────────────────────
# API
# ─────────────────────────────────────────────

class OptimizeRequest(BaseModel):
    p: int = Field(default=10, ge=1, le=20)
    radius_m: int = Field(default=2500)
    beta: float = Field(default=0.10, ge=0.0, le=0.30)
    d_min_m: float = Field(default=300, ge=0, le=1000)
    weight_mode: str = Field(default="기준")
    weather: str = Field(default="정상")
    incident_lat: Optional[float] = None
    incident_lon: Optional[float] = None
    incident_radius_m: float = Field(default=500)


@app.on_event("startup")
async def startup():
    load_all()


@app.post("/api/optimize")
async def optimize(req: OptimizeRequest):
    if not C.loaded:
        raise HTTPException(500, "loading")

    if req.weather in ("적설", "한파"):
        return {"status": "WeatherHalt", "anchors": [], "covered": [], "uncovered": [],
                "routes": [], "wcr": 0, "ucr": 0, "obj": 0, "time_ms": 0,
                "districts": {}, "p_selected": 0, "total": len(C.demand_ids), "covered_n": 0}

    if req.radius_m not in (2000, 2500, 3000):
        req.radius_m = 2500

    result = solve_fast(
        p=req.p, radius_m=req.radius_m, beta=req.beta,
        weight_mode=req.weight_mode, d_min_m=req.d_min_m,
        incident_lat=req.incident_lat, incident_lon=req.incident_lon,
        incident_radius_m=req.incident_radius_m,
    )
    return result


@app.get("/api/status")
async def get_status():
    if not C.loaded: return {"status": "loading"}
    return {"status": "ready", "demands": len(C.demand_ids), "candidates": len(C.cand_ids), "routes_2500": len(C.connections_2500 or [])}


@app.get("/api/tram_stations")
async def tram_stations():
    if C.stations is None: return []
    return [{"no": int(r["station_no"]), "name": r["station_name"], "lat": float(r["lat"]), "lon": float(r["lon"])} for _, r in C.stations.iterrows()]


@app.get("/api/tram_line")
async def tram_line():
    if C.stations is None: return {"type": "FeatureCollection", "features": []}
    st = C.stations.sort_values("station_no")
    main = st[st["station_no"].between(201,240)].sort_values("station_no")
    mc = [[float(r["lon"]),float(r["lat"])] for _,r in main.iterrows()]
    if len(mc)>1: mc.append(mc[0])
    b1 = st[st["station_no"].isin([212,241,242,243,244])].sort_values("station_no")
    b1c = [[float(r["lon"]),float(r["lat"])] for _,r in b1.iterrows()]
    b2 = st[st["station_no"].isin([233,245])].sort_values("station_no")
    b2c = [[float(r["lon"]),float(r["lat"])] for _,r in b2.iterrows()]
    feats = [{"type":"Feature","geometry":{"type":"LineString","coordinates":mc},"properties":{"branch":"main"}}]
    if len(b1c)>1: feats.append({"type":"Feature","geometry":{"type":"LineString","coordinates":b1c},"properties":{"branch":"east"}})
    if len(b2c)>1: feats.append({"type":"Feature","geometry":{"type":"LineString","coordinates":b2c},"properties":{"branch":"south"}})
    return {"type":"FeatureCollection","features":feats}


@app.get("/api/construction_zones")
async def construction_zones():
    return C.tram_corridor or {"type":"FeatureCollection","features":[]}


@app.get("/api/boundary")
async def boundary():
    return C.boundary or {"type":"FeatureCollection","features":[]}


@app.get("/api/road_network")
async def road_network():
    return C.road_network or {"type":"FeatureCollection","features":[]}


@app.get("/api/warehouses")
async def get_warehouses():
    """택배 물류 창고 (탑차 출발지)."""
    wh_path = PROJECT_ROOT / "data" / "택배창고_좌표.csv"
    if not wh_path.exists():
        return []
    wh = pd.read_csv(wh_path)
    return [{"name": r["name"], "lat": float(r["lat"]), "lon": float(r["lon"]), "address": r["address"]} for _, r in wh.iterrows()]


@app.get("/api/truck_routes")
async def get_truck_routes():
    """탑차 경로: 각 창고 → 선택 거점들 (최근 optimize 결과 기반 시뮬레이션)."""
    wh_path = PROJECT_ROOT / "data" / "택배창고_좌표.csv"
    if not wh_path.exists():
        return {"type": "FeatureCollection", "features": []}
    wh = pd.read_csv(wh_path)
    # 허용된 거점 좌표
    cands = C.candidates[C.candidates["model_review_candidate"]==1].reset_index(drop=True)
    # 각 창고에서 가장 가까운 3개 거점으로 시뮬레이션 경로 생성
    features = []
    for _, w in wh.iterrows():
        wlat, wlon = float(w["lat"]), float(w["lon"])
        # 거리 계산
        dists = np.sqrt((cands["latitude"].astype(float) - wlat)**2 + (cands["longitude"].astype(float) - wlon)**2)
        nearest_3 = dists.nsmallest(3).index
        for ci in nearest_3:
            c = cands.iloc[ci]
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[wlon, wlat], [float(c["longitude"]), float(c["latitude"])]],
                },
                "properties": {"warehouse": w["name"], "anchor": c["candidate_name"]},
            })
    return {"type": "FeatureCollection", "features": features}
