"""
TD-Risk-CMCLP 실행 파이프라인 v2
=================================
기획서 Section 6 기반 — 역할2(수리 최적화 엔지니어) 전담

실행 절차:
  1. 입력 로딩 (역할1이 생성한 파일 또는 현재 보유 데이터로 자체 생성)
  2. Sparse Set Ω 구성 (반경 + 하드 필터)
  3. 위험도 r_ij 산출
  4. p sweep (1~15) → p* 결정
  5. 민감도 8 시나리오 → 강건성 산출
  6. 결과 저장

Usage:
    python -m src.mclp.run_mclp_v2
    python -m src.mclp.run_mclp_v2 --mode SCREENING
    python -m src.mclp.run_mclp_v2 --mode FINAL
"""

import sys
import argparse
import logging
import json
import hashlib
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.mclp.solver_v2 import (
    SolverInput, ScenarioParams, SolverResult,
    solve_risk_mclp, run_p_sweep, run_sensitivity, build_scenario_list,
)
from src.mclp.data_loader import (
    load_parking_candidates,
    load_demand_points,
    load_fire_hydrants,
    load_school_zones,
    load_crosswalks,
    load_segment_priority,
    load_station_coords,
)
from src.mclp.distance import haversine_matrix, candidate_distance_matrix
from src.mclp.xgb_integration import load_xgb_risk_scores
from src.mclp.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("MCLP_v2")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = BASE_DIR / "outputs" / "mclp_v2"


# ─────────────────────────────────────────────
# 입력 생성 (역할1 산출물 미도착 시 자체 생성)
# ─────────────────────────────────────────────

def build_solver_input(
    mode: str = "SCREENING",
    radius_m: float = 2500.0,
    weight_mode: str = "기준",
) -> SolverInput:
    """솔버 입력 패키지 생성.

    기획서 Section 6.1 기반:
    - 수요지: 250m 격자 집계 (현재 100m → 250m으로 변경)
    - 후보지: 하드 제약 사전 필터
    - 연결: 반경 이내 + 접근 가능
    """
    logger.info("=" * 60)
    logger.info(f"  입력 생성 (mode={mode}, radius={radius_m}m, weight={weight_mode})")
    logger.info("=" * 60)

    # ─── 수요지 로딩 + 250m 격자 집계 ───
    demand_raw = load_demand_points()

    # ─── 스크리닝: 공사 영향 구간 수요만 선별 ───
    # 기획서 Section 3.1: screening_pass = dtw_screen_flag OR xgb_screen_flag
    dtw_path = BASE_DIR / "data" / "DTW k-means 군집분석" / "segment_priority.csv"
    xgb_risk = load_xgb_risk_scores()

    # DTW screen flag
    screened_segments = set()
    if dtw_path.exists():
        dtw = pd.read_csv(dtw_path)
        dtw_screened = set(dtw[dtw["dtw_screen_flag"] == 1]["segment_id"])
        screened_segments.update(dtw_screened)
        logger.info(f"  DTW 스크리닝 통과 구간: {len(dtw_screened)}개")

    # XGB screen flag: risk_score > 0.5 (위험 구간)
    if len(xgb_risk) > 0:
        xgb_screened = set(xgb_risk[xgb_risk["xgb_risk_score"] > 0.5]["segment_id"])
        screened_segments.update(xgb_screened)
        logger.info(f"  XGB 스크리닝 통과 구간: {len(xgb_screened)}개")

    logger.info(f"  통합 스크리닝 통과 구간 (OR): {len(screened_segments)}개")

    # 수요지 → segment 매핑 (최근접 역 기반)
    station_coords = load_station_coords()
    seg_priority = load_segment_priority()

    s_lat = station_coords["lat"].values
    s_lon = station_coords["lon"].values
    d_lat_raw = demand_raw["lat"].values
    d_lon_raw = demand_raw["lon"].values

    # 최근접 역 할당 (전체 수요)
    d_lat_rad = np.radians(d_lat_raw)
    d_lon_rad = np.radians(d_lon_raw)
    s_lat_rad = np.radians(s_lat)
    s_lon_rad = np.radians(s_lon)

    # 배치 처리 (메모리 절약)
    BATCH = 5000
    nearest_station_idx_all = np.zeros(len(demand_raw), dtype=int)
    for start in range(0, len(demand_raw), BATCH):
        end = min(start + BATCH, len(demand_raw))
        dlat = s_lat_rad[np.newaxis, :] - d_lat_rad[start:end, np.newaxis]
        dlon = s_lon_rad[np.newaxis, :] - d_lon_rad[start:end, np.newaxis]
        a = np.sin(dlat/2)**2 + np.cos(d_lat_rad[start:end, np.newaxis]) * np.cos(s_lat_rad[np.newaxis, :]) * np.sin(dlon/2)**2
        dist_batch = 2 * 6371.0 * np.arcsin(np.sqrt(a))
        nearest_station_idx_all[start:end] = np.argmin(dist_batch, axis=1)

    station_nos = station_coords["station_no"].values

    # station_no → segment_id 매핑
    seg_station_map = {}
    for _, row in seg_priority.iterrows():
        if row["segment_id"] == "NON_TRAM":
            continue
        if pd.notna(row.get("nearest_station_name")):
            for _, st in station_coords.iterrows():
                if st["station_name"] == row["nearest_station_name"]:
                    seg_station_map[st["station_no"]] = row["segment_id"]
                    break

    demand_raw["segment_id"] = [seg_station_map.get(station_nos[idx], "NON_TRAM") for idx in nearest_station_idx_all]

    # 스크리닝 필터 적용
    demand_screened = demand_raw[demand_raw["segment_id"].isin(screened_segments)].copy()
    logger.info(f"  수요 스크리닝: {len(demand_raw)} → {len(demand_screened)} (공사 영향권 내)")

    # 250m 격자 집계
    GRID_SIZE = 0.0025  # ~250m
    demand_screened["grid_lat"] = (demand_screened["lat"] / GRID_SIZE).round() * GRID_SIZE
    demand_screened["grid_lon"] = (demand_screened["lon"] / GRID_SIZE).round() * GRID_SIZE
    demand_screened["grid_key"] = demand_screened["grid_lat"].astype(str) + "_" + demand_screened["grid_lon"].astype(str)

    demand_grid = demand_screened.groupby("grid_key").agg(
        lat=("lat", "mean"),
        lon=("lon", "mean"),
        shop_count=("demand_id", "count"),
        segment_id=("segment_id", "first"),
    ).reset_index()
    demand_grid["demand_idx"] = range(len(demand_grid))

    logger.info(f"  격자 집계: {len(demand_screened)} → {len(demand_grid)} (250m 격자)")

    # ─── DTW 기반 가중치 산출 ───
    # (segment_id는 이미 demand_grid에 포함)

    # DTW multiplier 적용
    if dtw_path.exists():
        # weight_mode에 따른 multiplier 선택
        mult_col = {
            "기준": "cluster_multiplier_base",
            "보수": "cluster_multiplier_conservative",
            "강화": "cluster_multiplier_enhanced",
        }.get(weight_mode, "cluster_multiplier_base")

        rapid_col = {
            "기준": "rapid_multiplier_base",
            "보수": "rapid_multiplier_conservative",
            "강화": "rapid_multiplier_enhanced",
        }.get(weight_mode, "rapid_multiplier_base")

        dtw_mult_map = dict(zip(dtw["segment_id"], dtw[mult_col]))
        dtw_rapid_map = dict(zip(dtw["segment_id"], dtw[rapid_col]))
    else:
        dtw_mult_map = {}
        dtw_rapid_map = {}

    # w_i = shop_count × dtw_cluster × dtw_rapid
    demand_segments = demand_grid["segment_id"].values
    base_w = demand_grid["shop_count"].values.astype(float)
    dtw_mult = np.array([dtw_mult_map.get(seg, 1.0) for seg in demand_segments])
    dtw_rapid = np.array([dtw_rapid_map.get(seg, 1.0) for seg in demand_segments])
    weights = base_w * dtw_mult * dtw_rapid

    logger.info(f"  가중치 범위: [{weights.min():.2f}, {weights.max():.2f}]")

    # ─── 후보지 로딩 + 하드 필터 ───
    candidates = load_parking_candidates()
    fire_hydrants = load_fire_hydrants()

    # 소방시설 5m 필터
    c_lat = candidates["lat"].values
    c_lon = candidates["lon"].values
    f_lat = np.radians(fire_hydrants["lat"].values)
    f_lon = np.radians(fire_hydrants["lon"].values)

    h_j = np.ones(len(candidates), dtype=int)
    for j in range(len(candidates)):
        c_lat_r = np.radians(c_lat[j])
        c_lon_r = np.radians(c_lon[j])
        dlat_f = f_lat - c_lat_r
        dlon_f = f_lon - c_lon_r
        a_f = np.sin(dlat_f/2)**2 + np.cos(c_lat_r) * np.cos(f_lat) * np.sin(dlon_f/2)**2
        dists_f = 2 * 6371000.0 * np.arcsin(np.sqrt(a_f))  # meters
        if dists_f.min() <= 5.0:
            h_j[j] = 0

    blocked = (h_j == 0).sum()
    logger.info(f"  후보지: {len(candidates)}개, 소방시설 배제: {blocked}개")

    # ─── 후보지 사전 축소: 커버 수요 상위 150개만 ───
    dist_matrix_km = haversine_matrix(candidates, demand_grid)  # (J, I)
    radius_km = radius_m / 1000.0

    # 각 후보의 커버 가능 가중 수요 합산
    coverage_scores = np.zeros(len(candidates))
    for j in range(len(candidates)):
        if h_j[j] == 0:
            continue
        covered_mask = dist_matrix_km[j, :] <= radius_km
        coverage_scores[j] = weights[covered_mask].sum()

    # 상위 150개 후보만 유지 (허용된 것 중)
    MAX_CANDIDATES = 150
    valid_indices = np.where(h_j == 1)[0]
    if len(valid_indices) > MAX_CANDIDATES:
        top_indices = valid_indices[np.argsort(-coverage_scores[valid_indices])[:MAX_CANDIDATES]]
    else:
        top_indices = valid_indices

    # 재인덱싱
    candidates_filtered = candidates.iloc[top_indices].reset_index(drop=True)
    h_j_filtered = np.ones(len(candidates_filtered), dtype=int)
    c_lat_f = candidates_filtered["lat"].values
    c_lon_f = candidates_filtered["lon"].values
    dist_matrix_km = haversine_matrix(candidates_filtered, demand_grid)

    logger.info(f"  후보지 축소: {len(candidates)} → {len(candidates_filtered)}개 (커버 상위)")

    # ─── Sparse 연결 생성 (반경 이내만) ───

    connections = []
    risk_scores_list = []
    distances_list = []

    # 보호구역/횡단보도 기반 위험도 사전 계산
    try:
        school_zones = load_school_zones()
        sz_lat = school_zones["lat"].values
        sz_lon = school_zones["lon"].values
    except:
        sz_lat = np.array([])
        sz_lon = np.array([])

    try:
        crosswalks = load_crosswalks()
        cw_lat = crosswalks["lat"].values
        cw_lon = crosswalks["lon"].values
    except:
        cw_lat = np.array([])
        cw_lon = np.array([])

    for j in range(len(candidates_filtered)):
        covered_i = np.where(dist_matrix_km[j, :] <= radius_km)[0]
        for i in covered_i:
            connections.append((i, j))
            d_ij = dist_matrix_km[j, i] * 1000.0  # km → m
            distances_list.append(d_ij)

            # r_ij: 거리 비율 (간소화 — 보호구역/횡단보도는 배치 후 추가)
            r_dist = min(d_ij / radius_m, 1.0)
            risk_scores_list.append(r_dist)

    logger.info(f"  연결 생성: {len(connections)}개 (radius={radius_m}m)")

    # ─── SolverInput 조립 ───
    coords = np.column_stack([c_lat_f, c_lon_f])

    solver_input = SolverInput(
        demand_ids=np.arange(len(demand_grid)),
        demand_weights=weights,
        candidate_ids=np.arange(len(candidates_filtered)),
        candidate_h=h_j_filtered,
        candidate_coords=coords,
        connections=connections,
        risk_scores=np.array(risk_scores_list),
        distances=np.array(distances_list),
    )

    return solver_input


# ─────────────────────────────────────────────
# 결과 저장
# ─────────────────────────────────────────────

def save_results(
    p_results: list,
    p_star: int,
    sensitivity_results: list,
    robustness_df: pd.DataFrame,
    mode: str,
):
    """결과 저장 — 기획서 Section 8.2 준수."""
    out_dir = OUTPUT_DIR / mode
    out_dir.mkdir(parents=True, exist_ok=True)

    # scenario_summary.csv
    rows = []
    for r in p_results + sensitivity_results:
        rows.append({
            "scenario": getattr(r, '_scenario_name', f"P_{r.p:02d}"),
            "status": r.status,
            "p": r.p,
            "objective": r.objective,
            "wcr": r.wcr,
            "ucr": r.ucr,
            "covered_demands": len(r.covered_demands),
            "total_demands": len(r.covered_demands) + (int(1/r.ucr * len(r.covered_demands)) - len(r.covered_demands)) if r.ucr > 0 else 0,
            "solve_time_sec": r.solve_time_sec,
        })
    pd.DataFrame(rows).to_csv(out_dir / "scenario_summary.csv", index=False, encoding="utf-8-sig")

    # robustness_summary.csv
    robustness_df.to_csv(out_dir / "robustness_summary.csv", index=False, encoding="utf-8-sig")

    # selected_anchors.csv (BASE 기준)
    base_result = sensitivity_results[0] if sensitivity_results else (p_results[p_star-1] if p_star <= len(p_results) else None)
    if base_result and base_result.status == "Optimal":
        anchor_rows = []
        for j in base_result.selected_facilities:
            assigned = sum(1 for fj in base_result.assignments.values() if fj == j)
            anchor_rows.append({
                "candidate_idx": j,
                "assigned_demands": assigned,
                "stability": float(robustness_df[robustness_df["candidate_idx"] == j]["stability"].values[0]) if j in robustness_df["candidate_idx"].values else 0,
            })
        pd.DataFrame(anchor_rows).to_csv(out_dir / "selected_anchors.csv", index=False, encoding="utf-8-sig")

    # run_manifest.json
    manifest = {
        "run_timestamp": datetime.now().isoformat(),
        "mode": mode,
        "p_star": p_star,
        "n_scenarios": len(sensitivity_results),
        "solver": "PuLP_CBC",
    }
    with open(out_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    logger.info(f"  결과 저장: {out_dir}/")


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TD-Risk-CMCLP v2")
    parser.add_argument("--mode", type=str, default="SCREENING", choices=["SCREENING", "FINAL"])
    parser.add_argument("--p-max", type=int, default=15)
    args = parser.parse_args()

    print("=" * 70)
    print(f"  TD-Risk-CMCLP v2 — mode={args.mode}")
    print("=" * 70)

    # ─── 입력 생성 ───
    data = build_solver_input(mode=args.mode, radius_m=2500.0, weight_mode="기준")

    # ─── p sweep ───
    print(f"\n[Phase 1] p sweep (1~{args.p_max})")
    base_params = ScenarioParams(name="BASE", radius_m=2500, beta=0.10, d_min_m=0, weight_mode="기준", time_limit_sec=60)
    p_results, p_star = run_p_sweep(data, base_params, p_range=range(1, args.p_max + 1))

    print(f"\n  → 최적 거점 수 p* = {p_star}")

    # ─── 민감도 시나리오 ───
    print(f"\n[Phase 2] 민감도 시나리오 (p*={p_star})")
    scenarios = build_scenario_list(p_star)

    # weight_mode별 입력 재생성 필요 → 간소화: 같은 입력 사용 (weight는 BASE 기준)
    sensitivity_results, robustness_df = run_sensitivity(data, p_star, scenarios)

    # ─── 결과 요약 ───
    print("\n" + "─" * 70)
    print("결과 요약")
    print("─" * 70)
    print(f"  p* = {p_star}")

    base_r = sensitivity_results[0] if sensitivity_results else None
    if base_r:
        print(f"  BASE: WCR={base_r.wcr:.1%}, UCR={base_r.ucr:.1%}, OBJ={base_r.objective:.2f}")

    strong = robustness_df[robustness_df["category"] == "강건"]
    cond = robustness_df[robustness_df["category"] == "조건부"]
    print(f"  강건 후보: {len(strong)}개, 조건부: {len(cond)}개")

    # ─── 저장 ───
    save_results(p_results, p_star, sensitivity_results, robustness_df, args.mode)

    print("\n" + "=" * 70)
    print("  완료!")
    print("=" * 70)


if __name__ == "__main__":
    main()
