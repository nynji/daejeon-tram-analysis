"""
MCLP 최적화 진입점 스크립트
============================
전체 파이프라인 실행: 데이터 로딩 → 제약 적용 → 최적화 → 출력

Usage:
    python -m src.mclp.run_mclp
    python src/mclp/run_mclp.py
    python src/mclp/run_mclp.py --config configs/mclp_config.yaml
    python src/mclp/run_mclp.py --scenarios   # 시나리오 비교 모드
"""

import sys
import argparse
import logging
from pathlib import Path
import numpy as np
import time

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.mclp.config import load_config, MCLPConfig, ScenarioConfig
from src.mclp.data_loader import (
    load_parking_candidates,
    load_demand_points,
    load_fire_hydrants,
    load_school_zones,
    load_crosswalks,
    load_construction_zones,
    load_pedestrian_roads,
    load_parking_zones,
)
from src.mclp.distance import haversine_matrix, build_coverage_sets, candidate_distance_matrix
from src.mclp.weights import compute_base_weights, apply_soft_penalties
from src.mclp.constraints import (
    apply_hard_constraints,
    compute_candidate_preference,
)
from src.mclp.solver import solve_cmclp
from src.mclp.output import save_results_csv, save_map, save_scenario_comparison

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("MCLP")


def run_single(config: MCLPConfig, run_scenarios: bool = False):
    """단일 실행 또는 시나리오 비교."""
    print("=" * 70)
    print("  MCLP 최적화: AMR-탑차 연계 거점 배치")
    print("=" * 70)

    # ─── Step 1: 데이터 로딩 ───
    print("\n[Step 1/6] 데이터 로딩...")
    candidates = load_parking_candidates()
    demand_points = load_demand_points()
    fire_hydrants = load_fire_hydrants()

    # Optional data
    try:
        school_zones = load_school_zones()
    except FileNotFoundError:
        school_zones = None
        logger.warning("어린이보호구역 데이터 없음, 건너뜀")

    try:
        crosswalks = load_crosswalks()
    except FileNotFoundError:
        crosswalks = None
        logger.warning("횡단보도 데이터 없음, 건너뜀")

    try:
        construction_zones = load_construction_zones()
    except FileNotFoundError:
        construction_zones = None
        logger.warning("공사구간 데이터 없음, 건너뜀")

    try:
        parking_zones = load_parking_zones()
    except FileNotFoundError:
        parking_zones = None

    print(f"  후보지: {len(candidates)}개")
    print(f"  수요지점: {len(demand_points)}개")
    print(f"  소방시설: {len(fire_hydrants)}개")

    # ─── Step 2: Hard 제약 적용 ───
    print("\n[Step 2/6] Hard 제약 적용...")
    candidates, j_danger, exclusion_log = apply_hard_constraints(
        candidates, demand_points, fire_hydrants, config,
        school_zones=school_zones,
        construction_zones=construction_zones,
    )
    print(f"  통과 후보지: {len(candidates)}개, J_danger: {len(j_danger)}개")

    # ─── Step 3: 수요지점 집계 (대규모 문제 최적화) ───
    print("\n[Step 3/6] 거리 행렬 산출...")

    # 78,000+ 수요지점을 그리드 셀로 집계 (ILP 계산량 감소)
    # 100m × 100m 그리드 → 대표점 + 집계 가중치
    GRID_SIZE = 0.001  # ~100m in degrees
    demand_points_full = demand_points.copy()

    demand_points_agg = demand_points.copy()
    demand_points_agg["grid_lat"] = (demand_points_agg["lat"] / GRID_SIZE).round() * GRID_SIZE
    demand_points_agg["grid_lon"] = (demand_points_agg["lon"] / GRID_SIZE).round() * GRID_SIZE
    demand_points_agg["grid_key"] = (
        demand_points_agg["grid_lat"].astype(str) + "_" + demand_points_agg["grid_lon"].astype(str)
    )

    # 그리드별 대표점 (중심) + 수요 수
    grid_groups = demand_points_agg.groupby("grid_key").agg(
        lat=("lat", "mean"),
        lon=("lon", "mean"),
        count=("demand_id", "count"),
        name=("name", "first"),
    ).reset_index()
    grid_groups["demand_id"] = range(len(grid_groups))

    print(f"  수요지점 집계: {len(demand_points)} → {len(grid_groups)} 그리드 셀 (100m 해상도)")
    demand_points_solver = grid_groups  # 솔버에는 집계된 수요 사용

    dist_matrix = haversine_matrix(candidates, demand_points_solver)
    coverage_sets = build_coverage_sets(dist_matrix, config.solver.coverage_radius_km)

    # 거점 간 거리 행렬 (최소 거리 제약용)
    cand_dist = candidate_distance_matrix(candidates)

    # ─── Step 4: 수요 가중치 ───
    print("\n[Step 4/6] 수요 가중치 산출...")
    # 그리드 셀 단위로 가중치 산출 (count 반영)
    weights = compute_base_weights(demand_points_solver, config)
    # count(그리드 내 수요지점 수)를 가중치에 반영
    grid_counts = demand_points_solver["count"].values.astype(float)
    weights = weights * np.log1p(grid_counts)  # log 스케일로 반영 (과도한 집중 방지)
    weights = apply_soft_penalties(
        weights, demand_points_solver, candidates, coverage_sets, config,
        school_zones=school_zones,
        crosswalks=crosswalks,
    )

    # ─── Step 5: 거점 선호도 + 용량 ───
    print("\n[Step 5/6] 거점 선호도/용량 산출...")
    f_j = compute_candidate_preference(candidates, parking_zones, config)

    # 용량: 주차면수 × 비율
    capacities = (
        candidates["capacity"].values * config.capacity.parking_to_robot_ratio
    )
    capacities = np.maximum(capacities, config.capacity.min_capacity)

    # ─── Step 6: 최적화 실행 ───
    if run_scenarios and config.scenarios:
        print("\n[Step 6/6] 시나리오 비교 실행...")
        scenario_results = []

        for sc in config.scenarios:
            print(f"\n  --- 시나리오: {sc.name} (P={sc.P}, R={sc.coverage_radius_km}km) ---")
            # 시나리오별 커버리지 재계산
            sc_coverage = build_coverage_sets(dist_matrix, sc.coverage_radius_km)

            # 설정 복사 + 오버라이드
            sc_config = MCLPConfig(
                solver=config.solver.__class__(
                    P=sc.P,
                    coverage_radius_km=sc.coverage_radius_km,
                    alpha=config.solver.alpha,
                    D_min_m=config.solver.D_min_m,
                    time_limit_sec=config.solver.time_limit_sec,
                    amr_speed_kmh=config.solver.amr_speed_kmh,
                ),
                capacity=config.capacity,
                soft=config.soft,
                hard=config.hard,
                weights=config.weights,
            )

            sc_result = solve_cmclp(
                n_facilities=len(candidates),
                n_demands=len(demand_points_solver),
                coverage_sets=sc_coverage,
                weights=weights,
                f_j=f_j,
                capacities=capacities,
                j_danger=j_danger,
                config=sc_config,
                candidate_dist_matrix=cand_dist,
            )

            scenario_results.append({
                "name": sc.name,
                "P": sc.P,
                "radius": sc.coverage_radius_km,
                "n_selected": len(sc_result.selected_facilities),
                "coverage_ratio": sc_result.coverage_ratio,
                "objective": sc_result.objective_value,
                "time": sc_result.solve_time_sec,
            })

        save_scenario_comparison(scenario_results)
        print("\n  시나리오 비교 완료!")

        # 기본 시나리오 결과를 메인 출력으로 사용
        print(f"\n  기본 시나리오로 최종 출력 생성...")
        config_for_main = config

    else:
        print("\n[Step 6/6] CMCLP 솔버 실행...")

    # 메인 실행
    result = solve_cmclp(
        n_facilities=len(candidates),
        n_demands=len(demand_points_solver),
        coverage_sets=coverage_sets,
        weights=weights,
        f_j=f_j,
        capacities=capacities,
        j_danger=j_danger,
        config=config,
        candidate_dist_matrix=cand_dist,
    )

    # ─── 결과 출력 ───
    # 원본 수요지점 기준으로 커버리지 재산정
    # 집계 그리드 → 원본 수요지점 복원
    covered_grid_ids = set(result.covered_demands)
    # 원본에서 커버된 수요지점 수 추산
    total_original_covered = sum(
        demand_points_solver.iloc[i]["count"]
        for i in covered_grid_ids
        if i < len(demand_points_solver)
    )
    result.total_demand = len(demand_points_full)
    original_coverage = total_original_covered / len(demand_points_full)

    print("\n" + "─" * 70)
    print("결과 요약")
    print("─" * 70)
    print(f"  상태: {result.status}")
    print(f"  선택 거점: {len(result.selected_facilities)}개 / P={config.solver.P}")
    print(f"  커버리지 (그리드): {result.coverage_ratio:.1%} ({len(result.covered_demands)}/{len(demand_points_solver)})")
    print(f"  커버리지 (원본): {original_coverage:.1%} (~{int(total_original_covered)}/{len(demand_points_full)})")
    print(f"  목적함수: {result.objective_value:.2f}")
    print(f"  실행시간: {result.solve_time_sec:.1f}초")

    if result.selected_facilities:
        print("\n  선택된 거점:")
        for j in result.selected_facilities:
            assigned = sum(1 for fj in result.assignments.values() if fj == j)
            row = candidates.iloc[j]
            print(f"    [{j}] {row['name']} (주차면수 {row['capacity']}) → 할당 {assigned}개")

    # CSV 저장
    save_results_csv(result, candidates, demand_points_solver, config)

    # 지도 저장 (원본 수요지점 사용하여 시각화)
    save_map(result, candidates, demand_points_full, config, j_danger, construction_zones)

    print("\n" + "=" * 70)
    print("  완료!")
    print("=" * 70)

    return result


def main():
    parser = argparse.ArgumentParser(description="MCLP 최적화 실행")
    parser.add_argument("--config", type=str, default=None, help="설정 파일 경로")
    parser.add_argument("--scenarios", action="store_true", help="시나리오 비교 모드")
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else None
    config = load_config(config_path)

    run_single(config, run_scenarios=args.scenarios)


if __name__ == "__main__":
    main()
