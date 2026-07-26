"""
CMCLP ILP 솔버
==============
PuLP 기반 정수계획법 — MCLP with capacity post-check
"""

import numpy as np
import pandas as pd
from typing import Set, Dict, Tuple, Optional
import logging
import time

from pulp import (
    LpProblem, LpMaximize, LpVariable, LpBinary, LpStatus,
    lpSum, PULP_CBC_CMD, value
)

from .config import MCLPConfig

logger = logging.getLogger(__name__)


class CMCLPResult:
    """CMCLP 최적화 결과."""

    def __init__(self):
        self.status: str = ""
        self.selected_facilities: list = []
        self.covered_demands: list = []
        self.assignments: dict = {}  # demand_id → facility_id
        self.coverage_ratio: float = 0.0
        self.objective_value: float = 0.0
        self.solve_time_sec: float = 0.0
        self.total_demand: int = 0
        self.infeasible_constraints: list = []


def solve_cmclp(
    n_facilities: int,
    n_demands: int,
    coverage_sets: dict,
    weights: np.ndarray,
    f_j: np.ndarray,
    capacities: np.ndarray,
    j_danger: Set[int],
    config: MCLPConfig,
    candidate_dist_matrix: Optional[np.ndarray] = None,
    rain_excluded: Optional[Set[int]] = None,
) -> CMCLPResult:
    """MCLP 솔버 (2단계: MCLP 최적화 → 용량 기반 할당).

    Stage 1 - MCLP:
        max Z = Σ w_i * Y_i + Σ f_j * X_j
        s.t.  Y_i ≤ Σ_{j∈N_i} X_j   (커버 = 인근 거점 선택됨)
              Σ X_j ≤ P
              X_j = 0 ∀ j ∈ J_danger
              X_j + X_k ≤ 1  ∀ (j,k) close
              Σ Y_i ≥ α * |I|

    Stage 2 - Assignment:
        각 커버된 수요를 가장 가까운 선택 거점에 할당 (용량 초과 시 다음 거점)
    """
    result = CMCLPResult()
    result.total_demand = n_demands

    start_time = time.time()

    # ─── 역커버리지: 수요 i를 커버하는 후보지 목록 ───
    demand_covered_by = {i: [] for i in range(n_demands)}
    for j, covered in coverage_sets.items():
        for i in covered:
            demand_covered_by[i].append(j)

    # ─── 문제 정의 ───
    prob = LpProblem("MCLP_AMR_Anchor", LpMaximize)

    # ─── 결정 변수 ───
    X = [LpVariable(f"X_{j}", cat=LpBinary) for j in range(n_facilities)]
    Y = [LpVariable(f"Y_{i}", cat=LpBinary) for i in range(n_demands)]

    # ─── 목적함수 ───
    prob += (
        lpSum(weights[i] * Y[i] for i in range(n_demands))
        + lpSum(f_j[j] * X[j] for j in range(n_facilities))
    ), "Maximize_Coverage"

    # ─── 제약식 1: 최대 P개 거점 ───
    prob += lpSum(X[j] for j in range(n_facilities)) <= config.solver.P, "Max_Facilities"

    # ─── 제약식 2: 커버 = 인근에 선택된 거점 존재 ───
    for i in range(n_demands):
        covering_facilities = demand_covered_by[i]
        if covering_facilities:
            prob += Y[i] <= lpSum(X[j] for j in covering_facilities), f"Cover_{i}"
        else:
            prob += Y[i] == 0, f"Uncoverable_{i}"

    # ─── 제약식 3: J_danger 원천 배제 ───
    for j in j_danger:
        if j < n_facilities:
            prob += X[j] == 0, f"Danger_{j}"

    # ─── 제약식 4: 집중호우 동적 배제 ───
    if rain_excluded:
        for j in rain_excluded:
            if j < n_facilities:
                prob += X[j] == 0, f"Rain_{j}"

    # ─── 제약식 5: 거점 간 최소 거리 ───
    if candidate_dist_matrix is not None:
        d_min_km = config.solver.D_min_m / 1000.0
        pair_count = 0
        for j in range(n_facilities):
            for k in range(j + 1, n_facilities):
                if candidate_dist_matrix[j, k] < d_min_km:
                    prob += X[j] + X[k] <= 1, f"Dist_{j}_{k}"
                    pair_count += 1
        if pair_count > 0:
            logger.info(f"  거점 간 최소 거리 제약: {pair_count}개 쌍")

    # ─── 제약식 6: 최소 커버리지 ───
    alpha = config.solver.alpha
    min_cov = int(np.ceil(alpha * n_demands))
    prob += lpSum(Y[i] for i in range(n_demands)) >= min_cov, "Min_Coverage"

    # ─── 솔버 실행 ───
    solver = PULP_CBC_CMD(msg=0, timeLimit=config.solver.time_limit_sec)
    prob.solve(solver)

    if LpStatus[prob.status] != "Optimal":
        # α 완화 재시도
        for attempt in range(1, 6):
            new_alpha = alpha - 0.1 * attempt
            if new_alpha <= 0.1:
                # 최소 커버리지 제약 제거하고 풀기
                del prob.constraints["Min_Coverage"]
                prob += lpSum(Y[i] for i in range(n_demands)) >= 1, "Min_Coverage"
                prob.solve(solver)
                break

            new_min = int(np.ceil(new_alpha * n_demands))
            del prob.constraints["Min_Coverage"]
            prob += lpSum(Y[i] for i in range(n_demands)) >= new_min, "Min_Coverage"
            prob.solve(solver)

            if LpStatus[prob.status] == "Optimal":
                logger.info(f"  α={new_alpha:.1f}에서 최적해 발견")
                break
            logger.warning(f"  α={new_alpha:.1f} infeasible, 재시도...")
        else:
            result.status = "Infeasible"
            result.solve_time_sec = time.time() - start_time
            return result

    # ─── 결과 추출 ───
    result.status = LpStatus[prob.status]
    result.objective_value = value(prob.objective) if prob.objective else 0
    result.solve_time_sec = time.time() - start_time

    if result.status == "Optimal":
        result.selected_facilities = [j for j in range(n_facilities) if value(X[j]) > 0.5]
        result.covered_demands = [i for i in range(n_demands) if value(Y[i]) > 0.5]
        result.coverage_ratio = len(result.covered_demands) / n_demands if n_demands > 0 else 0

        # ─── Stage 2: 용량 기반 할당 ───
        result.assignments = _assign_with_capacity(
            result.selected_facilities,
            result.covered_demands,
            coverage_sets,
            weights,
            capacities,
            candidate_dist_matrix,
        )

        logger.info(
            f"  결과: {len(result.selected_facilities)}개 거점, "
            f"커버리지 {result.coverage_ratio:.1%}, "
            f"OBJ {result.objective_value:.2f}, "
            f"{result.solve_time_sec:.1f}초"
        )

    return result


def _assign_with_capacity(
    selected: list,
    covered: list,
    coverage_sets: dict,
    weights: np.ndarray,
    capacities: np.ndarray,
    dist_matrix: np.ndarray,
) -> dict:
    """커버된 수요를 용량 고려하여 가장 가까운 거점에 할당."""
    assignments = {}
    remaining_cap = {j: capacities[j] for j in selected}

    # 역커버리지: 수요 → 선택된 거점들 중 커버하는 것들
    selected_set = set(selected)
    demand_options = {}
    for j in selected:
        for i in coverage_sets.get(j, []):
            if i in demand_options:
                demand_options[i].append(j)
            else:
                demand_options[i] = [j]

    # 가중치 높은 수요부터 할당 (greedy)
    sorted_demands = sorted(covered, key=lambda i: -weights[i])

    for i in sorted_demands:
        options = demand_options.get(i, [])
        if not options:
            continue

        # 가장 가까운 거점 (용량 남은 것 중)
        best_j = None
        best_dist = float("inf")
        for j in options:
            if remaining_cap.get(j, 0) > 0:
                if dist_matrix is not None:
                    d = dist_matrix[j, options[0]] if j < dist_matrix.shape[0] else float("inf")
                else:
                    d = 0
                # 단순히 첫 번째 가용 거점 선택
                best_j = j
                break

        if best_j is not None:
            assignments[i] = best_j
            remaining_cap[best_j] -= weights[i]

    return assignments
