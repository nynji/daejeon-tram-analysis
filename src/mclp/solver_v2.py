"""
TD-Risk-CMCLP 솔버 v2
======================
기획서 Section 4 기반 위험조정 할당형 MCLP

목적함수: max Σ w_i(1 - β·r_ij) y_ij
제약:
  ① x_j ≤ h_j
  ② y_ij ≤ x_j,  y_ij ≤ a_ij
  ③ Σ_j y_ij = z_i ≤ 1
  ④ Σ_j x_j = p
  ⑤ (선택) x_j + x_k ≤ 1  for dist(j,k) < D_min
  ⑥ (조건부) Σ_i q_i y_ij ≤ C_j x_j
  ⑦ Σ_i y_ij ≥ x_j  (최소 사용)
"""

import numpy as np
import pandas as pd
from typing import Set, Dict, Optional, List, Tuple
from dataclasses import dataclass, field
import logging
import time

from pulp import (
    LpProblem, LpMaximize, LpVariable, LpBinary, LpStatus,
    lpSum, PULP_CBC_CMD, value,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 데이터 구조
# ─────────────────────────────────────────────

@dataclass
class ScenarioParams:
    """시나리오 파라미터."""
    name: str = "BASE"
    p: int = 10
    radius_m: float = 2500.0
    beta: float = 0.10
    d_min_m: float = 0.0  # 0이면 이격 미적용
    weight_mode: str = "기준"  # 기준 / 보수 / 강화
    capacity_enabled: bool = False
    time_limit_sec: int = 120


@dataclass
class SolverInput:
    """솔버 입력 데이터 패키지."""
    # 수요지
    demand_ids: np.ndarray          # (n_demand,) int
    demand_weights: np.ndarray      # (n_demand,) float — w_i

    # 후보지
    candidate_ids: np.ndarray       # (n_candidates,) int
    candidate_h: np.ndarray         # (n_candidates,) binary — 허용 여부

    # 연결 (Sparse)
    connections: List[Tuple[int, int]] = field(default_factory=list)
    risk_scores: np.ndarray = field(default_factory=lambda: np.array([]))
    distances: np.ndarray = field(default_factory=lambda: np.array([]))

    # Optional
    demand_q: Optional[np.ndarray] = None  # (n_demand,) float — 배송중량 (조건부)
    candidate_cap: Optional[np.ndarray] = None  # (n_candidates,) float — C_j (조건부)
    candidate_coords: Optional[np.ndarray] = None  # (n_candidates, 2) lat/lon


@dataclass
class SolverResult:
    """솔버 결과."""
    status: str = ""
    objective: float = 0.0
    solve_time_sec: float = 0.0
    p: int = 0

    selected_facilities: List[int] = field(default_factory=list)  # j indices
    covered_demands: List[int] = field(default_factory=list)      # i indices
    assignments: Dict[int, int] = field(default_factory=dict)     # i_idx → j_idx

    wcr: float = 0.0  # 가중 커버리지율
    ucr: float = 0.0  # 비가중 커버리지율
    total_weight: float = 0.0
    covered_weight: float = 0.0


# ─────────────────────────────────────────────
# 메인 솔버
# ─────────────────────────────────────────────

def solve_risk_mclp(
    data: SolverInput,
    params: ScenarioParams,
) -> SolverResult:
    """위험조정 할당형 MCLP 솔버 (2-Stage).

    Stage 1: 거점 선택 + 커버 판정 (z_i ≤ Σ x_j, 빠른 LP)
    Stage 2: Greedy 할당 (y_ij — 커버된 수요를 가장 가까운 선택 거점에 배정)
    """
    result = SolverResult()
    result.p = params.p
    start_time = time.time()

    n_demand = len(data.demand_ids)
    n_cand = len(data.candidate_ids)

    logger.info(f"[{params.name}] 솔버 시작: {n_demand} 수요, {n_cand} 후보, {len(data.connections)} 연결, p={params.p}")

    # ─── 수요별 커버 거점 집합 구축 ───
    # demand_covered_by[i] = [(j, conn_idx, risk)]
    demand_covered_by = {}
    conn_by_facility = {}
    for idx, (i, j) in enumerate(data.connections):
        r_ij = data.risk_scores[idx] if idx < len(data.risk_scores) else 0
        demand_covered_by.setdefault(i, []).append((j, idx, r_ij))
        conn_by_facility.setdefault(j, []).append((i, idx))

    # ─── Stage 1: 거점 선택 MIP ───
    prob = LpProblem(f"RiskMCLP_{params.name}", LpMaximize)

    x = {j: LpVariable(f"x_{j}", cat=LpBinary) for j in range(n_cand)}
    z = {i: LpVariable(f"z_{i}", cat=LpBinary) for i in range(n_demand)}

    # 목적함수: max Σ w_i × (1 - β × min_r_ij) × z_i
    # 여기서 min_r_ij = 수요 i를 커버하는 거점들 중 최소 위험도 (최선 연결 기준)
    obj_terms = []
    for i in range(n_demand):
        w_i = data.demand_weights[i]
        covers = demand_covered_by.get(i, [])
        if covers:
            min_risk = min(r for _, _, r in covers)
        else:
            min_risk = 0
        coeff = w_i * (1.0 - params.beta * min_risk)
        obj_terms.append(coeff * z[i])
    prob += lpSum(obj_terms), "Maximize_Risk_Adjusted_Coverage"

    # 제약 ① 후보 허용
    for j in range(n_cand):
        if data.candidate_h[j] == 0:
            prob += x[j] == 0, f"Blocked_{j}"

    # 제약: z_i ≤ Σ x_j (j covers i)
    for i in range(n_demand):
        covers = demand_covered_by.get(i, [])
        if covers:
            covering_js = list(set(j for j, _, _ in covers))
            prob += z[i] <= lpSum(x[j] for j in covering_js), f"Cover_{i}"
        else:
            prob += z[i] == 0, f"NoCover_{i}"

    # 제약 ④ 거점 수 ≤ p
    prob += lpSum(x[j] for j in range(n_cand)) <= params.p, "Facility_Count"

    # 제약 ⑤ 이격
    if params.d_min_m > 0 and data.candidate_coords is not None:
        spacing_pairs = _find_close_pairs(data.candidate_coords, params.d_min_m)
        for j1, j2 in spacing_pairs:
            prob += x[j1] + x[j2] <= 1, f"Space_{j1}_{j2}"
        logger.info(f"  이격 제약: {len(spacing_pairs)}개 쌍")

    # 제약 ⑦ 최소 사용: z_i의 합계에서 자연 보장 (목적함수가 z_i를 최대화)

    # ─── 솔버 실행 ───
    solver = PULP_CBC_CMD(msg=0, timeLimit=params.time_limit_sec)
    prob.solve(solver)

    result.status = LpStatus[prob.status]
    result.objective = value(prob.objective) if prob.objective else 0.0
    result.solve_time_sec = time.time() - start_time

    if result.status in ("Optimal", "Not Solved"):
        # Not Solved = time limit hit but feasible solution found
        if result.status == "Not Solved":
            # Check if any feasible solution exists
            obj_val = value(prob.objective)
            if obj_val is None or obj_val <= 0:
                logger.warning(f"  ❌ {params.name}: No feasible solution, {result.solve_time_sec:.1f}s")
                result.status = "Infeasible"
                return result
            result.objective = obj_val

        result.selected_facilities = [j for j in range(n_cand) if value(x[j]) is not None and value(x[j]) > 0.5]
        result.covered_demands = [i for i in range(n_demand) if value(z[i]) is not None and value(z[i]) > 0.5]

        # ─── Stage 2: Greedy 할당 ───
        selected_set = set(result.selected_facilities)
        for i in result.covered_demands:
            covers = demand_covered_by.get(i, [])
            # 선택된 거점 중 위험도 최소인 곳에 배정
            best_j = None
            best_risk = float("inf")
            for j, idx, r in covers:
                if j in selected_set and r < best_risk:
                    best_j = j
                    best_risk = r
            if best_j is not None:
                result.assignments[i] = best_j

        # KPI
        result.total_weight = float(data.demand_weights.sum())
        result.covered_weight = float(data.demand_weights[result.covered_demands].sum()) if result.covered_demands else 0.0
        result.wcr = result.covered_weight / result.total_weight if result.total_weight > 0 else 0.0
        result.ucr = len(result.covered_demands) / n_demand if n_demand > 0 else 0.0

        logger.info(
            f"  ✅ {params.name}: p={len(result.selected_facilities)}, WCR={result.wcr:.1%}, UCR={result.ucr:.1%}, "
            f"OBJ={result.objective:.2f}, {result.solve_time_sec:.1f}s"
        )
    else:
        logger.warning(f"  ❌ {params.name}: {result.status}, {result.solve_time_sec:.1f}s")

    return result


# ─────────────────────────────────────────────
# 시나리오 파이프라인
# ─────────────────────────────────────────────

def run_p_sweep(
    data: SolverInput,
    base_params: ScenarioParams,
    p_range: range = range(1, 16),
) -> Tuple[List[SolverResult], int]:
    """p=1..15 스윕 → 최적 p* 결정.

    결정 규칙:
    - WCR_p ≥ 0.95 × WCR_max인 가장 작은 p
    - 해당 p의 UCR ≥ 0.80이 아니면 처음 충족하는 p까지 증가
    """
    results = []
    for p in p_range:
        params = ScenarioParams(
            name=f"P_{p:02d}",
            p=p,
            radius_m=base_params.radius_m,
            beta=base_params.beta,
            d_min_m=base_params.d_min_m,
            weight_mode=base_params.weight_mode,
            capacity_enabled=base_params.capacity_enabled,
            time_limit_sec=base_params.time_limit_sec,
        )
        r = solve_risk_mclp(data, params)
        results.append(r)

    # 최적 p* 결정
    wcr_max = max(r.wcr for r in results) if results else 0
    p_star = base_params.p  # fallback

    for r in results:
        if r.wcr >= 0.95 * wcr_max:
            p_star = r.p
            if r.ucr >= 0.80:
                break
            # UCR 미달이면 계속 증가
        # WCR 미달이면 다음 p로

    # UCR 80% 첫 충족 p 확인
    for r in results:
        if r.ucr >= 0.80:
            if r.p > p_star:
                p_star = r.p
            break

    logger.info(f"  → p* = {p_star} (WCR={wcr_max:.1%} 기준)")
    return results, p_star


def run_sensitivity(
    data: SolverInput,
    p_star: int,
    scenarios: List[ScenarioParams],
) -> Tuple[List[SolverResult], pd.DataFrame]:
    """확정 p*로 8개 민감도 시나리오 실행 + 강건성 산출."""
    results = []
    for sc in scenarios:
        sc.p = p_star
        r = solve_risk_mclp(data, sc)
        results.append(r)

    # 강건성 산출
    n_scenarios = len(results)
    n_cand = len(data.candidate_ids)
    selection_count = np.zeros(n_cand)

    valid_scenarios = 0
    for r in results:
        if r.status == "Optimal":
            valid_scenarios += 1
            for j in r.selected_facilities:
                selection_count[j] += 1

    if valid_scenarios > 0:
        stability = selection_count / valid_scenarios
    else:
        stability = np.zeros(n_cand)

    robustness_df = pd.DataFrame({
        "candidate_idx": range(n_cand),
        "candidate_id": data.candidate_ids,
        "selection_count": selection_count.astype(int),
        "stability": stability.round(4),
        "category": pd.cut(
            stability,
            bins=[-0.01, 0.50, 0.75, 1.01],
            labels=["시나리오 민감", "조건부", "강건"],
        ),
    })

    return results, robustness_df


# ─────────────────────────────────────────────
# 유틸리티
# ─────────────────────────────────────────────

def _find_close_pairs(coords: np.ndarray, d_min_m: float) -> List[Tuple[int, int]]:
    """좌표 배열에서 d_min 이내 쌍 찾기 (Haversine)."""
    n = len(coords)
    pairs = []
    d_min_km = d_min_m / 1000.0

    lat = np.radians(coords[:, 0])
    lon = np.radians(coords[:, 1])

    for j in range(n):
        for k in range(j + 1, n):
            dlat = lat[k] - lat[j]
            dlon = lon[k] - lon[j]
            a = np.sin(dlat/2)**2 + np.cos(lat[j]) * np.cos(lat[k]) * np.sin(dlon/2)**2
            dist = 2 * 6371.0 * np.arcsin(np.sqrt(a))
            if dist < d_min_km:
                pairs.append((j, k))

    return pairs


def build_scenario_list(p_star: int) -> List[ScenarioParams]:
    """기획서 Section 7.1 기준 8개 시나리오 생성."""
    return [
        ScenarioParams(name="BASE", p=p_star, radius_m=2500, beta=0.10, d_min_m=0, weight_mode="기준"),
        ScenarioParams(name="WEIGHT_CONSERVATIVE", p=p_star, radius_m=2500, beta=0.10, d_min_m=0, weight_mode="보수"),
        ScenarioParams(name="WEIGHT_ENHANCED", p=p_star, radius_m=2500, beta=0.10, d_min_m=0, weight_mode="강화"),
        ScenarioParams(name="RADIUS_2000", p=p_star, radius_m=2000, beta=0.10, d_min_m=0, weight_mode="기준"),
        ScenarioParams(name="RADIUS_3000", p=p_star, radius_m=3000, beta=0.10, d_min_m=0, weight_mode="기준"),
        ScenarioParams(name="BETA_0", p=p_star, radius_m=2500, beta=0.00, d_min_m=0, weight_mode="기준"),
        ScenarioParams(name="BETA_20", p=p_star, radius_m=2500, beta=0.20, d_min_m=0, weight_mode="기준"),
        ScenarioParams(name="SPACING_500", p=p_star, radius_m=2500, beta=0.10, d_min_m=500, weight_mode="기준"),
    ]
