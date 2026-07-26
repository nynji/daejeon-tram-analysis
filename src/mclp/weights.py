"""
수요 가중치 산출 모듈
====================
segment priority 기반 + XGBoost 위험도 + Soft 제약 패널티 적용
"""

import pandas as pd
import numpy as np
from typing import Optional
import logging

from .config import MCLPConfig
from .distance import haversine_distance
from .data_loader import load_station_coords, load_segment_priority
from .xgb_integration import get_xgb_multiplier_map

logger = logging.getLogger(__name__)


def assign_demand_to_segment(
    demand_points: pd.DataFrame,
    station_coords: pd.DataFrame,
    segment_priority: pd.DataFrame,
) -> pd.Series:
    """수요지점을 가장 가까운 segment에 매핑.

    각 수요지점을 가장 가까운 역(station)으로 매핑하고,
    해당 역이 포함된 segment의 priority를 할당.

    Args:
        demand_points: 수요지점 (lat, lon)
        station_coords: 역 좌표 (station_no, lat, lon)
        segment_priority: segment별 우선순위

    Returns:
        Series of segment_id for each demand point
    """
    # 각 수요지점에서 가장 가까운 역 찾기
    d_lat = demand_points["lat"].values
    d_lon = demand_points["lon"].values
    s_lat = station_coords["lat"].values
    s_lon = station_coords["lon"].values

    # Vectorized nearest station assignment
    d_lat_rad = np.radians(d_lat)
    d_lon_rad = np.radians(d_lon)
    s_lat_rad = np.radians(s_lat)
    s_lon_rad = np.radians(s_lon)

    # (n_demand, n_stations)
    dlat = s_lat_rad[np.newaxis, :] - d_lat_rad[:, np.newaxis]
    dlon = s_lon_rad[np.newaxis, :] - d_lon_rad[:, np.newaxis]
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(d_lat_rad[:, np.newaxis]) * np.cos(s_lat_rad[np.newaxis, :])
        * np.sin(dlon / 2) ** 2
    )
    dist = 2 * 6371.0 * np.arcsin(np.sqrt(a))

    nearest_idx = np.argmin(dist, axis=1)
    nearest_station_no = station_coords["station_no"].values[nearest_idx]

    # station_no → segment_id 매핑
    # segment_priority에서 from_station_no 또는 to_station_no에 해당하는 segment 찾기
    # 간단히: 가장 가까운 역 번호가 포함된 첫 번째 segment 할당
    seg_lookup = {}
    if "nearest_station_name" in segment_priority.columns:
        # 네트워크 모듈 형식 - station name으로 매핑
        station_name_to_seg = {}
        for _, row in segment_priority.iterrows():
            if row["segment_id"] == "NON_TRAM":
                continue
            for col in ["nearest_station_name", "second_station_name"]:
                if pd.notna(row.get(col)):
                    name = row[col]
                    if name not in station_name_to_seg:
                        station_name_to_seg[name] = row["segment_id"]

        # station_no → station_name → segment_id
        no_to_name = dict(zip(station_coords["station_no"], station_coords["station_name"]))
        for no, name in no_to_name.items():
            if name in station_name_to_seg:
                seg_lookup[no] = station_name_to_seg[name]

    # Assign segments
    segments = []
    for sno in nearest_station_no:
        segments.append(seg_lookup.get(sno, "NON_TRAM"))

    return pd.Series(segments, index=demand_points.index)


def compute_base_weights(
    demand_points: pd.DataFrame,
    config: MCLPConfig,
) -> np.ndarray:
    """수요지점별 기본 가중치 산출.

    segment의 integrated_score (또는 priority_score)를 기반으로 가중치 생성.
    고위험 구간(🔴심각/🟠경고)에는 추가 인센티브 적용.

    Returns:
        np.ndarray of shape (|I|,) — 각 수요지점의 가중치
    """
    # segment priority 로딩
    seg_priority = load_segment_priority()
    station_coords = load_station_coords()

    # 수요지점 → segment 매핑
    demand_segments = assign_demand_to_segment(demand_points, station_coords, seg_priority)
    demand_points = demand_points.copy()
    demand_points["segment_id"] = demand_segments

    # segment별 스코어 매핑
    score_col = "integrated_score" if "integrated_score" in seg_priority.columns and config.weights.use_integrated_score else "priority_score"
    grade_col = "risk_grade"

    seg_score_map = dict(zip(seg_priority["segment_id"], seg_priority[score_col]))
    seg_grade_map = dict(zip(seg_priority["segment_id"], seg_priority[grade_col]))

    # 기본 가중치 = 정규화된 스코어 (0~1)
    scores = demand_points["segment_id"].map(seg_score_map).fillna(30.0)
    max_score = scores.max() if scores.max() > 0 else 1.0
    weights = (scores / max_score).values

    # 최소 가중치 보장
    weights = np.maximum(weights, 0.1)

    # 고위험 인센티브 적용
    grades = demand_points["segment_id"].map(seg_grade_map).fillna("🟢정상")
    high_risk_mask = grades.str.contains("심각|경고", na=False).values
    weights[high_risk_mask] *= config.weights.high_risk_incentive

    # ─── XGBoost 위험도 반영 ───
    xgb_mult_map = get_xgb_multiplier_map()
    if xgb_mult_map:
        xgb_mults = demand_points["segment_id"].map(xgb_mult_map).fillna(1.0).values
        weights = weights * xgb_mults
        xgb_applied = (xgb_mults > 1.0).sum()
        logger.info(f"  XGB multiplier 적용: {xgb_applied}개 수요지점에 위험도 반영")
    else:
        logger.warning("  XGB 위험도 미반영 (reference tables 없음)")

    logger.info(
        f"수요 가중치 산출: {len(weights)}개, "
        f"범위 [{weights.min():.3f}, {weights.max():.3f}], "
        f"고위험 인센티브 적용 {high_risk_mask.sum()}개"
    )
    return weights


def apply_soft_penalties(
    weights: np.ndarray,
    demand_points: pd.DataFrame,
    candidates: pd.DataFrame,
    coverage_sets: dict,
    config: MCLPConfig,
    school_zones: Optional[pd.DataFrame] = None,
    crosswalks: Optional[pd.DataFrame] = None,
) -> np.ndarray:
    """Soft 제약 패널티/인센티브를 가중치에 적용.

    Args:
        weights: 기본 가중치 배열
        demand_points: 수요지점
        candidates: 거점 후보지
        coverage_sets: 커버리지 집합
        config: 설정
        school_zones: 어린이보호구역 (optional)
        crosswalks: 횡단보도 (optional)

    Returns:
        패널티 적용된 가중치 배열
    """
    w = weights.copy()

    # ─── 제약8: 보호구역 패널티 ───
    if school_zones is not None and len(school_zones) > 0:
        _apply_school_zone_penalty(w, demand_points, school_zones, config)

    # ─── 제약10: 횡단보도 패널티 ───
    if crosswalks is not None and len(crosswalks) > 0:
        _apply_crosswalk_penalty(w, demand_points, crosswalks, config)

    logger.info(f"Soft 패널티 적용 후 가중치 범위: [{w.min():.3f}, {w.max():.3f}]")
    return w


def _apply_school_zone_penalty(
    weights: np.ndarray,
    demand_points: pd.DataFrame,
    school_zones: pd.DataFrame,
    config: MCLPConfig,
):
    """보호구역 인근 수요지점에 패널티 적용 (반경 300m 내)."""
    zone_lat = school_zones["lat"].values
    zone_lon = school_zones["lon"].values
    d_lat = demand_points["lat"].values
    d_lon = demand_points["lon"].values

    penalty_count = 0
    buffer_km = 0.3  # 300m

    for i in range(len(demand_points)):
        # 각 수요지점에서 가장 가까운 보호구역까지 거리
        dlat = np.radians(zone_lat - d_lat[i])
        dlon = np.radians(zone_lon - d_lon[i])
        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(np.radians(d_lat[i])) * np.cos(np.radians(zone_lat))
            * np.sin(dlon / 2) ** 2
        )
        dists = 2 * 6371.0 * np.arcsin(np.sqrt(a))
        min_dist = dists.min()

        if min_dist <= buffer_km:
            weights[i] *= config.soft.school_zone_penalty
            penalty_count += 1

    if penalty_count > 0:
        logger.info(f"  보호구역 패널티 적용: {penalty_count}개 수요지점")


def _apply_crosswalk_penalty(
    weights: np.ndarray,
    demand_points: pd.DataFrame,
    crosswalks: pd.DataFrame,
    config: MCLPConfig,
):
    """수요지점 주변 횡단보도 밀도 기반 패널티."""
    cw_lat = crosswalks["lat"].values
    cw_lon = crosswalks["lon"].values
    d_lat = demand_points["lat"].values
    d_lon = demand_points["lon"].values

    penalty_count = 0
    buffer_km = 0.5  # 500m 반경 내 횡단보도 수 카운트

    for i in range(len(demand_points)):
        dlat = np.radians(cw_lat - d_lat[i])
        dlon = np.radians(cw_lon - d_lon[i])
        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(np.radians(d_lat[i])) * np.cos(np.radians(cw_lat))
            * np.sin(dlon / 2) ** 2
        )
        dists = 2 * 6371.0 * np.arcsin(np.sqrt(a))
        crossings = (dists <= buffer_km).sum()

        if crossings >= config.soft.crosswalk_heavy_threshold:
            weights[i] *= config.soft.crosswalk_heavy_penalty_per_crossing ** (crossings - 2)
            penalty_count += 1
        elif crossings >= 3:
            weights[i] *= config.soft.crosswalk_penalty_per_crossing ** (crossings - 2)
            penalty_count += 1

    if penalty_count > 0:
        logger.info(f"  횡단보도 패널티 적용: {penalty_count}개 수요지점")
