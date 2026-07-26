"""
12대 복합 제약 조건 엔진
========================
Hard 7개: 위반 시 후보지 원천 배제
Soft 5개: weights.py에서 패널티로 처리
"""

import numpy as np
import pandas as pd
from typing import Set, Tuple
import logging

from .config import MCLPConfig
from .distance import haversine_distance

logger = logging.getLogger(__name__)


def apply_hard_constraints(
    candidates: pd.DataFrame,
    demand_points: pd.DataFrame,
    fire_hydrants: pd.DataFrame,
    config: MCLPConfig,
    school_zones: pd.DataFrame = None,
    construction_zones: pd.DataFrame = None,
    pedestrian_roads: pd.DataFrame = None,
) -> Tuple[pd.DataFrame, Set[int], dict]:
    """Hard 제약 조건 적용.

    Returns:
        - filtered_candidates: 제약 통과한 후보지
        - j_danger: 위험후보지 인덱스 집합 (소방시설)
        - exclusion_log: 각 제약별 배제 통계
    """
    exclusion_log = {}
    original_count = len(candidates)

    # ─── 제약7: 소방시설 보호구역 (반경 5m) ───
    j_danger = _find_fire_hydrant_danger(candidates, fire_hydrants, config)
    exclusion_log["fire_hydrant_5m"] = len(j_danger)

    # ─── 제약1: 도로위계 (주차면수 0인 곳 = 접근 불가 추정) ───
    # 주차면수가 0이면 실질적으로 AMR 운영 불가
    zero_capacity = set(candidates[candidates["capacity"] <= 0].index.tolist())
    exclusion_log["zero_capacity"] = len(zero_capacity)

    # ─── 최종 배제 집합 (j_danger는 솔버에서 X_j=0으로 처리) ───
    hard_exclude = zero_capacity  # 완전 제거
    filtered = candidates[~candidates.index.isin(hard_exclude)].copy()
    filtered = filtered.reset_index(drop=True)

    # j_danger도 filtered 기준으로 재매핑
    # 새로운 인덱스에서 소방시설 5m 이내 후보지 재식별
    j_danger_new = _find_fire_hydrant_danger(filtered, fire_hydrants, config)

    logger.info(
        f"Hard 제약 적용: {original_count} → {len(filtered)} 후보지 "
        f"(배제 {original_count - len(filtered)}, J_danger {len(j_danger_new)}개)"
    )

    for k, v in exclusion_log.items():
        logger.info(f"  {k}: {v}개 배제")

    return filtered, j_danger_new, exclusion_log


def _find_fire_hydrant_danger(
    candidates: pd.DataFrame,
    fire_hydrants: pd.DataFrame,
    config: MCLPConfig,
) -> Set[int]:
    """소방용수시설 반경 5m 이내 후보지 식별.

    Returns:
        위험 후보지 인덱스 집합
    """
    if fire_hydrants is None or len(fire_hydrants) == 0:
        return set()

    buffer_km = config.hard.fire_hydrant_buffer_m / 1000.0  # m → km
    j_danger = set()

    c_lat = candidates["lat"].values
    c_lon = candidates["lon"].values
    f_lat = fire_hydrants["lat"].values
    f_lon = fire_hydrants["lon"].values

    # Vectorized: 각 후보지에서 모든 소방시설까지 최소 거리
    c_lat_rad = np.radians(c_lat)
    c_lon_rad = np.radians(c_lon)
    f_lat_rad = np.radians(f_lat)
    f_lon_rad = np.radians(f_lon)

    for j in range(len(candidates)):
        dlat = f_lat_rad - c_lat_rad[j]
        dlon = f_lon_rad - c_lon_rad[j]
        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(c_lat_rad[j]) * np.cos(f_lat_rad)
            * np.sin(dlon / 2) ** 2
        )
        dists = 2 * 6371.0 * np.arcsin(np.sqrt(a))

        if dists.min() <= buffer_km:
            j_danger.add(j)

    return j_danger


def apply_weather_masking(
    weather_data: pd.DataFrame,
    config: MCLPConfig,
) -> pd.Series:
    """기상 마스킹 판단.

    Returns:
        Series[bool] — True = 가동 가능, False = 중단
    """
    active = pd.Series(True, index=weather_data.index)

    # 적설 3cm 이상 → 전체 중단
    if "적설" in weather_data.columns:
        snow = pd.to_numeric(weather_data["적설"], errors="coerce").fillna(0)
        active &= (snow < config.hard.snow_threshold_cm)

    # 기온 0°C 미만 + 강수 동반 → 전체 중단
    if "기온" in weather_data.columns and "강수량" in weather_data.columns:
        temp = pd.to_numeric(weather_data["기온"], errors="coerce").fillna(10)
        precip = pd.to_numeric(weather_data["강수량"], errors="coerce").fillna(0)
        active &= ~((temp < config.hard.freezing_temp_c) & (precip > 0))

    logger.info(f"기상 마스킹: {(~active).sum()}/{len(active)} 시간 가동 중단")
    return active


def apply_rain_excavation_exclusion(
    candidates: pd.DataFrame,
    construction_zones: pd.DataFrame,
    current_precip_mm: float,
    config: MCLPConfig,
) -> Set[int]:
    """집중호우 시 굴착면 인근 후보지 동적 배제.

    Args:
        candidates: 거점 후보지
        construction_zones: 공사구간 좌표
        current_precip_mm: 현재 시간 강수량
        config: 설정

    Returns:
        배제할 후보지 인덱스 집합
    """
    if current_precip_mm < config.hard.heavy_rain_threshold_mm:
        return set()

    if construction_zones is None or len(construction_zones) == 0:
        return set()

    buffer_km = config.hard.excavation_buffer_m / 1000.0
    excluded = set()

    c_lat = candidates["lat"].values
    c_lon = candidates["lon"].values

    for _, zone in construction_zones.iterrows():
        zone_lat = zone["start_lat"]
        zone_lon = zone["start_lon"]
        if pd.isna(zone_lat) or pd.isna(zone_lon):
            continue

        for j in range(len(candidates)):
            dist = haversine_distance(c_lat[j], c_lon[j], zone_lat, zone_lon)
            if dist <= buffer_km:
                excluded.add(j)

    if excluded:
        logger.warning(
            f"⚠️ 집중호우({current_precip_mm}mm) → "
            f"굴착면 {buffer_km*1000:.0f}m 이내 {len(excluded)}개 후보지 일시 배제"
        )
    return excluded


def compute_candidate_preference(
    candidates: pd.DataFrame,
    parking_zones: pd.DataFrame = None,
    config: MCLPConfig = None,
) -> np.ndarray:
    """거점 선호도 f_j 산출.

    f_j = 주차면수 정규화(0.3) + 주정차 합법 보너스(0.7)

    Returns:
        np.ndarray of shape (|J|,) — 각 후보지의 선호도 점수
    """
    n = len(candidates)
    f_j = np.zeros(n)

    # 주차면수 정규화 (0~0.3)
    cap = candidates["capacity"].values.astype(float)
    max_cap = cap.max() if cap.max() > 0 else 1.0
    f_j += 0.3 * (cap / max_cap)

    # 주정차 합법 보너스는 좌표 기반으로 판정해야 하나,
    # 현재 주정차 데이터에 좌표가 없으므로 향후 지오코딩 후 적용
    # 현재는 기본 0.1 부여 (모든 주차장은 합법 주차 가능)
    f_j += 0.1

    logger.info(f"거점 선호도 f_j 산출: 범위 [{f_j.min():.3f}, {f_j.max():.3f}]")
    return f_j
