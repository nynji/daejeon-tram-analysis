"""
거리 행렬 산출 모듈
==================
Haversine 공식 기반 거리 행렬 + 커버리지 집합 생성
"""

import numpy as np
import pandas as pd
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

# 지구 반지름 (km)
EARTH_RADIUS_KM = 6371.0


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """두 지점 간 Haversine 거리 (km)."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def haversine_matrix(
    candidates: pd.DataFrame,
    demand_points: pd.DataFrame,
) -> np.ndarray:
    """거점 후보지 J와 수요지점 I 간 거리 행렬 산출.

    Args:
        candidates: 거점 후보지 DataFrame (lat, lon 필수)
        demand_points: 수요지점 DataFrame (lat, lon 필수)

    Returns:
        np.ndarray of shape (|J|, |I|) — 거리(km)
    """
    j_lat = np.radians(candidates["lat"].values)
    j_lon = np.radians(candidates["lon"].values)
    i_lat = np.radians(demand_points["lat"].values)
    i_lon = np.radians(demand_points["lon"].values)

    # Broadcasting: (|J|, 1) vs (1, |I|)
    dlat = i_lat[np.newaxis, :] - j_lat[:, np.newaxis]
    dlon = i_lon[np.newaxis, :] - j_lon[:, np.newaxis]

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(j_lat[:, np.newaxis]) * np.cos(i_lat[np.newaxis, :]) * np.sin(dlon / 2) ** 2
    )
    dist_km = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))

    logger.info(f"거리 행렬 산출: {dist_km.shape[0]} 후보지 × {dist_km.shape[1]} 수요지점")
    return dist_km


def build_coverage_sets(
    dist_matrix: np.ndarray,
    coverage_radius_km: float,
) -> dict:
    """커버리지 집합 N_j 생성.

    Args:
        dist_matrix: (|J|, |I|) 거리 행렬
        coverage_radius_km: 커버리지 반경

    Returns:
        dict[j] = list of demand_point indices covered by candidate j
    """
    coverage = {}
    n_j, n_i = dist_matrix.shape

    for j in range(n_j):
        covered = np.where(dist_matrix[j, :] <= coverage_radius_km)[0].tolist()
        coverage[j] = covered

    total_covered = len(set(i for indices in coverage.values() for i in indices))
    logger.info(
        f"커버리지 집합 생성: R={coverage_radius_km}km, "
        f"커버 가능 수요지점 {total_covered}/{n_i} ({100*total_covered/n_i:.1f}%)"
    )
    return coverage


def candidate_distance_matrix(candidates: pd.DataFrame) -> np.ndarray:
    """거점 후보지 간 거리 행렬 (거점 간 최소 거리 제약용).

    Returns:
        np.ndarray of shape (|J|, |J|) — 거리(km)
    """
    n = len(candidates)
    lat = np.radians(candidates["lat"].values)
    lon = np.radians(candidates["lon"].values)

    dlat = lat[:, np.newaxis] - lat[np.newaxis, :]
    dlon = lon[:, np.newaxis] - lon[np.newaxis, :]

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat[:, np.newaxis]) * np.cos(lat[np.newaxis, :]) * np.sin(dlon / 2) ** 2
    )
    dist_km = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))

    return dist_km
