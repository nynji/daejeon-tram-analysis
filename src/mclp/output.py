"""
결과 출력 모듈
==============
CSV + Folium 지도 출력
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Set
import logging

try:
    import folium
    from folium.plugins import MarkerCluster
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

from .solver import CMCLPResult
from .config import MCLPConfig

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"


def save_results_csv(
    result: CMCLPResult,
    candidates: pd.DataFrame,
    demand_points: pd.DataFrame,
    config: MCLPConfig,
) -> None:
    """최적화 결과를 CSV로 저장.

    - outputs/mclp_optimal_anchors.csv: 선택된 거점 목록
    - outputs/mclp_summary.csv: 요약 통계
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ─── 거점 목록 CSV ───
    if result.selected_facilities:
        anchor_rows = []
        for j in result.selected_facilities:
            assigned_demands = [i for i, fj in result.assignments.items() if fj == j]
            row = {
                "거점ID": j,
                "주차장명": candidates.iloc[j]["name"],
                "위도": candidates.iloc[j]["lat"],
                "경도": candidates.iloc[j]["lon"],
                "주차면수": candidates.iloc[j]["capacity"],
                "커버_수요지점수": len(assigned_demands),
                "커버리지_비율": len(assigned_demands) / result.total_demand if result.total_demand > 0 else 0,
            }
            anchor_rows.append(row)

        anchor_df = pd.DataFrame(anchor_rows)
        anchor_df = anchor_df.sort_values("커버_수요지점수", ascending=False)
        anchor_path = OUTPUT_DIR / "mclp_optimal_anchors.csv"
        anchor_df.to_csv(anchor_path, index=False, encoding="utf-8-sig")
        logger.info(f"거점 목록 저장: {anchor_path}")

    # ─── 요약 CSV ───
    summary = pd.DataFrame([{
        "총_수요지점수": result.total_demand,
        "커버된_수요지점수": len(result.covered_demands),
        "커버리지_비율": result.coverage_ratio,
        "선택_거점수": len(result.selected_facilities),
        "최대_거점수(P)": config.solver.P,
        "커버리지_반경(km)": config.solver.coverage_radius_km,
        "목적함수_값": result.objective_value,
        "실행시간(초)": result.solve_time_sec,
        "솔버_상태": result.status,
    }])
    summary_path = OUTPUT_DIR / "mclp_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    logger.info(f"요약 저장: {summary_path}")


def save_map(
    result: CMCLPResult,
    candidates: pd.DataFrame,
    demand_points: pd.DataFrame,
    config: MCLPConfig,
    j_danger: Set[int] = None,
    construction_zones: pd.DataFrame = None,
) -> None:
    """Folium 기반 대화형 지도 생성."""
    if not HAS_FOLIUM:
        logger.warning("folium 미설치, 지도 생성 건너뜀. pip install folium")
        return

    # 대전 중심
    center_lat = 36.35
    center_lon = 127.385

    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="CartoDB positron")

    # ─── 선택된 거점 (빨간 마커) ───
    anchor_group = folium.FeatureGroup(name="✅ 선택 거점", show=True)
    for j in result.selected_facilities:
        row = candidates.iloc[j]
        assigned = sum(1 for fj in result.assignments.values() if fj == j)
        popup_text = (
            f"<b>{row['name']}</b><br>"
            f"주차면수: {row['capacity']}<br>"
            f"커버 수요: {assigned}개<br>"
            f"좌표: ({row['lat']:.5f}, {row['lon']:.5f})"
        )
        folium.Marker(
            location=[row["lat"], row["lon"]],
            popup=folium.Popup(popup_text, max_width=250),
            icon=folium.Icon(color="red", icon="star", prefix="fa"),
        ).add_to(anchor_group)

        # 커버리지 반경 원
        folium.Circle(
            location=[row["lat"], row["lon"]],
            radius=config.solver.coverage_radius_km * 1000,
            color="red",
            fill=True,
            fill_opacity=0.08,
            weight=1,
        ).add_to(anchor_group)
    anchor_group.add_to(m)

    # ─── 커버된 수요지점 (파란, 클러스터) ───
    covered_group = folium.FeatureGroup(name="🔵 커버된 수요", show=False)
    covered_cluster = MarkerCluster().add_to(covered_group)
    for i in result.covered_demands[:5000]:  # 성능상 상위 5000개
        row = demand_points.iloc[i]
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=2,
            color="blue",
            fill=True,
            fill_opacity=0.5,
        ).add_to(covered_cluster)
    covered_group.add_to(m)

    # ─── 미커버 수요지점 (회색) ───
    uncovered_ids = set(range(len(demand_points))) - set(result.covered_demands)
    if uncovered_ids:
        uncovered_group = folium.FeatureGroup(name="⚪ 미커버 수요", show=False)
        uncovered_cluster = MarkerCluster().add_to(uncovered_group)
        for i in list(uncovered_ids)[:2000]:
            row = demand_points.iloc[i]
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=2,
                color="gray",
                fill=True,
                fill_opacity=0.3,
            ).add_to(uncovered_cluster)
        uncovered_group.add_to(m)

    # ─── J_danger (소방시설, 기본 비활성) ───
    if j_danger:
        danger_group = folium.FeatureGroup(name="🚒 소방시설 배제구역", show=False)
        for j in j_danger:
            if j < len(candidates):
                row = candidates.iloc[j]
                folium.CircleMarker(
                    location=[row["lat"], row["lon"]],
                    radius=5,
                    color="orange",
                    fill=True,
                    fill_opacity=0.6,
                ).add_to(danger_group)
        danger_group.add_to(m)

    # ─── 공사구간 (주황색 라인) ───
    if construction_zones is not None and len(construction_zones) > 0:
        cz_group = folium.FeatureGroup(name="🚧 공사구간", show=True)
        for _, zone in construction_zones.iterrows():
            if pd.notna(zone["start_lat"]) and pd.notna(zone["end_lat"]):
                folium.PolyLine(
                    locations=[
                        [zone["start_lat"], zone["start_lon"]],
                        [zone["end_lat"], zone["end_lon"]],
                    ],
                    color="orange",
                    weight=4,
                    opacity=0.8,
                    popup=zone["zone_name"],
                ).add_to(cz_group)
        cz_group.add_to(m)

    # 레이어 컨트롤
    folium.LayerControl(collapsed=False).add_to(m)

    # 저장
    map_path = OUTPUT_DIR / "mclp_anchor_map.html"
    m.save(str(map_path))
    logger.info(f"지도 저장: {map_path}")


def save_scenario_comparison(scenarios: list) -> None:
    """시나리오 비교 결과 저장."""
    if not scenarios:
        return

    rows = []
    for s in scenarios:
        rows.append({
            "시나리오": s["name"],
            "P": s["P"],
            "Coverage_Radius_km": s["radius"],
            "선택_거점수": s["n_selected"],
            "커버리지_비율": s["coverage_ratio"],
            "목적함수_값": s["objective"],
            "실행시간_초": s["time"],
        })

    df = pd.DataFrame(rows)
    path = OUTPUT_DIR / "mclp_scenario_comparison.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info(f"시나리오 비교 저장: {path}")
