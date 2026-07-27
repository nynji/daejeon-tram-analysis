"""
대표 시나리오(정부청사 <-> 둔산, 2026-06-29 18:30) folium 시각화.

정적경로(회색), 예측반영경로(파란 굵은선), 회피 구간(빨간 점선 + prob_risk
팝업)을 한 지도에 겹쳐 그리고 상단에 범례·요약 텍스트를 넣는다.

build_route_comparison_map()은 다른 스크립트(routing/dev/generate_case_study_maps.py
등)에서 import해서 재사용하는 공용 함수다 - 이 파일을 직접 실행할 필요 없이
함수만 가져다 쓸 수 있다.
"""

import os
import sys

import folium
import pandas as pd

sys.path.insert(0, "routing/scripts")
import pipeline as pl

ORIGIN = "정부청사"
DESTINATION = "둔산"
REPLAY_DATE = "2026-06-29"
T0_TIME = "18:30"

OUT_HTML = "routing/dataset/output/demo_map.html"

STATIC_COLOR = "#4d4d4d"
PREDICTED_COLOR = "#1f5fd6"
AVOIDED_COLOR = "#d62728"


def node_latlon(node_coords: pd.DataFrame, node_id: str):
    row = node_coords[node_coords["NODE_ID"] == node_id].iloc[0]
    return float(row["lat"]), float(row["lon"])


def path_to_latlon(node_coords: pd.DataFrame, path: list) -> list:
    return [node_latlon(node_coords, n) for n in path]


def build_route_comparison_map(result: dict, out_html: str, origin_label: str, destination_label: str,
                                extra_note_html: str = "") -> None:
    """정적경로/예측반영경로/회피구간을 한 지도에 겹쳐 그리고 저장한다.
    result: run_routing_scenario()의 반환 dict 그대로.
    extra_note_html: 요약 패널 상단에 덧붙일 설명 문단(선택)."""
    node_coords = pl._load_node_coords()

    static_latlon = path_to_latlon(node_coords, result["static_path"])
    predicted_latlon = path_to_latlon(node_coords, result["predicted_path"])

    center_lat = sum(p[0] for p in static_latlon) / len(static_latlon)
    center_lon = sum(p[1] for p in static_latlon) / len(static_latlon)

    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles="cartodbpositron")

    # 회피구간은 정적경로와 좌표가 거의 겹친다 - 위에 덧그리면 정적경로 선을
    # 완전히 가려버리므로, "뒤에 까는 굵고 옅은 하이라이트"로 먼저 그리고
    # 그 위에 정적경로 실선을 나중에(=위에) 그린다. Leaflet/folium은 나중에
    # add_to한 레이어가 위에 렌더링된다.
    for link in result["avoided_links"]:
        u_latlon = node_latlon(node_coords, link["u"])
        v_latlon = node_latlon(node_coords, link["v"])
        seg_label = f"{link['segment_id']}_{link['direction']}" if link["segment_id"] else "(미매핑 링크)"
        prob_risk = link["prob_risk"]
        prob_risk_str = f"{prob_risk:.3f}" if prob_risk is not None else "N/A"
        popup_html = (
            f"<b>회피 구간</b><br>"
            f"LINK_ID: {link['link_id']}<br>"
            f"도로명: {link['road_name']}<br>"
            f"segment: {seg_label}<br>"
            f"prob_risk: {prob_risk_str}"
        )
        folium.PolyLine(
            [u_latlon, v_latlon], color=AVOIDED_COLOR, weight=14, opacity=0.35,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"회피: {link['road_name']} (prob_risk={prob_risk_str})",
        ).add_to(fmap)

    folium.PolyLine(
        static_latlon, color=STATIC_COLOR, weight=5, opacity=1.0,
        tooltip=f"정적 경로 (순수 이동시간 {result['static_time_min']:.2f}분)",
    ).add_to(fmap)

    folium.PolyLine(
        predicted_latlon, color=PREDICTED_COLOR, weight=6, opacity=1.0,
        tooltip=f"예측반영 경로 (순수 이동시간 {result['predicted_time_min']:.2f}분)",
    ).add_to(fmap)

    folium.Marker(
        static_latlon[0], tooltip=f"출발: {origin_label}",
        icon=folium.Icon(color="green", icon="play"),
    ).add_to(fmap)
    folium.Marker(
        static_latlon[-1], tooltip=f"도착: {destination_label}",
        icon=folium.Icon(color="red", icon="stop"),
    ).add_to(fmap)

    delta_time_pct = (result["predicted_time_min"] - result["static_time_min"]) / result["static_time_min"] * 100 \
        if result["static_time_min"] else 0.0
    delta_risk = result["static_risk_exposure"] - result["predicted_risk_exposure"]

    summary_html = f"""
    <div style="
        position: fixed; top: 12px; left: 60px; z-index: 9999;
        background: white; padding: 12px 16px; border: 2px solid #333;
        border-radius: 8px; font-family: sans-serif; font-size: 13px;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.3); max-width: 360px;">
      <b style="font-size:14px;">{origin_label} → {destination_label} 리플레이</b><br>
      <span style="color:#888;">{result['t0']}</span>
      {extra_note_html}
      <hr style="margin:6px 0;">
      <span style="color:{STATIC_COLOR};">━━</span> 정적 경로: {result['static_time_min']:.2f}분,
        위험노출량 {result['static_risk_exposure']:.3f}<br>
      <span style="color:{PREDICTED_COLOR}; font-weight:bold;">━━</span> 예측반영 경로: {result['predicted_time_min']:.2f}분,
        위험노출량 {result['predicted_risk_exposure']:.3f}<br>
      <span style="color:{AVOIDED_COLOR};">┅┅</span> 회피 구간 ({len(result['avoided_links'])}개)<br>
      <hr style="margin:6px 0;">
      Δ시간: <b>{delta_time_pct:+.1f}%</b> / Δ위험노출량 개선: <b>{delta_risk:+.3f}</b>
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(summary_html))

    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    fmap.save(out_html)


def main():
    print(f"시나리오 실행: {ORIGIN} -> {DESTINATION} @ {REPLAY_DATE} {T0_TIME}")
    result = pl.run_routing_scenario(ORIGIN, DESTINATION, REPLAY_DATE, T0_TIME)
    build_route_comparison_map(result, OUT_HTML, ORIGIN, DESTINATION)
    print(f"저장 완료: {OUT_HTML}")


if __name__ == "__main__":
    main()
