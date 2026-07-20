"""
visualize.py
대전 트램 공사 구간 네트워크 분석 결과 시각화
- Map 1: 전체 링크 매개중심성 (공사 전)
- Map 2: 공사 후 매개중심성 증가 Top 구간 (위험 스필오버 지역)
- Map 3: 공사 전/후 비교 레이어 (토글)
- Map 4: 구간 우선순위 통합 지도 (대시보드 메인)  ← v3 신규
"""

import os
import geopandas as gpd
import pandas as pd
import folium
from folium.plugins import MiniMap

# ─────────────────────────────────────────
# 0. 설정값
# ─────────────────────────────────────────
OUTPUT_DIR = "outputs"
DATA_DIR = "data"
MAP1_FILE = os.path.join(OUTPUT_DIR, "map1_free_flow_centrality.html")
MAP2_FILE = os.path.join(OUTPUT_DIR, "map2_risk_spillover.html")
MAP3_FILE = os.path.join(OUTPUT_DIR, "map3_before_after.html")
MAP4_FILE = os.path.join(OUTPUT_DIR, "map4_priority_dashboard.html")

CENTER = [36.3504, 127.3845]   # 대전시 중심 좌표

# ─────────────────────────────────────────
# 1. 데이터 로드
# ─────────────────────────────────────────
print("[Viz] Loading data...")
df_bc = pd.read_csv(os.path.join(OUTPUT_DIR, "network_betweenness.csv"))
df_seg = pd.read_csv(os.path.join(OUTPUT_DIR, "segment_priority.csv"))
gdf_link = gpd.read_file(os.path.join(DATA_DIR, "network", "daejeon_link.geojson"))
gdf_link["LINK_ID"] = gdf_link["LINK_ID"].astype(int)
gdf = gdf_link.merge(df_bc, on="LINK_ID", suffixes=("", "_y"))
df_stations = pd.read_csv(os.path.join(DATA_DIR, "network", "역_좌표.csv"))

# 공사 정보 로드
df_construction = pd.read_excel(
    os.path.join(DATA_DIR, "트램_공구별_통제현황.xlsx"), header=3
).dropna(subset=["공구"]).iloc[:25]
active_roads = set(df_construction[df_construction["상태(7/8기준)"] == "활성"]["노선명"].dropna())

# 우선순위 등급별 색상
GRADE_COLOR = {
    "🔴 심각": "#ff2020",
    "🟠 경고": "#ff8800",
    "🟡 주의": "#ffdd00",
    "🟢 정상": "#44cc44",
}

# ─────────────────────────────────────────
# 2. 색상 매핑 유틸
# ─────────────────────────────────────────
def bc_to_color(val, vmax, mode="free"):
    """매개중심성 값을 색상으로 변환"""
    ratio = min(val / vmax, 1.0) if vmax > 0 else 0
    if mode == "free":
        # 파랑(낮음) → 노랑(중간) → 빨강(높음)
        if ratio < 0.5:
            r = int(ratio * 2 * 255)
            g = int(ratio * 2 * 255)
            b = 255
        else:
            r = 255
            g = int((1 - ratio) * 2 * 255)
            b = 0
    else:
        # 변화량: 초록(감소) → 흰(무변화) → 빨강(급증)
        if ratio < 0.5:
            r = int(ratio * 2 * 255)
            g = 180
            b = int(ratio * 2 * 120)
        else:
            r = 255
            g = int((1 - ratio) * 2 * 180)
            b = 0
    return f"#{r:02x}{g:02x}{b:02x}"

# ─────────────────────────────────────────
# MAP 1: 공사 전 전체 매개중심성 지도
# ─────────────────────────────────────────
print("[Viz] Creating Map 1: Free-flow Betweenness Centrality...")
m1 = folium.Map(location=CENTER, zoom_start=12, tiles="cartodbdark_matter")
MiniMap(toggle_display=True).add_to(m1)

vmax_free = gdf["bc_free_flow"].quantile(0.99)
fg_links = folium.FeatureGroup(name="전체 도로망 매개중심성", show=True)

plotted = 0
for _, row in gdf.iterrows():
    geom = row["geometry"]
    bc_val = row["bc_free_flow"]
    if bc_val <= 0 or geom.geom_type != "LineString":
        continue
    coords = [[y, x] for x, y in geom.coords]
    color = bc_to_color(bc_val, vmax_free, mode="free")
    weight = 1 + int(min(bc_val / vmax_free, 1.0) * 5)
    folium.PolyLine(
        coords,
        color=color,
        weight=weight,
        opacity=0.6,
        tooltip=f"{row['ROAD_NAME_x'] if 'ROAD_NAME_x' in row else row.get('ROAD_NAME','')} | BC: {bc_val:.5f}",
    ).add_to(fg_links)
    plotted += 1

fg_links.add_to(m1)

# 정거장 마커
fg_stations = folium.FeatureGroup(name="트램 정거장 (45개)", show=True)
for _, s in df_stations.iterrows():
    folium.CircleMarker(
        location=[s["lat"], s["lon"]],
        radius=5,
        color="#00e5ff",
        fill=True,
        fill_color="#00e5ff",
        fill_opacity=0.9,
        tooltip=f"Station {s['station_no']}: {s['station_name']}",
    ).add_to(fg_stations)
fg_stations.add_to(m1)

folium.LayerControl(collapsed=False).add_to(m1)

# 범례 HTML
legend_html = """
<div style="position:fixed; bottom:30px; left:30px; z-index:9999;
     background:rgba(20,20,30,0.88); color:white; padding:14px 18px;
     border-radius:10px; font-family:sans-serif; font-size:13px;
     border:1px solid rgba(255,255,255,0.2);">
  <b>🔵 매개중심성 (공사 전)</b><br>
  <span style="background:linear-gradient(to right,#0000ff,#ffff00,#ff0000);
    display:inline-block;width:160px;height:12px;border-radius:6px;margin-top:6px;"></span><br>
  <small>낮음 ← 중간 → 높음</small><br><br>
  <span style="color:#00e5ff;">●</span> 트램 정거장<br>
  <small style="color:#aaa">높을수록 차단 시 파급력 큼</small>
</div>
"""
m1.get_root().html.add_child(folium.Element(legend_html))
m1.save(MAP1_FILE)
print(f"  → Saved: {MAP1_FILE}  ({plotted} links plotted)")

# ─────────────────────────────────────────
# MAP 2: 공사 후 매개중심성 급증 (위험 구간)
# ─────────────────────────────────────────
print("[Viz] Creating Map 2: Spillover Risk Zones (top centrality increase)...")
m2 = folium.Map(location=CENTER, zoom_start=12, tiles="cartodbdark_matter")
MiniMap(toggle_display=True).add_to(m2)

# 상위 400개 변화 링크만
top_n = gdf.sort_values("bc_change", ascending=False).head(400)
vmax_change = top_n["bc_change"].quantile(0.97)

fg_risk = folium.FeatureGroup(name="정체 전이 위험 구간 (Top)", show=True)
for _, row in top_n.iterrows():
    geom = row["geometry"]
    if geom.geom_type != "LineString":
        continue
    val = row["bc_change"]
    if val <= 0:
        continue
    coords = [[y, x] for x, y in geom.coords]
    ratio = min(val / vmax_change, 1.0)
    weight = 3 + int(ratio * 7)
    # 위험도 색상: 주황 → 빨강
    r = 255
    g = int((1 - ratio) * 120)
    b = 0
    color = f"#{r:02x}{g:02x}{b:02x}"
    road_name = row.get("ROAD_NAME_x", row.get("ROAD_NAME", ""))
    popup_html = f"""
    <div style='font-family:sans-serif;font-size:13px;min-width:220px'>
      <b>⚠️ 정체 전이 위험 구간</b><br>
      <b>도로명:</b> {road_name}<br>
      <b>인접 정거장:</b> {row['nearest_station_name']}<br>
      <b>BC (공사 전):</b> {row['bc_free_flow']:.6f}<br>
      <b>BC (공사 중):</b> {row['bc_under_construction']:.6f}<br>
      <b>변화량 ▲:</b> <span style='color:red;font-weight:bold'>{val:.6f}</span>
    </div>"""
    folium.PolyLine(
        coords,
        color=color,
        weight=weight,
        opacity=0.85,
        tooltip=f"⚠ {road_name} | 위험 증가: {val:.5f}",
        popup=folium.Popup(popup_html, max_width=280),
    ).add_to(fg_risk)
fg_risk.add_to(m2)

# 정거장
fg_stations2 = folium.FeatureGroup(name="트램 정거장", show=True)
for _, s in df_stations.iterrows():
    folium.CircleMarker(
        location=[s["lat"], s["lon"]],
        radius=5,
        color="#00e5ff",
        fill=True,
        fill_color="#00e5ff",
        fill_opacity=0.9,
        tooltip=f"Station {s['station_no']}: {s['station_name']}",
    ).add_to(fg_stations2)
fg_stations2.add_to(m2)

folium.LayerControl(collapsed=False).add_to(m2)

legend2_html = """
<div style="position:fixed; bottom:30px; left:30px; z-index:9999;
     background:rgba(20,20,30,0.88); color:white; padding:14px 18px;
     border-radius:10px; font-family:sans-serif; font-size:13px;
     border:1px solid rgba(255,255,255,0.2);">
  <b>🔴 공사 후 매개중심성 급증 구간</b><br>
  <span style="background:linear-gradient(to right,#ff7700,#ff0000);
    display:inline-block;width:160px;height:12px;border-radius:6px;margin-top:6px;"></span><br>
  <small>밝을수록 정체 전이 위험 큼</small><br><br>
  <span style="color:#00e5ff;">●</span> 트램 정거장<br>
  <small style="color:#aaa">선 굵기 = 위험도 비례</small>
</div>
"""
m2.get_root().html.add_child(folium.Element(legend2_html))
m2.save(MAP2_FILE)
print(f"  → Saved: {MAP2_FILE}")

# ─────────────────────────────────────────
# MAP 3: 공사 전 vs 공사 후 비교 레이어 (토글)
# ─────────────────────────────────────────
print("[Viz] Creating Map 3: Before vs After comparison (toggleable layers)...")
m3 = folium.Map(location=CENTER, zoom_start=12, tiles="cartodbdark_matter")
MiniMap(toggle_display=True).add_to(m3)

fg_before = folium.FeatureGroup(name="🔵 공사 전 (Free Flow)", show=True)
fg_after  = folium.FeatureGroup(name="🔴 공사 중 (Under Construction)", show=False)

vmax_b = gdf["bc_free_flow"].quantile(0.99)
vmax_a = gdf["bc_under_construction"].quantile(0.99)

for _, row in gdf.iterrows():
    geom = row["geometry"]
    if geom.geom_type != "LineString":
        continue
    coords = [[y, x] for x, y in geom.coords]

    bc_b = row["bc_free_flow"]
    bc_a = row["bc_under_construction"]
    road_name = row.get("ROAD_NAME_x", row.get("ROAD_NAME", ""))

    if bc_b > 0:
        color_b = bc_to_color(bc_b, vmax_b, mode="free")
        w_b = 1 + int(min(bc_b / vmax_b, 1.0) * 4)
        folium.PolyLine(
            coords, color=color_b, weight=w_b, opacity=0.65,
            tooltip=f"{road_name} BC(전): {bc_b:.5f}"
        ).add_to(fg_before)

    if bc_a > 0:
        color_a = bc_to_color(bc_a, vmax_a, mode="free")
        w_a = 1 + int(min(bc_a / vmax_a, 1.0) * 4)
        folium.PolyLine(
            coords, color=color_a, weight=w_a, opacity=0.65,
            tooltip=f"{road_name} BC(후): {bc_a:.5f}"
        ).add_to(fg_after)

fg_before.add_to(m3)
fg_after.add_to(m3)

# 정거장
fg_stations3 = folium.FeatureGroup(name="트램 정거장", show=True)
for _, s in df_stations.iterrows():
    folium.CircleMarker(
        location=[s["lat"], s["lon"]],
        radius=5,
        color="#00e5ff",
        fill=True,
        fill_color="#00e5ff",
        fill_opacity=0.9,
        tooltip=f"{s['station_name']}",
    ).add_to(fg_stations3)
fg_stations3.add_to(m3)

folium.LayerControl(collapsed=False).add_to(m3)

legend3_html = """
<div style="position:fixed; bottom:30px; left:30px; z-index:9999;
     background:rgba(20,20,30,0.88); color:white; padding:14px 18px;
     border-radius:10px; font-family:sans-serif; font-size:13px;
     border:1px solid rgba(255,255,255,0.2);">
  <b>⚖️ 공사 전/후 매개중심성 비교</b><br>
  <small style='color:#aaa'>우측 레이어 토글로 전환</small><br><br>
  <span style="background:linear-gradient(to right,#0000ff,#ffff00,#ff0000);
    display:inline-block;width:130px;height:10px;border-radius:4px;margin-top:4px;"></span><br>
  <small>낮음 → 높음</small><br><br>
  <span style="color:#00e5ff;">●</span> 트램 정거장
</div>
"""
m3.get_root().html.add_child(folium.Element(legend3_html))
m3.save(MAP3_FILE)
print(f"  → Saved: {MAP3_FILE}")

print("\n[Viz] 시각화 완료!")
print(f"  Map 1 (공사 전 전체): {MAP1_FILE}")
print(f"  Map 2 (위험 급증):    {MAP2_FILE}")
print(f"  Map 3 (전/후 비교):   {MAP3_FILE}")


# ─────────────────────────────────────────
# MAP 4: 구간 우선순위 통합 대시보드 지도
# ─────────────────────────────────────────
print("[Viz] Creating Map 4: Priority Dashboard ...")
m4 = folium.Map(location=CENTER, zoom_start=12, tiles="cartodbdark_matter")
MiniMap(toggle_display=True).add_to(m4)

# 레이어: 등급별 분리
fg_critical = folium.FeatureGroup(name="🔴 심각 구간", show=True)
fg_warning  = folium.FeatureGroup(name="🟠 경고 구간", show=True)
fg_caution  = folium.FeatureGroup(name="🟡 주의 구간", show=True)
fg_normal   = folium.FeatureGroup(name="🟢 정상 구간", show=False)
fg_stn4     = folium.FeatureGroup(name="트램 정거장", show=True)

# segment_id별 우선순위 정보 딕셔너리
seg_info = df_seg.set_index('segment_id').to_dict('index')

for _, row in gdf.iterrows():
    geom = row["geometry"]
    if geom.geom_type != "LineString":
        continue
    if 'segment_id' not in row or 'priority_score' not in row:
        continue

    score = row.get('priority_score', 0) or 0
    # 정상(낮은점수) 링크는 show=False 레이어에만 추가 — 초기 렌더링 부하 감소
    # BC > 0 없는 링크는 완전 생략
    if row.get('bc_under_construction', 0) <= 0 and score < 5:
        continue
    grade = row.get('risk_grade', '🟢 정상') or '🟢 정상'
    seg_id = row.get('segment_id', '')
    road_name = row.get("ROAD_NAME_x", row.get("ROAD_NAME", ""))

    color = GRADE_COLOR.get(grade, "#44cc44")
    weight = 2 + int(min(score / 100, 1.0) * 6)
    coords = [[y, x] for x, y in geom.coords]

    seg_data = seg_info.get(seg_id, {})
    seg_name = seg_data.get('segment_name', seg_id)

    popup_html = f"""
    <div style='font-family:sans-serif;font-size:13px;min-width:260px;
                background:#1a1a2e;color:#eee;padding:12px;border-radius:8px'>
      <div style='font-size:15px;font-weight:bold;margin-bottom:8px'>
        {grade} {road_name}
      </div>
      <table style='width:100%;border-collapse:collapse'>
        <tr><td style='color:#aaa;padding:2px 0'>구간</td>
            <td style='text-align:right'>{seg_name}</td></tr>
        <tr><td style='color:#aaa;padding:2px 0'>우선순위 점수</td>
            <td style='text-align:right;color:{color};font-weight:bold'>{score:.1f} / 100</td></tr>
        <tr><td style='color:#aaa;padding:2px 0'>BC (공사중)</td>
            <td style='text-align:right'>{row['bc_under_construction']:.5f}</td></tr>
        <tr><td style='color:#aaa;padding:2px 0'>BC 변화량</td>
            <td style='text-align:right;color:#ff8800'>{row['bc_change']:+.5f}</td></tr>
        <tr><td style='color:#aaa;padding:2px 0'>잔여 차로 비율</td>
            <td style='text-align:right'>{row['lane_remain_ratio']:.0%}</td></tr>
        <tr><td style='color:#aaa;padding:2px 0'>속도 패널티</td>
            <td style='text-align:right'>{row['speed_penalty_multiplier']:.2f}×</td></tr>
      </table>
    </div>"""

    line = folium.PolyLine(
        coords, color=color, weight=weight, opacity=0.85,
        tooltip=f"{grade} {road_name} | 점수: {score:.1f}",
        popup=folium.Popup(popup_html, max_width=300),
    )

    if grade == "🔴 심각":
        line.add_to(fg_critical)
    elif grade == "🟠 경고":
        line.add_to(fg_warning)
    elif grade == "🟡 주의":
        line.add_to(fg_caution)
    else:
        line.add_to(fg_normal)

# 정거장 마커 (순위 표시)
for _, s in df_stations.iterrows():
    folium.CircleMarker(
        location=[s["lat"], s["lon"]],
        radius=6, color="#00e5ff", fill=True,
        fill_color="#00e5ff", fill_opacity=0.9,
        tooltip=f"STN{s['station_no']}: {s['station_name']}",
    ).add_to(fg_stn4)

fg_critical.add_to(m4)
fg_warning.add_to(m4)
fg_caution.add_to(m4)
fg_normal.add_to(m4)
fg_stn4.add_to(m4)
folium.LayerControl(collapsed=False).add_to(m4)

# 우선순위 상위 10 구간 사이드 패널
top10 = df_seg.head(10)
panel_rows = ""
for _, r in top10.iterrows():
    c = GRADE_COLOR.get(r['risk_grade'], '#44cc44')
    panel_rows += (
        f"<tr><td style='color:{c};font-weight:bold'>{int(r['rank'])}</td>"
        f"<td>{r['segment_name']}</td>"
        f"<td style='color:{c};font-weight:bold;text-align:right'>{r['priority_score']:.1f}</td>"
        f"<td style='text-align:right'>{r['risk_grade']}</td></tr>"
    )

panel_html = f"""
<div style="position:fixed; top:60px; right:10px; z-index:9999; width:380px;
     background:rgba(15,15,25,0.93); color:white; padding:14px 16px;
     border-radius:10px; font-family:sans-serif; font-size:12px;
     border:1px solid rgba(255,255,255,0.15); max-height:420px; overflow-y:auto">
  <div style='font-size:14px;font-weight:bold;margin-bottom:10px'>
    📊 구간 우선순위 Top 10
  </div>
  <table style='width:100%;border-collapse:collapse'>
    <thead>
      <tr style='color:#aaa;border-bottom:1px solid #333'>
        <th style='text-align:left;padding:3px 0'>#</th>
        <th style='text-align:left'>구간</th>
        <th style='text-align:right'>점수</th>
        <th style='text-align:right'>등급</th>
      </tr>
    </thead>
    <tbody>{panel_rows}</tbody>
  </table>
  <div style='margin-top:10px;color:#888;font-size:11px'>
    BC(40%) + 차로감소(30%) + 도로위계(20%) + 기상(10%)
  </div>
</div>"""
m4.get_root().html.add_child(folium.Element(panel_html))

legend4_html = """
<div style="position:fixed; bottom:30px; left:30px; z-index:9999;
     background:rgba(15,15,25,0.93); color:white; padding:14px 18px;
     border-radius:10px; font-family:sans-serif; font-size:13px;
     border:1px solid rgba(255,255,255,0.15);">
  <b>🗺 구간 우선순위 등급</b><br><br>
  <span style="color:#ff2020;">━━</span> 🔴 심각 (≥80점)<br>
  <span style="color:#ff8800;">━━</span> 🟠 경고 (60~79점)<br>
  <span style="color:#ffdd00;">━━</span> 🟡 주의 (40~59점)<br>
  <span style="color:#44cc44;">━━</span> 🟢 정상 (&lt;40점)<br><br>
  <span style="color:#00e5ff;">●</span> 트램 정거장<br>
  <small style="color:#aaa">선 굵기 = 점수 비례</small>
</div>"""
m4.get_root().html.add_child(folium.Element(legend4_html))
m4.save(MAP4_FILE)
print(f"  → Saved: {MAP4_FILE}")

print("\n[Viz] 시각화 완료!")
print(f"  Map 1 (공사 전 전체):    {MAP1_FILE}")
print(f"  Map 2 (위험 급증):       {MAP2_FILE}")
print(f"  Map 3 (전/후 비교):      {MAP3_FILE}")
print(f"  Map 4 (우선순위 대시보드): {MAP4_FILE}")
