"""
network_simulation.py
공사 구간 속도 패널티 시뮬레이터

패널티 산정 근거 (잔여 차로 비율 기반)
────────────────────────────────────────────────────────────────
교통공학 기본 원리: 도로 용량(capacity)은 차로 수에 비례하고,
속도-유량 관계(Speed-Flow Relationship, HCM 2010)에 따라
용량 감소 → 밀도 증가 → 속도 저하가 발생한다.

단순화 모델:  speed_ratio ≈ f(잔여차로비율)
  잔여비율 > 0.75 → 속도 영향 소 → 패널티 0.75  (편도 1개 감소 수준)
  잔여비율 0.50~0.75 → 중간 영향 → 패널티 0.55
  잔여비율 0.35~0.50 → 심각 영향 → 패널티 0.40
  잔여비율 < 0.35     → 거의 차단 → 패널티 0.25

근거 문헌:
- HCM 2010 (Highway Capacity Manual) Chapter 10: Freeway Capacity
- 대전시 트램 교통대책 3단계 기준 (20km/h, 15km/h 임계)
  · 20km/h = 자유속도 50km/h 대비 40% 수준
  · 15km/h = 자유속도 50km/h 대비 30% 수준
  → 패널티 하한 0.25~0.30 설정의 근거

실제 공사 데이터 기반 잔여 차로 비율 (트램_공구별_통제현황.xlsx):
  - 계족로 2공구:   전체4 → 폐쇄3 → 잔여25%  → 패널티 0.25
  - 동광장로 13공구: 전체4 → 폐쇄2 → 잔여50%  → 패널티 0.40
  - 계백로 12공구:  전체8 → 폐쇄3 → 잔여62%   → 패널티 0.55
  - 한밭대로 3공구: 전체11→ 폐쇄3 → 잔여73%   → 패널티 0.75
  - 도안대로 7공구: 전체12→ 폐쇄2 → 잔여83%   → 패널티 0.75
────────────────────────────────────────────────────────────────
"""

import re
import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────
# 패널티 계수 결정 함수
# ─────────────────────────────────────────────────────────────

def _parse_lane_info(method_str: str):
    """
    통제방법(차로) 문자열에서 (총차로, 폐쇄차로) 숫자를 추출한다.

    예시 입력:
      "왕복 11개차로 중 편도 3개차로"  → total=11, closed=3
      "왕복 9개차로 중 2개차로"        → total=9,  closed=2
      "편도3차로 중 1차로"             → total=3,  closed=1
      "왕복12차로 중 중앙2차로"        → total=12, closed=2
      "1~4차로 단계별통제"             → total=4,  closed=2  (범위 중간값)
      "부분통제"                        → None, None
    """
    if not method_str or str(method_str).strip() in ('nan', '부분통제', '미상'):
        return None, None

    s = str(method_str).strip()

    # 패턴 1: "왕복 N개차로 중 M개차로" 또는 "편도N차로 중 M차로"
    # "왕복 6~8개차로 중 2~4개차로" → total=7, closed=3 (범위 중간값)
    pattern1 = re.search(
        r'(?:왕복|편도)?\s*(\d+)(?:~(\d+))?[개]?차로\s*중\s*(?:편도|중앙|양쪽)?\s*(\d+)(?:~(\d+))?[개]?차로',
        s
    )
    if pattern1:
        t1, t2 = int(pattern1.group(1)), pattern1.group(2)
        c1, c2 = int(pattern1.group(3)), pattern1.group(4)
        total = (int(t1) + int(t2)) // 2 if t2 else int(t1)
        closed = (int(c1) + int(c2)) // 2 if c2 else int(c1)
        return total, closed

    # 패턴 2: "N~M차로 단계별통제" → 최대 차로 기준, 절반 폐쇄 가정
    pattern2 = re.search(r'(\d+)[~-](\d+)차로\s*단계별통제', s)
    if pattern2:
        total = int(pattern2.group(2))
        closed = total // 2
        return total, closed

    # 패턴 3: "편도 1·2차선 부분통제" → total=2, closed=1
    pattern3 = re.search(r'편도\s*(\d+)[·,]\s*(\d+)차[로선]', s)
    if pattern3:
        total = int(pattern3.group(2))
        closed = int(pattern3.group(1))
        return total, closed

    # 패턴 4: 단독 숫자+차로 (예: "법동방향 2개차로")
    pattern4 = re.findall(r'(\d+)[개]?차[로선]', s)
    if len(pattern4) >= 2:
        nums = sorted([int(x) for x in pattern4])
        return nums[-1], nums[0]
    elif len(pattern4) == 1:
        # 차로 수만 있으면 폐쇄 1개 가정
        return None, int(pattern4[0])

    return None, None


def _lane_ratio_to_penalty(total: int, closed: int) -> float:
    """
    잔여 차로 비율을 속도 패널티 계수로 변환한다.

    HCM 기반 단순화 모델:
      잔여비율 > 0.75  →  0.75  (소폭 감속)
      0.50 ~ 0.75     →  0.55  (중간 감속)
      0.35 ~ 0.50     →  0.40  (심각 감속, 대전시 주의 임계 20km/h 수준)
      < 0.35          →  0.25  (거의 차단, 대전시 심각 임계 15km/h 수준)
    """
    if total is None or total == 0:
        return 0.50  # 파싱 실패 시 보수적 기본값

    remain_ratio = (total - closed) / total
    remain_ratio = max(0.0, min(1.0, remain_ratio))  # 0~1 클리핑

    if remain_ratio > 0.75:
        return 0.75
    elif remain_ratio > 0.50:
        return 0.55
    elif remain_ratio > 0.35:
        return 0.40
    else:
        return 0.25


def _split_compound_road_name(road_name: str) -> list:
    """
    복합 도로명을 개별 도로명 리스트로 분리한다.
    표준노드링크에 미등재된 도로명은 대체 도로명으로 변환.
    """
    # 표준노드링크 미등재 도로 → 대체 도로명
    ROAD_ALIAS = {
        '동광장로': '중앙로',  # 13공구: 실제 위치는 중앙로 구간 (대전역~죽전)
    }

    if not road_name or str(road_name).strip() == 'nan':
        return []
    s = str(road_name).strip()
    s = re.sub(r'\(.*?\)', '', s).strip()
    parts = re.split(r'[·/,]', s)
    result = []
    for p in parts:
        p = p.strip()
        if p:
            # 대체 도로명 적용
            p = ROAD_ALIAS.get(p, p)
            result.append(p)
    return result


# ─────────────────────────────────────────────────────────────
# 메인 함수
# ─────────────────────────────────────────────────────────────

def apply_construction_rules(gdf_link, df_construction, use_gis=True):
    """
    트램_공구별_통제현황.xlsx 데이터를 파싱하여 링크별 속도 패널티를 적용한다.

    v3 개선: GIS 좌표 기반 세밀 패널티 (use_gis=True)
    - geocode_construction.py에서 추출한 시점/종점 좌표를 활용
    - 시점~종점 사이 도로에 위치한 링크만 패널티 적용 (도로명 전체 X)
    - 시점/종점 좌표에서 버퍼 거리 내 링크를 필터

    폴백 (use_gis=False 또는 좌표 없을 때):
    - 기존 v2: 도로명 전체에 패널티 적용
    """
    import geopandas as gpd
    from shapely.geometry import Point, LineString
    from shapely.ops import nearest_points

    print("[Simulation] Applying construction speed penalties (v3 - GIS coordinate based)...")

    # 링크 패널티 초기화
    link_multipliers = {int(lid): 1.0 for lid in gdf_link['LINK_ID']}

    # 활성 구간 + GIS 좌표 로드
    active = df_construction[df_construction['상태(7/8기준)'] == '활성'].copy()
    print(f"  활성 공사 구간 수: {len(active)}개")

    # GIS 좌표 로드
    import os
    zones_path = os.path.join("outputs", "construction_zones_geocoded.csv")
    df_zones = None
    if use_gis and os.path.exists(zones_path):
        df_zones = pd.read_csv(zones_path)
        print(f"  GIS 좌표 파일 로드: {len(df_zones)}개 구간")
    else:
        print("  GIS 좌표 미사용 — 도로명 전체 매칭 (v2 폴백)")

    # 링크 투영 (거리 계산용)
    gdf_link_proj = gdf_link.to_crs(epsg=5186) if gdf_link.crs else gdf_link

    applied_count = 0
    skipped_roads = []
    gis_applied = 0

    for idx, row in active.iterrows():
        road_raw = str(row['노선명']).strip()
        method_str = str(row.get('통제방법(차로)', '')).strip()
        gonggu = str(row.get('공구', '')).strip()

        # 패널티 계산
        total, closed = _parse_lane_info(method_str)
        is_parsed = (total is not None and closed is not None)
        penalty = _lane_ratio_to_penalty(total, closed) if is_parsed else _fallback_penalty(method_str)

        # GIS 기반 적용 시도
        applied_this = False
        if df_zones is not None and idx < len(df_zones):
            zone_row = df_zones.iloc[idx]
            start_lat = zone_row.get('start_lat')
            start_lon = zone_row.get('start_lon')
            end_lat   = zone_row.get('end_lat')
            end_lon   = zone_row.get('end_lon')

            if pd.notna(start_lat) and pd.notna(start_lon):
                # 시점/종점에서 버퍼(500m) 내 + 해당 도로명 링크만 필터
                road_parts = _split_compound_road_name(road_raw)
                road_mask = pd.Series([False] * len(gdf_link), index=gdf_link.index)
                for rp in road_parts:
                    road_mask |= gdf_link['ROAD_NAME'].str.contains(rp, na=False, regex=False)

                if road_mask.any():
                    # 시점 좌표를 투영
                    start_pt = gpd.GeoDataFrame(
                        geometry=[Point(start_lon, start_lat)], crs="EPSG:4326"
                    ).to_crs(epsg=5186).geometry.iloc[0]

                    if pd.notna(end_lat) and pd.notna(end_lon) and (end_lat != start_lat or end_lon != start_lon):
                        end_pt = gpd.GeoDataFrame(
                            geometry=[Point(end_lon, end_lat)], crs="EPSG:4326"
                        ).to_crs(epsg=5186).geometry.iloc[0]
                        # 시점~종점 사이 거리 + 500m 버퍼
                        corridor_len = start_pt.distance(end_pt)
                        buffer_dist = max(corridor_len * 0.6, 500)  # 구간 길이의 60% 또는 최소 500m
                        # 시점/종점 모두에서 buffer 내 링크
                        start_buf = start_pt.buffer(buffer_dist)
                        end_buf = end_pt.buffer(buffer_dist)
                        combined_buf = start_buf.union(end_buf)
                    else:
                        # 시점만 있으면 반경 800m 버퍼
                        combined_buf = start_pt.buffer(800)

                    # 도로명 매칭 + 공간 버퍼 내 링크
                    spatial_mask = gdf_link_proj.geometry.centroid.within(combined_buf)
                    final_mask = road_mask & spatial_mask

                    matched_links = gdf_link[final_mask]
                    if len(matched_links) > 0:
                        for lid in matched_links['LINK_ID'].astype(int):
                            link_multipliers[lid] = min(link_multipliers[lid], penalty)
                        applied_count += len(matched_links)
                        gis_applied += 1
                        applied_this = True

                        if is_parsed and total and closed:
                            ratio_str = f"{total}→폐쇄{closed}→잔여{((total-closed)/total*100):.0f}%"
                        else:
                            ratio_str = "파싱불가(fallback)"
                        print(f"  [{gonggu}] {road_raw} | GIS구간 | {ratio_str} → "
                              f"패널티 {penalty:.2f} | {len(matched_links)}개 링크 (버퍼 내)")

        # GIS 실패 시 도로명 + 시점 좌표 반경으로 한정 (v3.1 개선)
        if not applied_this:
            road_parts = _split_compound_road_name(road_raw)
            for rp in road_parts:
                matched = gdf_link[gdf_link['ROAD_NAME'].str.contains(rp, na=False, regex=False)]
                if len(matched) > 0:
                    # 시점 좌표가 있으면 반경 1.5km 내로 한정
                    if df_zones is not None and idx < len(df_zones):
                        zone_row = df_zones.iloc[idx]
                        s_lat = zone_row.get('start_lat')
                        s_lon = zone_row.get('start_lon')
                        if pd.notna(s_lat) and pd.notna(s_lon):
                            s_pt = gpd.GeoDataFrame(
                                geometry=[Point(s_lon, s_lat)], crs="EPSG:4326"
                            ).to_crs(epsg=5186).geometry.iloc[0]
                            spatial_mask = gdf_link_proj.geometry.centroid.within(s_pt.buffer(1500))
                            road_mask = gdf_link['ROAD_NAME'].str.contains(rp, na=False, regex=False)
                            final_mask = road_mask & spatial_mask
                            matched = gdf_link[final_mask]

                    if len(matched) > 0:
                        for lid in matched['LINK_ID'].astype(int):
                            link_multipliers[lid] = min(link_multipliers[lid], penalty)
                        applied_count += len(matched)
                        if is_parsed and total and closed:
                            ratio_str = f"{total}→폐쇄{closed}→잔여{((total-closed)/total*100):.0f}%"
                        else:
                            ratio_str = "파싱불가(fallback)"
                        print(f"  [{gonggu}] {rp} | 도로명+반경 | {ratio_str} → "
                              f"패널티 {penalty:.2f} | {len(matched)}개 링크")
                    else:
                        # 반경 내 매칭 실패 → 도로명 전체로 폴백
                        matched_full = gdf_link[gdf_link['ROAD_NAME'].str.contains(rp, na=False, regex=False)]
                        if len(matched_full) > 0:
                            for lid in matched_full['LINK_ID'].astype(int):
                                link_multipliers[lid] = min(link_multipliers[lid], penalty)
                            applied_count += len(matched_full)
                            if is_parsed and total and closed:
                                ratio_str = f"{total}→폐쇄{closed}→잔여{((total-closed)/total*100):.0f}%"
                            else:
                                ratio_str = "파싱불가(fallback)"
                            print(f"  [{gonggu}] {rp} | 도로명전체(폴백) | {ratio_str} → "
                                  f"패널티 {penalty:.2f} | {len(matched_full)}개 링크")
                        else:
                            skipped_roads.append(f"{gonggu}/{rp}")
                else:
                    skipped_roads.append(f"{gonggu}/{rp}")

    print(f"\n  총 패널티 적용 링크: {applied_count}개")
    print(f"  GIS 좌표 기반 적용: {gis_applied}개 구간 / 도로명 폴백: {len(active) - gis_applied}개 구간")
    if skipped_roads:
        print(f"  매칭 실패: {skipped_roads}")

    # 패널티 분포 요약
    p_series = pd.Series(list(link_multipliers.values()))
    print(f"\n  패널티 분포:")
    print(f"    1.00 (무패널티): {(p_series == 1.0).sum()}개")
    print(f"    0.75 (잔여>75%): {(p_series == 0.75).sum()}개")
    print(f"    0.55 (잔여50~75%): {(p_series == 0.55).sum()}개")
    print(f"    0.40 (잔여35~50%): {(p_series == 0.40).sum()}개")
    print(f"    0.25 (잔여<35%):  {(p_series == 0.25).sum()}개")
    print(f"    0.50 (파싱기본):   {(p_series == 0.50).sum()}개")

    gdf_link = gdf_link.copy()
    gdf_link['speed_penalty_multiplier'] = gdf_link['LINK_ID'].astype(int).map(link_multipliers)

    # 잔여 차로 비율 피처
    remain_ratios = {}
    for idx2, row in active.iterrows():
        method_str = str(row.get('통제방법(차로)', ''))
        total, closed = _parse_lane_info(method_str)
        if total and closed and total > 0:
            ratio = (total - closed) / total
            road_parts = _split_compound_road_name(str(row['노선명']))
            for rp in road_parts:
                matched = gdf_link[gdf_link['ROAD_NAME'].str.contains(rp, na=False, regex=False)]
                for lid in matched['LINK_ID'].astype(int):
                    if lid not in remain_ratios or ratio < remain_ratios[lid]:
                        remain_ratios[lid] = ratio

    gdf_link['lane_remain_ratio'] = gdf_link['LINK_ID'].astype(int).map(remain_ratios).fillna(1.0)

    return gdf_link


def _fallback_penalty(method_str: str) -> float:
    """
    차로 숫자 파싱이 불가능한 경우 키워드 기반 폴백 패널티.
    (구버전 로직 유지, 파싱 실패 안전망)
    """
    s = str(method_str).lower()
    if '전체통제' in s or '전일차단' in s or '전일통제' in s:
        return 0.25
    elif '부분통제' in s:
        return 0.50
    return 0.55
