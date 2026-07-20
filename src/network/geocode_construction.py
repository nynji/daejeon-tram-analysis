"""
geocode_construction.py
공사 통제 구간 자연어 주소 → GIS 좌표 변환

방법: 표준노드링크의 NODE_NAME에서 교차로명(네거리/삼거리/오거리)을 검색하여
시점·종점 노드를 특정하고, 두 노드 사이의 링크를 공사 영향 구간으로 판정.

이 방식의 장점:
  - 외부 API(네이버/카카오 지오코딩) 불필요
  - 표준노드링크와 완전 일치 (링크 ID로 직접 연결 가능)
  - 공사 구간이 교차로 단위로 기술되어 있어 정확도 높음
"""

import re
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point


def extract_intersection_names(location_str: str) -> list:
    """
    통제구간 문자열에서 교차로/지점명 키워드를 추출한다.

    예시:
      "농수산오거리 ~ 동대전농협 대덕지점" → ["농수산오거리", "동대전농협"]
      "선사유적네거리~정부청사역네거리~한밭대로네거리" → ["선사유적네거리", "정부청사역네거리", "한밭대로네거리"]
      "정림삼거리 ~ 도마삼거리(양방향)" → ["정림삼거리", "도마삼거리"]
    """
    if pd.isna(location_str) or not str(location_str).strip():
        return []

    s = str(location_str).strip()
    # 괄호 내용 제거
    s = re.sub(r'\(.*?\)', '', s).strip()
    # 방향 표시 제거
    s = re.sub(r'방향|방면', '', s)
    # 구분자로 분리
    parts = re.split(r'[~→,/]', s)
    keywords = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # "읍내삼거리 일원" → "읍내삼거리"로 정리
        p = re.sub(r'\s*(일원|부근|인근|주변|시점부)$', '', p).strip()
        # "L=180m" 같은 수치 정보 제거
        p = re.sub(r'L=\d+m', '', p).strip()
        if not p:
            continue
        # 교차로명 패턴: XXX네거리, XXX삼거리, XXX오거리, XXX교, XXX역
        if any(kw in p for kw in ['네거리', '삼거리', '오거리', '사거리', '교차로']):
            keywords.append(p)
        elif any(kw in p for kw in ['지하차도', '육교', '교']):
            keywords.append(p)
        elif any(kw in p for kw in ['역', '시장']):
            keywords.append(p)
        elif len(p) >= 2:
            keywords.append(p)
    return keywords


def match_node_by_name(keyword: str, gdf_node: gpd.GeoDataFrame,
                        threshold: float = 0.8) -> list:
    """
    키워드로 NODE_NAME을 검색하여 매칭 노드를 반환한다.
    단계적으로 검색 범위를 넓혀간다.
    """
    # 키워드 정규화: 공백 제거
    kw = keyword.replace(' ', '').strip()
    node_names_clean = gdf_node['NODE_NAME'].str.replace(' ', '')

    # 1차: 완전 포함 검색
    matches = gdf_node[node_names_clean.str.contains(kw, na=False, regex=False)]
    if len(matches) > 0:
        return matches

    # 2차: 교차로 접미사 제거 후 검색
    short_kw = re.sub(r'(네거리|삼거리|오거리|사거리|교차로|지하차도|육교|보도육교|시점부)', '', kw)
    if len(short_kw) >= 2:
        matches = gdf_node[node_names_clean.str.contains(short_kw, na=False, regex=False)]
        if 0 < len(matches) <= 10:
            return matches

    # 3차: 앞 3글자만으로 검색 (최소 2자)
    for length in range(min(len(kw), 5), 1, -1):
        sub = kw[:length]
        matches = gdf_node[node_names_clean.str.contains(sub, na=False, regex=False)]
        if 0 < len(matches) <= 5:
            return matches

    return gdf_node.iloc[0:0]  # 빈 GeoDataFrame


def geocode_construction_zones(df_construction: pd.DataFrame,
                                 gdf_node: gpd.GeoDataFrame,
                                 gdf_link: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    공사 통제 구간을 표준노드링크 기반으로 GIS 좌표에 매핑한다.

    매칭 전략:
    1. 교차로명 → NODE_NAME 부분문자열 검색
    2. 실패 시 → 수동 보정 테이블 (건물명 → 근접 교차로명)
    3. 여전히 실패 시 → 해당 도로 링크 중점의 평균 좌표를 폴백
    """
    # 수동 보정 테이블: NODE_NAME 검색 불가한 지점 → 대체 검색어
    MANUAL_FALLBACK = {
        '동부여성가족원':  '읍내삼거리',       # 1공구 계족로 2구간
        '읍내동 보도육교': '읍내삼거리',       # 동일 위치
        '알뜰주유소':      '연축동',           # 1공구 신탄진로 (연축동 방향)
        '성우보육원':      '연축동',           # 동일 위치
        '트리풀시티5단지네거리': '상대지하차도', # 7공구 도안대로 1단계
        '상대지하차도 시점부': '상대지하차도',  # 동일
        '대전역 죽전네거리': '죽전교',         # 13공구 동광장로
        '죽전네거리':      '죽전교',           # 동일
        '베스티안병원네거리': '대전역네거리',   # 13공구 동광장로 종점
    }

    active = df_construction[df_construction['상태(7/8기준)'] == '활성'].copy()
    print(f"[Geocode] 활성 공사 구간 {len(active)}개 → 노드 매칭 시작 ...")

    results = []
    for _, row in active.iterrows():
        loc = row.get('통제구간(도로상 위치)', '')
        keywords = extract_intersection_names(loc)
        gonggu = row['공구']
        road = row['노선명']

        matched_nodes = []
        for kw in keywords:
            # 수동 보정 테이블 적용
            search_kw = MANUAL_FALLBACK.get(kw.replace(' ',''), kw)
            m = match_node_by_name(search_kw, gdf_node)
            if len(m) > 0:
                matched_nodes.append({
                    'keyword': kw,
                    'search_used': search_kw,
                    'node_id': int(m.iloc[0]['NODE_ID']),
                    'node_name': m.iloc[0]['NODE_NAME'],
                    'lat': m.iloc[0].geometry.y,
                    'lon': m.iloc[0].geometry.x,
                    'match_count': len(m),
                })

        # 폴백: 키워드 전부 실패 시 도로명 기반 링크 중점
        if not matched_nodes:
            road_parts = [r.strip() for r in re.sub(r'\(.*?\)', '', str(road)).split('·') if r.strip()]
            for rp in road_parts:
                road_links = gdf_link[gdf_link['ROAD_NAME'].str.contains(rp, na=False, regex=False)]
                if len(road_links) > 0:
                    centroid = road_links.geometry.centroid
                    avg_y = centroid.y.mean()
                    avg_x = centroid.x.mean()
                    matched_nodes.append({
                        'keyword': f'[도로명 폴백: {rp}]',
                        'search_used': rp,
                        'node_id': None,
                        'node_name': f'{rp} 중점',
                        'lat': avg_y,
                        'lon': avg_x,
                        'match_count': len(road_links),
                    })
                    break

        # 시점/종점 결정
        start = matched_nodes[0] if len(matched_nodes) >= 1 else None
        end   = matched_nodes[-1] if len(matched_nodes) >= 2 else start

        # 두 노드 사이 링크 찾기
        affected_links = []
        if start and end and start.get('node_id') and end.get('node_id'):
            s_id, e_id = start['node_id'], end['node_id']
            if s_id != e_id:
                mask = (
                    (gdf_link['F_NODE'].astype(int) == s_id) |
                    (gdf_link['T_NODE'].astype(int) == s_id) |
                    (gdf_link['F_NODE'].astype(int) == e_id) |
                    (gdf_link['T_NODE'].astype(int) == e_id)
                )
                affected_links = gdf_link[mask]['LINK_ID'].astype(int).tolist()

        results.append({
            '공구': gonggu,
            '노선명': road,
            '통제구간_원문': loc,
            '키워드_추출': keywords,
            '매칭_노드수': len(matched_nodes),
            'start_keyword': start['keyword'] if start else None,
            'start_node_id': start['node_id'] if start else None,
            'start_lat': start['lat'] if start else None,
            'start_lon': start['lon'] if start else None,
            'end_keyword': end['keyword'] if end else None,
            'end_node_id': end['node_id'] if end else None,
            'end_lat': end['lat'] if end else None,
            'end_lon': end['lon'] if end else None,
            'affected_link_count': len(affected_links),
            'affected_link_ids': affected_links[:50],
        })

    df_zones = pd.DataFrame(results)
    matched_count = (df_zones['매칭_노드수'] > 0).sum()
    print(f"  노드 매칭 성공: {matched_count}/{len(df_zones)}개 구간")
    if matched_count < len(df_zones):
        failed = df_zones[df_zones['매칭_노드수'] == 0]
        for _, r in failed.iterrows():
            print(f"    실패: [{r['공구']}] {r['노선명']} — '{r['통제구간_원문']}'")
    return df_zones


if __name__ == "__main__":
    gdf_link = gpd.read_file('data/network/daejeon_link.geojson')
    gdf_link['LINK_ID'] = gdf_link['LINK_ID'].astype(int)
    gdf_node = gpd.read_file('data/network/daejeon_node.geojson')
    df_construction = pd.read_excel('data/트램_공구별_통제현황.xlsx', header=3)
    df_construction = df_construction.dropna(subset=['공구']).iloc[:25]

    df_zones = geocode_construction_zones(df_construction, gdf_node, gdf_link)
    print()
    cols = ['공구','노선명','start_keyword','start_lat','start_lon','end_keyword','end_lat','end_lon','매칭_노드수','affected_link_count']
    print(df_zones[cols].to_string())

    # CSV 저장
    df_zones.to_csv('outputs/construction_zones_geocoded.csv', index=False, encoding='utf-8-sig')
    print('\n→ outputs/construction_zones_geocoded.csv 저장 완료')
