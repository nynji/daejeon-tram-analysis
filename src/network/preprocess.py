"""
preprocess.py
노드/링크 GeoJSON 로딩, 정거장 구간 매핑, 공사 데이터 정제

개선 사항 (v3):
  - segment_id: 단일 정거장 기반(STN{no}) → 인접 정거장 쌍 기반(SEG_{a}_{b})
    · 정거장 노선 순서(station_no 오름차순)를 기준으로 각 링크 중점이
      어느 두 정거장 사이 구간에 속하는지를 거리 비교로 결정
    · Prophet "정거장 간 구간" 단위와 완전히 일치
"""

import os
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point


def load_data(data_dir: str = "data"):
    """
    링크, 노드, 정거장 좌표, 공사 현황 데이터를 로드한다.
    """
    print("[Preprocessing] Loading data files ...")

    link_path  = os.path.join(data_dir, "network", "daejeon_link.geojson")
    node_path  = os.path.join(data_dir, "network", "daejeon_node.geojson")
    stn_path   = os.path.join(data_dir, "network", "역_좌표.csv")
    const_path = os.path.join(data_dir, "트램_공구별_통제현황.xlsx")

    gdf_link = gpd.read_file(link_path)
    gdf_link['LINK_ID'] = gdf_link['LINK_ID'].astype('int64')

    gdf_node = gpd.read_file(node_path)

    df_stations = pd.read_csv(stn_path)
    df_stations.columns = [c.strip() for c in df_stations.columns]
    df_stations = df_stations.sort_values('station_no').reset_index(drop=True)

    df_construction = pd.read_excel(const_path, header=3)
    df_construction = (
        df_construction
        .dropna(subset=['공구'])
        .iloc[:25]
        .reset_index(drop=True)
    )
    df_construction.columns = [c.strip() for c in df_construction.columns]

    print(f"  링크: {len(gdf_link):,}개  |  노드: {len(gdf_node):,}개  "
          f"|  정거장: {len(df_stations)}개  |  공사구간: {len(df_construction)}개")
    return gdf_link, gdf_node, df_stations, df_construction


def create_station_segments(df_stations: pd.DataFrame, gdf_link: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    팀 공용 segment_link_mapping.csv 기준으로 각 링크에 segment_id를 매핑한다.

    매핑 파일: data/segment_link_mapping.csv
      - segment_id: SEG_01_201_202 형태
      - link_id: 해당 구간에 포함된 링크 ID
      - direction: AB(정방향) / BA(역방향)

    매핑되지 않는 링크(트램 노선 밖 도로)는 segment_id='NON_TRAM'으로 표기.
    """
    import os

    seg_map_path = os.path.join("data", "segment_link_mapping.csv")
    print(f"[Preprocessing] Loading segment mapping: {seg_map_path} ...")

    df_seg = pd.read_csv(seg_map_path)
    df_seg['link_id'] = df_seg['link_id'].astype('int64')

    # 링크별로 하나의 segment_id 할당 (AB 방향 우선, 중복 시 첫 번째)
    seg_lookup = (
        df_seg.drop_duplicates(subset='link_id', keep='first')
        [['link_id', 'segment_id', 'from_station_no', 'from_station_name',
          'to_station_no', 'to_station_name', 'direction']]
        .set_index('link_id')
    )

    gdf_link = gdf_link.copy()
    gdf_link['LINK_ID_int'] = gdf_link['LINK_ID'].astype('int64')

    # segment_id 매핑
    gdf_link['segment_id'] = gdf_link['LINK_ID_int'].map(seg_lookup['segment_id']).fillna('NON_TRAM')
    gdf_link['nearest_station_no'] = gdf_link['LINK_ID_int'].map(seg_lookup['from_station_no'])
    gdf_link['nearest_station_name'] = gdf_link['LINK_ID_int'].map(seg_lookup['from_station_name']).fillna('비노선')
    gdf_link['second_station_no'] = gdf_link['LINK_ID_int'].map(seg_lookup['to_station_no'])
    gdf_link['second_station_name'] = gdf_link['LINK_ID_int'].map(seg_lookup['to_station_name']).fillna('비노선')

    # 비노선 링크는 nearest_station을 공간 최근접으로 폴백
    non_tram_mask = gdf_link['segment_id'] == 'NON_TRAM'
    if non_tram_mask.any() and len(df_stations) > 0:
        import numpy as np
        gdf_proj = gdf_link[non_tram_mask].to_crs(epsg=5186)
        geometry_stns = [Point(row['lon'], row['lat']) for _, row in df_stations.iterrows()]
        gdf_stns = gpd.GeoDataFrame(df_stations, geometry=geometry_stns, crs="EPSG:4326").to_crs(epsg=5186)
        stn_coords = np.array([[g.x, g.y] for g in gdf_stns.geometry])

        nearest_nos = []
        nearest_names = []
        for midpoint in gdf_proj.geometry.centroid:
            dists = np.sqrt((stn_coords[:, 0] - midpoint.x)**2 + (stn_coords[:, 1] - midpoint.y)**2)
            idx = np.argmin(dists)
            nearest_nos.append(int(df_stations.iloc[idx]['station_no']))
            nearest_names.append(str(df_stations.iloc[idx]['station_name']))

        gdf_link.loc[non_tram_mask, 'nearest_station_no'] = nearest_nos
        gdf_link.loc[non_tram_mask, 'nearest_station_name'] = nearest_names

    gdf_link.drop(columns=['LINK_ID_int'], inplace=True)

    tram_count = (gdf_link['segment_id'] != 'NON_TRAM').sum()
    seg_count = gdf_link[gdf_link['segment_id'] != 'NON_TRAM']['segment_id'].nunique()
    print(f"  매핑 완료 — 트램 노선 링크: {tram_count}개 / 구간: {seg_count}개 / 비노선: {non_tram_mask.sum()}개")
    return gdf_link
