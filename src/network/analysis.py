"""
analysis.py
NetworkX 그래프 구성 및 매개중심성(Edge Betweenness Centrality) 계산

설계 원칙:
  - 엣지 가중치 = 통행시간(초) = 링크길이(m) / 실효속도(m/s)
  - 공사 중 실효속도 = MAX_SPD × speed_penalty_multiplier
  - 최소 속도 5km/h 보장 (0 나눗셈 방지)
  - BC 계산: k-샘플링으로 대규모 그래프 연산 현실화
    · sample_frac=0.05 (기본) → 최소 50개 노드, 최대 500개 상한
    · 결과 안정성 vs 연산 시간 균형점

매개중심성 해석:
  C_B(e) = Σ σ_st(e) / σ_st
  - 높을수록 해당 링크가 차단될 때 우회 수요가 집중됨
  - 공사 후 bc_change = bc_under_construction - bc_free_flow
    → 양수(+): 공사로 인해 우회 경로 집중도 증가 → 정체 전이 위험 상승
    → 음수(-): 공사 구간 자체가 최단경로에서 탈락 → 우회 분산
"""

import networkx as nx
import pandas as pd
import numpy as np
import random


def build_network_graph(gdf_link, speed_mode: str = "free_flow",
                        its_speed_dict: dict = None) -> nx.DiGraph:
    """
    표준노드링크 GeoDataFrame으로 방향성 가중 그래프를 생성한다.

    Parameters
    ----------
    gdf_link : GeoDataFrame
    speed_mode : str
        'free_flow'          → MAX_SPD (제한속도) 사용
        'under_construction' → MAX_SPD × 패널티 배율 사용
        'its_realtime'       → ITS 실측 속도 사용 (its_speed_dict 필수)
    its_speed_dict : dict
        {LINK_ID: speed_km/h}. speed_mode='its_realtime' 시 사용.
        실측 데이터 없는 링크는 MAX_SPD×패널티로 폴백.
    """
    mode_label = speed_mode
    if speed_mode == 'its_realtime':
        mode_label = f"its_realtime ({len(its_speed_dict or {}):,} links)"
    print(f"[Analysis] Building NetworkX DiGraph — mode: '{mode_label}' ...")
    G = nx.DiGraph()

    skip_count = 0
    its_used = 0
    for _, row in gdf_link.iterrows():
        u = int(row['F_NODE'])
        v = int(row['T_NODE'])
        link_id = int(row['LINK_ID'])
        length = float(row['LENGTH'])
        max_spd = float(row['MAX_SPD'])

        # 실효 속도 결정
        if speed_mode == "its_realtime" and its_speed_dict:
            its_spd = its_speed_dict.get(link_id)
            if its_spd and its_spd > 0:
                speed_kmh = float(its_spd)
                its_used += 1
            else:
                # ITS 데이터 없는 링크 → 패널티 적용 속도 폴백
                multiplier = float(row.get('speed_penalty_multiplier', 1.0))
                speed_kmh = max_spd * multiplier
        elif speed_mode == "under_construction" and 'speed_penalty_multiplier' in row.index:
            multiplier = float(row['speed_penalty_multiplier'])
            speed_kmh = max_spd * multiplier
        else:
            speed_kmh = max_spd

        speed_kmh = max(speed_kmh, 5.0)

        speed_ms = speed_kmh * (1000.0 / 3600.0)
        travel_time = length / speed_ms

        if u == v:
            skip_count += 1
            continue

        G.add_edge(u, v,
                   link_id=link_id,
                   weight=travel_time,
                   length=length,
                   speed=speed_kmh)

    node_count = G.number_of_nodes()
    edge_count = G.number_of_edges()
    print(f"  노드 수: {node_count:,}  |  엣지 수: {edge_count:,}  |  제외(자기루프): {skip_count}")
    if speed_mode == 'its_realtime':
        print(f"  ITS 실측 속도 적용: {its_used:,}개 링크 / 폴백: {edge_count - its_used:,}개")
    return G


def calculate_betweenness_centrality(
    G: nx.DiGraph,
    sample_frac: float = 0.05,
    max_sample: int = 1000,
    seed: int = 42,
    ensemble_seeds: list = None
) -> dict:
    """
    엣지 매개중심성을 샘플링 기반으로 계산한다.

    Parameters
    ----------
    G : nx.DiGraph
    sample_frac : float
        전체 노드 중 샘플 비율. 기본 0.05 (5%)
    max_sample : int
        샘플 상한. 연산 시간 제어용
    seed : int
        기본 시드
    ensemble_seeds : list of int or None
        복수 시드로 앙상블 평균 계산. None이면 단일 시드.
        예: [42, 0, 123, 777] → 4번 계산 후 평균 → 안정성 향상

    Returns
    -------
    link_bc : dict
        {link_id: betweenness_centrality_score}
    """
    nodes = list(G.nodes())
    n = len(nodes)
    k = min(max(int(n * sample_frac), 50), max_sample)

    seeds_to_run = ensemble_seeds if ensemble_seeds else [seed]
    print(f"[Analysis] Calculating Edge BC — {n:,}노드 중 {k}개 샘플 ({k/n*100:.1f}%), "
          f"{'앙상블 ' + str(len(seeds_to_run)) + '회' if len(seeds_to_run) > 1 else 'seed=' + str(seeds_to_run[0])} ...")

    # 앙상블: 복수 시드로 계산 후 평균
    all_results = []
    for s in seeds_to_run:
        random.seed(s)
        np.random.seed(s)
        edge_bc = nx.edge_betweenness_centrality(
            G, k=k, normalized=True, weight='weight', seed=s
        )
        # (u, v) → link_id 매핑
        link_bc = {}
        for (u, v), score in edge_bc.items():
            if G.has_edge(u, v):
                link_id = G[u][v].get('link_id')
                if link_id is not None:
                    link_bc[link_id] = score
        all_results.append(link_bc)

    # 앙상블 평균
    if len(all_results) == 1:
        final = all_results[0]
    else:
        all_ids = set()
        for r in all_results:
            all_ids.update(r.keys())
        final = {}
        for lid in all_ids:
            vals = [r.get(lid, 0.0) for r in all_results]
            final[lid] = float(np.mean(vals))

    nonzero = sum(1 for s in final.values() if s > 0)
    print(f"  BC 완료 — {len(final):,}개 링크, 비영(BC>0): {nonzero:,}개")
    return final


def compute_node_betweenness(G: nx.DiGraph, sample_frac: float = 0.05, seed: int = 42) -> dict:
    """
    노드 매개중심성 보조 계산. (교차로 단위 파급력 분석용)
    run.py에서 선택적으로 호출.
    """
    nodes = list(G.nodes())
    k = min(max(int(len(nodes) * sample_frac), 50), 500)
    print(f"[Analysis] Computing Node BC (k={k}) ...")
    node_bc = nx.betweenness_centrality(G, k=k, normalized=True, weight='weight', seed=seed)
    return node_bc
