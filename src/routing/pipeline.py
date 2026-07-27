"""
정류장/좌표 O-D 최단경로 라우팅 코어 — 대시보드 통합용.

이 파일은 routing/scripts/pipeline.py에서 "22개 피처 조립 + XGBoost
호출 + 리플레이 날짜 검증" 부분을 뺀 순수 라우팅 코어만 담는다. 그
부분은 XGBoost/대시보드 담당자가 이미 자기 파이프라인으로 갖고 있으므로
중복 구현하지 않는다 — 호출자가 특정 시각의 90개 구간 위험도
(segment_risk_table)를 직접 계산해서 넘겨주면, 이 모듈은 그래프에
반영해서 정적경로/예측반영경로를 비교해준다.

사용법은 routing/PIPELINE_GUIDE.md 참고.
"""

import pickle
from functools import lru_cache

import numpy as np
import pandas as pd
import networkx as nx

# ---------------------------------------------------------------------------
# 경로 상수
# ---------------------------------------------------------------------------
BASE_DIR = "."  # final_project/ 를 작업 디렉토리로 실행한다고 가정

STATION_LINKS_CSV = f"{BASE_DIR}/prophet/dataset/network/정류장_구간_링크.csv"
STATION_COORDS_CSV = f"{BASE_DIR}/prophet/dataset/coords/역_좌표.csv"

GRAPH_PICKLE = f"{BASE_DIR}/routing/dataset/graph_daejeon.gpickle"
NODE_COORDS_CSV = f"{BASE_DIR}/routing/dataset/node_coords.csv"

# ---------------------------------------------------------------------------
# 도메인 상수
# ---------------------------------------------------------------------------
# 90개 중 Prophet y_hat_t30 신뢰 불가(2026-07-18 성능 재검토에서 확정) 18개.
# 이 목록에 있는 segment_key는 y_hat_t30을 넘겨받아도 base 계산에 안 쓴다.
LOW_CONF_SEGMENTS = {
    "SEG_11_211_212_BA", "SEG_14_214_215_AB", "SEG_14_214_215_BA",
    "SEG_16_216_217_AB", "SEG_16_216_217_BA", "SEG_21_221_222_AB", "SEG_21_221_222_BA",
    "SEG_28_228_229_BA", "SEG_29_229_230_AB", "SEG_29_229_230_BA",
    "SEG_31_231_232_AB", "SEG_31_231_232_BA", "SEG_35_235_236_BA",
    "SEG_37_237_238_BA", "SEG_40_240_201_BA", "SEG_42_241_242_AB",
    "SEG_42_241_242_BA", "SEG_45_233_245_AB",
}

PENALTY_SCALE = 2  # 그리드서치 검증 결과 권장값은 4 (routing/PIPELINE_GUIDE.md 참고)

_MIN_PER_UNIT = 0.06  # base(=LENGTH[m]/speed[km/h]) -> 분 환산 (÷1000 * 60)


# ---------------------------------------------------------------------------
# 캐시된 정적 자원 로드 (프로세스당 1회)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _load_base_graph() -> nx.DiGraph:
    with open(GRAPH_PICKLE, "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def _load_node_coords() -> pd.DataFrame:
    """그래프 노드(18,215개)의 위경도(EPSG:4326) 좌표 lookup."""
    return pd.read_csv(NODE_COORDS_CSV, dtype={"NODE_ID": str})


@lru_cache(maxsize=1)
def _load_station_coords() -> pd.DataFrame:
    return pd.read_csv(STATION_COORDS_CSV)


@lru_cache(maxsize=1)
def _load_station_links() -> pd.DataFrame:
    return pd.read_csv(STATION_LINKS_CSV, dtype=str)


# ---------------------------------------------------------------------------
# O-D 정류장/좌표 -> 그래프 노드 스냅
# ---------------------------------------------------------------------------
def _lookup_station_row(station: str):
    """station: 정류장 이름(예: '정부청사') 또는 station_no(예: '218'/218) 모두 허용."""
    coords = _load_station_coords()

    row = None
    station_no_str = str(station).strip()
    if station_no_str.isdigit():
        hit = coords[coords["station_no"] == int(station_no_str)]
        if len(hit):
            row = hit.iloc[0]
    if row is None:
        hit = coords[coords["station_name"] == str(station)]
        if len(hit):
            row = hit.iloc[0]
    if row is None:
        raise ValueError(f"역_좌표.csv에서 정류장을 찾을 수 없음: {station!r}")
    return row


def _snap_latlon_to_node(lat: float, lon: float, role: str = "origin", k: int = 15) -> tuple:
    """임의의 (lat, lon)을 그래프에서 가장 가까운 노드로 스냅한다.

    role='origin'/'destination'로 방향 그래프의 막다른 노드(sink/source)를
    피한다: 최근접 노드가 origin인데 out_degree=0(나가는 길이 없는 종점)이거나
    destination인데 in_degree=0(들어오는 길이 없는 시작점)이면, k개 최근접
    후보 중 그 제약을 만족하는 다음으로 가까운 노드를 대신 채택한다."""
    node_coords = _load_node_coords()
    # 위경도 평면상 유클리드 근사(대전 시내 규모에서는 충분히 정확) - m 단위 아님, 상대 랭킹용
    dlat = node_coords["lat"] - lat
    dlon = (node_coords["lon"] - lon) * np.cos(np.radians(lat))
    dist_deg = np.sqrt(dlat ** 2 + dlon ** 2)
    order = dist_deg.nsmallest(k).index

    G = _load_base_graph()
    chosen_idx = None
    degree_rejected = 0
    for idx in order:
        node_id = node_coords.loc[idx, "NODE_ID"]
        if role == "origin" and G.out_degree(node_id) == 0:
            degree_rejected += 1
            continue
        if role == "destination" and G.in_degree(node_id) == 0:
            degree_rejected += 1
            continue
        chosen_idx = idx
        break
    if chosen_idx is None:  # k개 전부 막다른 노드인 극단적 경우 - 그냥 최근접으로 폴백
        chosen_idx = order[0]

    nearest = node_coords.loc[chosen_idx]
    dist_m = float(dist_deg.loc[chosen_idx] * 111_320)  # 1도 ≈ 111.32km 근사
    return nearest["NODE_ID"], dist_m, (degree_rejected > 0)


def resolve_station_node(station: str, role: str = "origin", k: int = 15) -> dict:
    """station: 정류장 이름(예: '정부청사') 또는 station_no(예: '218'/218) 모두 허용."""
    row = _lookup_station_row(station)
    node_id, dist_m, degree_fallback = _snap_latlon_to_node(row["lat"], row["lon"], role=role, k=k)

    return dict(
        station_name=row["station_name"], station_no=int(row["station_no"]),
        node_id=node_id, snap_distance_m=dist_m,
        degree_fallback_applied=degree_fallback,
    )


def resolve_coord_node(lat: float, lon: float, role: str = "origin", k: int = 15, label: str = "") -> dict:
    """정류장 이름 조회를 거치지 않고, 임의의 위경도 좌표를 직접 그래프 노드로 스냅한다."""
    node_id, dist_m, degree_fallback = _snap_latlon_to_node(lat, lon, role=role, k=k)
    return dict(
        station_name=label or f"({lat:.5f}, {lon:.5f})", station_no=None,
        lat=lat, lon=lon,
        node_id=node_id, snap_distance_m=dist_m,
        degree_fallback_applied=degree_fallback,
    )


# ---------------------------------------------------------------------------
# 비용함수 그래프 — 외부에서 계산한 위험도를 반영
# ---------------------------------------------------------------------------
def build_cost_graph(segment_risk_table: dict, penalty_scale: float = PENALTY_SCALE) -> nx.DiGraph:
    """
    segment_risk_table: {(segment_id, direction): {"prob_risk": float,
        "y_hat_t30": float | None}} — 호출자(XGBoost 파이프라인 보유자)가
        자신의 피처조립+모델로 계산해서 넘겨준다. y_hat_t30이 None이거나
        해당 segment_key가 LOW_CONF_SEGMENTS에 있으면 static_cost를
        base로 사용(그래프에 이미 계산돼 있는 값).
    """
    G = _load_base_graph().copy()
    station_links = _load_station_links()
    link_to_seg = station_links.drop_duplicates(subset="link_id", keep="first").set_index("link_id")[
        ["segment_id", "direction"]].to_dict("index")

    for _u, _v, data in G.edges(data=True):
        link_id = data["LINK_ID"]
        mapping = link_to_seg.get(link_id)
        if mapping is None:
            data["segment_id"] = None
            data["direction"] = None
            data["prob_risk"] = 0.0
            data["base"] = data["static_cost"]
            data["cost"] = data["static_cost"]
            continue

        seg_id, direction = mapping["segment_id"], mapping["direction"]
        seg_key_str = f"{seg_id}_{direction}"
        risk_info = segment_risk_table.get((seg_id, direction))

        prob_risk = float(risk_info["prob_risk"]) if risk_info and risk_info.get("prob_risk") is not None else 0.0
        y_hat_t30 = risk_info.get("y_hat_t30") if risk_info else None

        low_conf = (seg_key_str in LOW_CONF_SEGMENTS) or (y_hat_t30 is None) or (
            isinstance(y_hat_t30, float) and pd.isna(y_hat_t30))
        base = data["static_cost"] if low_conf else data["LENGTH"] / y_hat_t30

        data["segment_id"] = seg_id
        data["direction"] = direction
        data["prob_risk"] = prob_risk
        data["base"] = base
        data["cost"] = base * (1 + penalty_scale * prob_risk)

    return G


# ---------------------------------------------------------------------------
# 최단경로 탐색 및 비교
# ---------------------------------------------------------------------------
def _path_edges(G: nx.DiGraph, path: list) -> list:
    return [(u, v, G[u][v]) for u, v in zip(path[:-1], path[1:])]


def _path_pure_time_min(edges: list) -> float:
    return sum(d["base"] for _, _, d in edges) * _MIN_PER_UNIT


def _path_risk_exposure(edges: list) -> float:
    return sum((d["prob_risk"] or 0.0) * d["base"] * _MIN_PER_UNIT for _, _, d in edges)


def compare_paths(G: nx.DiGraph, o_node: str, d_node: str) -> dict:
    static_path = nx.shortest_path(G, o_node, d_node, weight="static_cost")
    predicted_path = nx.shortest_path(G, o_node, d_node, weight="cost")

    static_edges = _path_edges(G, static_path)
    predicted_edges = _path_edges(G, predicted_path)

    static_link_ids = {d["LINK_ID"] for _, _, d in static_edges}
    predicted_link_ids = {d["LINK_ID"] for _, _, d in predicted_edges}

    avoided_links = [
        dict(link_id=d["LINK_ID"], segment_id=d.get("segment_id"), direction=d.get("direction"),
             road_name=d["ROAD_NAME"], prob_risk=d.get("prob_risk"), u=u, v=v)
        for u, v, d in static_edges if d["LINK_ID"] not in predicted_link_ids
    ]
    new_links = [
        dict(link_id=d["LINK_ID"], segment_id=d.get("segment_id"), direction=d.get("direction"),
             road_name=d["ROAD_NAME"], prob_risk=d.get("prob_risk"), u=u, v=v)
        for u, v, d in predicted_edges if d["LINK_ID"] not in static_link_ids
    ]

    return dict(
        static_path=static_path,
        predicted_path=predicted_path,
        static_time_min=_path_pure_time_min(static_edges),
        predicted_time_min=_path_pure_time_min(predicted_edges),
        static_risk_exposure=_path_risk_exposure(static_edges),
        predicted_risk_exposure=_path_risk_exposure(predicted_edges),
        avoided_links=avoided_links,
        new_links=new_links,
        same_path=(static_path == predicted_path),
    )


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------
def find_route(origin, destination, segment_risk_table: dict,
                origin_is_coord: bool = False, destination_is_coord: bool = False,
                penalty_scale: float = PENALTY_SCALE) -> dict:
    """origin/destination: station_name(str) 또는 (lat, lon) 튜플.
    origin_is_coord/destination_is_coord로 어느 방식인지 지정.

    Returns
    -------
    dict with keys:
        static_path, predicted_path (노드ID 리스트),
        static_time_min, predicted_time_min (순수 이동시간, 분, penalty 제외),
        static_risk_exposure, predicted_risk_exposure (분 단위 위험노출량),
        avoided_links, new_links (LINK_ID/segment/prob_risk 딕셔너리 리스트),
        same_path (정적/예측 경로 동일 여부)
    """
    if origin_is_coord:
        origin_lat, origin_lon = origin
        origin_info = resolve_coord_node(origin_lat, origin_lon, role="origin")
    else:
        origin_info = resolve_station_node(origin, role="origin")

    if destination_is_coord:
        dest_lat, dest_lon = destination
        destination_info = resolve_coord_node(dest_lat, dest_lon, role="destination")
    else:
        destination_info = resolve_station_node(destination, role="destination")

    G = build_cost_graph(segment_risk_table, penalty_scale=penalty_scale)
    return compare_paths(G, origin_info["node_id"], destination_info["node_id"])
