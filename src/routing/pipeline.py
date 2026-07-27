"""
정류장 O-D 리플레이 시나리오 파이프라인 — 핵심 로직.

날짜/시각(t0)을 받아 그 시점의 90개 구간(segment_id x direction) 상태를
스냅샷으로 재구성하고, XGBoost 혼잡 위험도로 가중된 비용함수 그래프에서
"예측반영 경로"와 "정적 경로"(원본 static_cost 기준)를 비교한다.

핵심 설계 결정(자세한 배경은 routing/README.md 참고):
- 스냅샷 근사: t0 시점 하나로 그래프 가중치를 고정하고 최단경로를 구한다.
  이동 중 시간이 흘러도 가중치가 갱신되지 않는 정적 스냅샷이라
  실제 FIFO(도착시간에 따라 간선비용이 달라지는) 라우팅과는 다르다 —
  이동시간이 짧은(분 단위) 근거리 시나리오이므로 근사 오차가 작다고 보고
  채택했다.
- 페널티는 곱셈: cost = base * (1 + PENALTY_SCALE * prob_risk). 덧셈이
  아니라 곱셈이라 base(순수 이동시간)가 큰 간선일수록 같은 위험도에도
  페널티 절대량이 커진다 — "위험한데 오래 걸리는 구간"을 더 강하게
  회피하게 만드는 의도적 설계.
- 18개 저신뢰(Prophet y_hat_t30 결측) 세그먼트: base를 length/y_hat_t30이
  아니라 static_cost(length/MAX_SPD, 설계속도 기준)로 대체한다. 실측 예측이
  없다고 그 구간을 그래프에서 빼거나 무한대 비용을 주는 게 아니라, "알고
  있는 것 중 가장 중립적인 값"으로 대체한다는 뜻이다. prob_risk는 이
  구간들도 XGBoost가 다른 21개 피처만으로 정상적으로 산출한다(y_hat_t30
  NaN은 XGBoost가 학습 시부터 처리하도록 설계된 결측이다).
"""

import math
import pickle
from functools import lru_cache

import numpy as np
import pandas as pd
import networkx as nx
import xgboost as xgb
from pyproj import Transformer

# ---------------------------------------------------------------------------
# 경로 상수
# ---------------------------------------------------------------------------
BASE_DIR = "."  # final_project/ 를 작업 디렉토리로 실행한다고 가정

HOLIDAY_CSV = f"{BASE_DIR}/prophet/dataset/modeling/공휴일_2024_2026.csv"
STATION_FEATURES_CSV = f"{BASE_DIR}/prophet/dataset/modeling/station_segment_features.csv"
RAIN_CSV = f"{BASE_DIR}/prophet/dataset/modeling/대전_강수량_2024_2026.csv"
PROPHET_PREDICTIONS_CSV = f"{BASE_DIR}/prophet/dataset/modeling/result/prophet_predictions.csv"
STATION_LINKS_CSV = f"{BASE_DIR}/prophet/dataset/network/정류장_구간_링크.csv"
STATION_COORDS_CSV = f"{BASE_DIR}/prophet/dataset/coords/역_좌표.csv"

GRAPH_PICKLE = f"{BASE_DIR}/routing/dataset/graph_daejeon.gpickle"
XGB_MODEL_JSON = f"{BASE_DIR}/routing/xgboost/v3_modelB_alpha_threshold_tuned.json"
BOTTLENECK_CSV = f"{BASE_DIR}/routing/xgboost/reference_tables/is_bottleneck_slot_TRAIN_ONLY.csv"
NETWORK_FEATURES_PARQUET = f"{BASE_DIR}/routing/xgboost/reference_tables/network_features.parquet"
LANE_RATIO_PARQUET = f"{BASE_DIR}/routing/xgboost/reference_tables/construction_lane_ratio_daily.parquet"

# ---------------------------------------------------------------------------
# 도메인 상수
# ---------------------------------------------------------------------------
DATA_START = pd.Timestamp("2024-10-01")
DATA_END = pd.Timestamp("2026-06-30 23:50:00")  # 마지막 온전한 10분 슬롯
MIN_LOOKAHEAD_HOURS = 6

# 90개 중 Prophet y_hat_t30 신뢰 불가(2026-07-18 성능 재검토에서 확정) 18개
LOW_CONF_SEGMENTS = {
    "SEG_11_211_212_BA", "SEG_14_214_215_AB", "SEG_14_214_215_BA",
    "SEG_16_216_217_AB", "SEG_16_216_217_BA", "SEG_21_221_222_AB", "SEG_21_221_222_BA",
    "SEG_28_228_229_BA", "SEG_29_229_230_AB", "SEG_29_229_230_BA",
    "SEG_31_231_232_AB", "SEG_31_231_232_BA", "SEG_35_235_236_BA",
    "SEG_37_237_238_BA", "SEG_40_240_201_BA", "SEG_42_241_242_AB",
    "SEG_42_241_242_BA", "SEG_45_233_245_AB",
}

PENALTY_SCALE = 2
BEST_THRESHOLD = 0.25
SEVERE_CLASS_IDX = 2

FEATURE_ORDER = [
    "V_segment", "speed_last_10min", "speed_ma_30min", "speed_ma_1h", "speed_change_rate",
    "hour", "dow", "is_weekend", "is_bottleneck_slot",
    "betweenness_pre", "betweenness_during", "road_rank", "lanes",
    "lane_remain_ratio", "incident_flag", "incident_count",
    "precipitation_mm", "is_weather_alert", "is_freezing",
    "y_hat_t30", "y_hat_lower_t30", "y_hat_upper_t30",
]
BOOL_COLS = ["is_weekend", "is_bottleneck_slot", "incident_flag", "is_weather_alert", "is_freezing"]

_MIN_PER_UNIT = 0.06  # base(=LENGTH[m]/speed[km/h]) -> 분 환산 (÷1000 * 60)


# ---------------------------------------------------------------------------
# 캐시된 정적 자원 로드 (프로세스당 1회) — 배치로 여러 시나리오를 돌릴 때
# 같은 8GB급 CSV/그래프/모델을 매번 다시 읽지 않기 위함
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _load_holidays() -> set:
    hol = pd.read_csv(HOLIDAY_CSV, dtype=str)
    return set(hol["date"])


@lru_cache(maxsize=1)
def _load_station_features() -> pd.DataFrame:
    return pd.read_csv(
        STATION_FEATURES_CSV,
        usecols=["timestamp", "segment_id", "direction", "y_hat"],
        parse_dates=["timestamp"],
    )


@lru_cache(maxsize=1)
def _load_station_links() -> pd.DataFrame:
    return pd.read_csv(STATION_LINKS_CSV, dtype=str)


@lru_cache(maxsize=1)
def _load_station_coords() -> pd.DataFrame:
    return pd.read_csv(STATION_COORDS_CSV)


@lru_cache(maxsize=1)
def _load_rain() -> pd.DataFrame:
    return pd.read_csv(RAIN_CSV, parse_dates=["timestamp"])


@lru_cache(maxsize=1)
def _load_bottleneck() -> pd.DataFrame:
    return pd.read_csv(BOTTLENECK_CSV)


@lru_cache(maxsize=1)
def _load_network_features() -> pd.DataFrame:
    return pd.read_parquet(NETWORK_FEATURES_PARQUET)


@lru_cache(maxsize=1)
def _load_lane_ratio() -> pd.DataFrame:
    return pd.read_parquet(LANE_RATIO_PARQUET)


@lru_cache(maxsize=1)
def _load_xgb_model():
    model = xgb.XGBClassifier()
    model.load_model(XGB_MODEL_JSON)
    return model


@lru_cache(maxsize=1)
def _load_base_graph() -> nx.DiGraph:
    with open(GRAPH_PICKLE, "rb") as f:
        return pickle.load(f)


NODE_COORDS_CSV = f"{BASE_DIR}/routing/dataset/node_coords.csv"


@lru_cache(maxsize=1)
def _load_node_coords() -> pd.DataFrame:
    """그래프 노드(18,215개)의 위경도(EPSG:4326) 좌표 lookup.

    MOCT_NODE.shp에서 미리 뽑아 routing/node_coords.csv로 저장해둔 것을 읽는다
    (재생성 스크립트: routing/dev/build_node_coords.py, geopandas 필요).
    pipeline.py 자체는 geopandas 없이 pandas만으로 실행 가능하게 하기 위한
    분리다."""
    return pd.read_csv(NODE_COORDS_CSV, dtype={"NODE_ID": str})


_PROPHET_DAY_CACHE: dict[str, pd.DataFrame] = {}


def _load_prophet_day(date_str: str) -> pd.DataFrame:
    """prophet_predictions.csv(7GB+)에서 특정 날짜 하루치만 뽑아 캐시한다.
    같은 날짜를 여러 시나리오가 재사용하면 전체 스캔은 날짜당 1번만 발생."""
    if date_str in _PROPHET_DAY_CACHE:
        return _PROPHET_DAY_CACHE[date_str]

    chunks = pd.read_csv(
        PROPHET_PREDICTIONS_CSV,
        usecols=["segment_id", "direction", "timestamp", "yhat", "yhat_lower", "yhat_upper"],
        chunksize=2_000_000,
    )
    rows = []
    for chunk in chunks:
        hit = chunk[chunk["timestamp"].str.startswith(date_str)]
        if len(hit):
            rows.append(hit)
    day_df = pd.concat(rows, ignore_index=True)
    day_df["timestamp"] = pd.to_datetime(day_df["timestamp"])
    _PROPHET_DAY_CACHE[date_str] = day_df
    return day_df


# ---------------------------------------------------------------------------
# 작업0 로직: 리플레이 날짜/시각 유효성 검증
# ---------------------------------------------------------------------------
def _check_replay_datetime(t0: pd.Timestamp) -> tuple:
    """(유효여부, 사유 리스트) 반환. 세 조건:
    1) 평일 + 공휴일 아님
    2) t0+6시간이 데이터 끝(2026-06-30 23:50)을 넘지 않음
    3) 90개 segment x direction 전부 t0 시점 실측 y_hat 결측 없음
    """
    reasons = []
    date_str = t0.strftime("%Y-%m-%d")

    if t0.dayofweek >= 5:
        reasons.append(f"주말({t0.day_name()})")
    if date_str in _load_holidays():
        reasons.append("공휴일")
    if t0 + pd.Timedelta(hours=MIN_LOOKAHEAD_HOURS) > DATA_END:
        reasons.append(f"t0+{MIN_LOOKAHEAD_HOURS}시간이 데이터 끝({DATA_END})을 초과")

    feat = _load_station_features()
    at_t0 = feat[feat["timestamp"] == t0]
    if len(at_t0) < 90:
        reasons.append(f"t0 시점 데이터 자체가 90개 미만({len(at_t0)}개)")
    elif at_t0["y_hat"].isna().any():
        n_missing = int(at_t0["y_hat"].isna().sum())
        reasons.append(f"90개 구간 중 {n_missing}개 t0 시점 실측 결측")

    return (len(reasons) == 0), reasons


def resolve_replay_datetime(preferred_date: str, t0_time: str, search_radius_days: int = 180) -> dict:
    """preferred_date가 조건을 만족하면 그대로, 아니면 같은 t0_time으로
    가장 가까운(±1일, ±2일, ... 순) 대체 날짜를 데이터 범위 내에서 탐색한다."""
    preferred = pd.Timestamp(f"{preferred_date} {t0_time}")
    ok, reasons = _check_replay_datetime(preferred)
    if ok:
        return dict(t0=preferred, fallback_applied=False, original_reasons_rejected=[])

    log = [f"[날짜검증] {preferred} 조건 불만족: {', '.join(reasons)} -> 대체 날짜 탐색"]
    tried_reasons = {}
    for offset in range(1, search_radius_days + 1):
        for sign in (-1, 1):
            cand_date = preferred.normalize() + pd.Timedelta(days=sign * offset)
            if cand_date < DATA_START or cand_date > DATA_END.normalize():
                continue
            cand_t0 = pd.Timestamp(f"{cand_date.strftime('%Y-%m-%d')} {t0_time}")
            ok2, reasons2 = _check_replay_datetime(cand_t0)
            if ok2:
                log.append(f"[날짜검증] 대체 채택: {cand_t0}")
                return dict(t0=cand_t0, fallback_applied=True, original_reasons_rejected=reasons, log=log)
            tried_reasons[str(cand_t0)] = reasons2

    raise RuntimeError(
        f"{search_radius_days}일 반경 내에서 조건 만족하는 대체 날짜를 찾지 못함. "
        f"원래 사유: {reasons}, 탐색 실패 사유 예시: {dict(list(tried_reasons.items())[:5])}"
    )


# ---------------------------------------------------------------------------
# 작업1: O-D 정류장 -> 그래프 노드 스냅
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


def resolve_station_node(station: str, role: str = "origin", k: int = 15) -> dict:
    """station: 정류장 이름(예: '정부청사') 또는 station_no(예: '218'/218) 모두 허용.

    role='origin'/'destination'로 방향 그래프의 막다른 노드(sink/source)를
    피한다: 최근접 노드가 origin인데 out_degree=0(나가는 길이 없는 종점)이거나
    destination인데 in_degree=0(들어오는 길이 없는 시작점)이면, k개 최근접
    후보 중 그 제약을 만족하는 다음으로 가까운 노드를 대신 채택한다. 실제
    도로망에서 정류장 좌표가 우연히 일방통행 막다른 노드에 스냅되는 경우가
    있어 넣은 방어 로직이다(예: '대정' 정류장이 out_degree=0 노드에 스냅되던
    문제, 2026-07 배치 검증에서 발견).

    (2026-07-25 진단: 정류장 하나씩 독립적으로 "가장 가까운 노드"만 보는
    방식이라 O-D 두 정류장을 동시에 고려하지 않는다는 이론적 약점이 있어서
    resolve_station_node_v2()(팀원의 8후보+가상 source/sink 앵커 결정 방식)와
    비교 검증했다. 결과: 17개 시나리오 전부 이 함수가 v2보다 "실제로 그
    세그먼트 자체의 링크를 지나가는" 비율이 높았다(16/17 vs 7/17) — v2가 더
    정교해 보이는 방법이었지만 실측 비교에서는 오히려 이 단순한 방식이 더
    나았다. 그래서 기본값은 그대로 이 함수로 유지한다. 유일하게 이 함수도
    실패하는 SEG_13_213_214_AB는 v2로도 못 고쳤다 — 앵커 선택 문제가
    아니라, 그 세그먼트에 공식 배정된 링크 체인이 애초에 두 정류장 사이의
    최단(또는 최적) 경로가 아닌 것으로 보인다(같은 도로 한밭대로의 인접한
    다른 링크가 더 직선적인 경로를 만듦). resolve_station_node_v2()는
    삭제하지 않고 남겨뒀다 — 개별 시나리오를 교차검증하고 싶을 때 쓸 수
    있다."""
    row = _lookup_station_row(station)
    node_id, dist_m, degree_fallback = _snap_latlon_to_node(row["lat"], row["lon"], role=role, k=k)

    return dict(
        station_name=row["station_name"], station_no=int(row["station_no"]),
        node_id=node_id, snap_distance_m=dist_m,
        degree_fallback_applied=degree_fallback,
    )


def _snap_latlon_to_node(lat: float, lon: float, role: str = "origin", k: int = 15) -> tuple:
    """임의의 (lat, lon)을 그래프에서 가장 가까운 노드로 스냅한다(정류장 이름
    조회 없이 좌표만으로 동작) - resolve_station_node의 degree 방어 로직을
    그대로 공유하는 공용 구현. resolve_coord_node()와 resolve_station_node()가
    둘 다 이 함수를 쓴다."""
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


def resolve_coord_node(lat: float, lon: float, role: str = "origin", k: int = 15, label: str = "") -> dict:
    """정류장 이름 조회를 거치지 않고, 임의의 위경도 좌표를 직접 그래프
    노드로 스냅한다. 실제 내비게이션은 정류장과 무관한 임의 좌표를 입력받으므로
    (2026-07-27 검증), 그 경로를 그대로 지원하는 함수 - resolve_station_node와
    동일한 degree 방어 로직(_snap_latlon_to_node)을 공유한다."""
    node_id, dist_m, degree_fallback = _snap_latlon_to_node(lat, lon, role=role, k=k)
    return dict(
        station_name=label or f"({lat:.5f}, {lon:.5f})", station_no=None,
        lat=lat, lon=lon,
        node_id=node_id, snap_distance_m=dist_m,
        degree_fallback_applied=degree_fallback,
    )


# ---------------------------------------------------------------------------
# resolve_station_node_v2 — 팀원이 정류장_구간_링크.csv를 만들 때 쓴 앵커
# 결정 방식(8후보 + 가상 source/sink + 거리 기반 스냅비용)을 그대로 재사용.
# 이 리포지토리 어디에도 파일로 존재하지 않는 코드라 사용자가 대화로
# 전달한 원본을 그대로 옮겼다 - attach_snap_candidates/shortest_link_path의
# 로직은 한 줄도 바꾸지 않았고, 우리 쪽에서 준비하는 입력 그래프만 다르다.
#
# 원본은 length_m 그래프에서 돌았는데 우리 routing 그래프의 기본 weight는
# cost(시간+위험 페널티)라 단위가 안 맞는다. 그래서 "앵커 결정"(이 아래,
# length_m 무방향 그래프)과 "실제 경로 비용 계산"(build_cost_graph의 cost
# 그래프)을 완전히 분리했다 - 앵커만 이걸로 정하고, 실제 라우팅은 원래
# 방식 그대로 cost 그래프에서 한다.
# ---------------------------------------------------------------------------
SNAP_CANDIDATE_COUNT = 8
SNAP_COST_MULTIPLIER = 3.0


def attach_snap_candidates(stations, node_xy):
    """Transform stations to EPSG:5186 and keep their eight nearest nodes."""
    transformer = Transformer.from_crs(4326, 5186, always_xy=True)
    nodes = list(node_xy.items())
    for station in stations:
        x, y = transformer.transform(station["lon"], station["lat"])
        candidates = sorted(
            (math.hypot(x - node_x, y - node_y), node_id)
            for node_id, (node_x, node_y) in nodes
        )
        station["snap_candidates"] = candidates[:SNAP_CANDIDATE_COUNT]


def shortest_link_path(graph, start_station, end_station):
    """Find a direction-aware route, including station-to-node snap costs."""
    source = f"__source_{start_station['station_no']}"
    sink = f"__sink_{end_station['station_no']}"
    graph.add_node(source)
    graph.add_node(sink)
    for distance_m, node_id in start_station["snap_candidates"]:
        graph.add_edge(source, node_id, weight=max(distance_m * SNAP_COST_MULTIPLIER, 0.01))
    for distance_m, node_id in end_station["snap_candidates"]:
        graph.add_edge(node_id, sink, weight=max(distance_m * SNAP_COST_MULTIPLIER, 0.01))
    try:
        node_path = nx.shortest_path(graph, source, sink, weight="weight")
        return [
            dict(graph[f_node][t_node])
            for f_node, t_node in zip(node_path[1:-2], node_path[2:-1])
        ]
    except nx.NetworkXNoPath as exc:
        raise RuntimeError(
            f"No directed road path: {start_station['station_no']} -> {end_station['station_no']}"
        ) from exc
    finally:
        graph.remove_nodes_from([source, sink])


@lru_cache(maxsize=1)
def _build_node_xy_5186() -> dict:
    """routing/dataset/node_coords.csv(EPSG:4326)를 EPSG:5186으로 변환해 attach_snap_candidates가
    기대하는 {node_id: (x, y)} 형태로 만든다. 원래는 shapefile에서 직접 뽑았지만
    node_coords.csv로도 동일한 좌표를 얻을 수 있어 geopandas 불필요."""
    node_coords = _load_node_coords()
    transformer = Transformer.from_crs(4326, 5186, always_xy=True)
    xs, ys = transformer.transform(node_coords["lon"].values, node_coords["lat"].values)
    return dict(zip(node_coords["NODE_ID"], zip(xs, ys)))


@lru_cache(maxsize=1)
def _load_length_undirected_graph() -> nx.Graph:
    """graph_daejeon.gpickle을 length_m 기준 무방향 그래프로 복사(원본은 안 건드림).
    shortest_link_path가 기대하는 'weight' 속성(=LENGTH)과, 앵커 노드를 뽑아낼 때
    쓰는 'F_NODE'/'T_NODE' 속성을 채워 넣는다(원 그래프 엣지엔 u,v가 곧 그 자체라
    별도 속성으로 없었음 - 팀원 함수가 기대하는 형태로 여기서만 맞춰준다)."""
    G = _load_base_graph().to_undirected(as_view=False)
    for u, v, data in G.edges(data=True):
        data["weight"] = data["LENGTH"]
        data["F_NODE"] = u
        data["T_NODE"] = v
    return G


def resolve_station_node_v2(origin_name: str, destination_name: str) -> dict:
    """O-D 두 정류장을 동시에 고려해 앵커 노드를 정한다(팀원이
    정류장_구간_링크.csv를 만들 때 쓴 8후보+가상 source/sink 방식,
    length_m 무방향 그래프 기준).

    2026-07-25 진단 결과: resolve_station_node(정류장 하나씩 독립적으로
    최근접 노드만 보는 단순한 방식)보다 "실제로 그 세그먼트 링크를
    지나가는 비율"이 오히려 낮았다(7/17 vs 16/17) — 기본 경로 탐색에는
    쓰지 않는다(run_routing_scenario는 resolve_station_node를 그대로
    씀). 특정 O-D 조합의 앵커를 교차검증하고 싶을 때 참고용으로 호출하는
    용도로 남겨뒀다."""
    origin_row = _lookup_station_row(origin_name)
    dest_row = _lookup_station_row(destination_name)

    start_station = dict(
        station_no=int(origin_row["station_no"]), station_name=origin_row["station_name"],
        lat=float(origin_row["lat"]), lon=float(origin_row["lon"]),
    )
    end_station = dict(
        station_no=int(dest_row["station_no"]), station_name=dest_row["station_name"],
        lat=float(dest_row["lat"]), lon=float(dest_row["lon"]),
    )

    node_xy = _build_node_xy_5186()
    attach_snap_candidates([start_station, end_station], node_xy)

    G_len = _load_length_undirected_graph().copy()
    route = shortest_link_path(G_len, start_station, end_station)

    if not route:
        raise RuntimeError(
            f"{origin_name} -> {destination_name}: 앵커 결정 경로가 비어있음"
            f"(두 정류장이 같은 노드 근처로 스냅됐을 가능성)"
        )

    origin_anchor = route[0]["F_NODE"]
    destination_anchor = route[-1]["T_NODE"]

    return dict(
        origin=dict(station_name=origin_row["station_name"], station_no=int(origin_row["station_no"]),
                    node_id=origin_anchor),
        destination=dict(station_name=dest_row["station_name"], station_no=int(dest_row["station_no"]),
                          node_id=destination_anchor),
        route_length_m=sum(e["LENGTH"] for e in route),
        route_link_ids=[e["LINK_ID"] for e in route],
    )


# ---------------------------------------------------------------------------
# 작업2: 22개 피처 배치 조립
# ---------------------------------------------------------------------------
def assemble_features(t0: pd.Timestamp) -> pd.DataFrame:
    date_str = t0.strftime("%Y-%m-%d")
    slot_times = [t0 - pd.Timedelta(minutes=10 * k) for k in range(6)]

    speed = _load_station_features()
    speed = speed[speed["timestamp"].isin(slot_times)].copy()
    speed_pivot = speed.pivot_table(index=["segment_id", "direction"], columns="timestamp", values="y_hat")
    speed_pivot = speed_pivot[slot_times]

    feat = speed_pivot.reset_index()
    feat.columns = ["segment_id", "direction"] + [f"slot_{k}" for k in range(6)]
    feat["segment_key"] = feat["segment_id"] + "_" + feat["direction"]

    feat["V_segment"] = feat["slot_0"]
    feat["speed_last_10min"] = feat["slot_1"]
    feat["speed_ma_30min"] = feat[["slot_0", "slot_1", "slot_2"]].mean(axis=1)
    feat["speed_ma_1h"] = feat[[f"slot_{k}" for k in range(6)]].mean(axis=1)
    feat["speed_change_rate"] = feat["speed_ma_30min"] - feat["speed_ma_1h"]

    feat["hour"] = t0.hour
    feat["dow"] = t0.dayofweek
    feat["is_weekend"] = int(t0.dayofweek >= 5)

    bottleneck = _load_bottleneck()
    time_slot = t0.hour * 100 + t0.minute
    bn_t0 = bottleneck[bottleneck["time_slot"] == time_slot][["segment_key", "is_bottleneck_slot"]]
    feat = feat.merge(bn_t0, on="segment_key", how="left")

    net = _load_network_features()
    feat = feat.merge(net, on="segment_key", how="left")

    lane_ratio = _load_lane_ratio()
    lane_ratio_day = lane_ratio[lane_ratio["date"] == pd.Timestamp(date_str)][["segment_id", "lane_remain_ratio"]]
    feat = feat.merge(lane_ratio_day, on="segment_id", how="left")

    feat["incident_flag"] = 0
    feat["incident_count"] = 0

    rain = _load_rain()
    rain_hour = rain[rain["timestamp"] == pd.Timestamp(f"{date_str} {t0.hour:02d}:00")]
    precip = float(rain_hour["precipitation_mm"].iloc[0]) if len(rain_hour) else np.nan
    feat["precipitation_mm"] = precip
    feat["is_weather_alert"] = int(precip >= 30) if pd.notna(precip) else 0
    feat["is_freezing"] = 0

    t0_plus_30 = t0 + pd.Timedelta(minutes=30)
    day_df = _load_prophet_day(date_str)
    if t0_plus_30.strftime("%Y-%m-%d") != date_str:
        day_df = pd.concat([day_df, _load_prophet_day(t0_plus_30.strftime("%Y-%m-%d"))], ignore_index=True)
    prophet_t30 = day_df[day_df["timestamp"] == t0_plus_30][
        ["segment_id", "direction", "yhat", "yhat_lower", "yhat_upper"]
    ].rename(columns={"yhat": "y_hat_t30", "yhat_lower": "y_hat_lower_t30", "yhat_upper": "y_hat_upper_t30"})
    feat = feat.merge(prophet_t30, on=["segment_id", "direction"], how="left")

    mask_low = feat["segment_key"].isin(LOW_CONF_SEGMENTS)
    feat.loc[mask_low, ["y_hat_t30", "y_hat_lower_t30", "y_hat_upper_t30"]] = np.nan

    for c in BOOL_COLS:
        feat[c] = feat[c].astype("boolean").fillna(False).astype("int8")

    return feat.sort_values(["segment_id", "direction"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 작업3: XGBoost 스코어링
# ---------------------------------------------------------------------------
def _predict_with_threshold(proba: np.ndarray, threshold_severe: float = BEST_THRESHOLD) -> np.ndarray:
    return np.where(
        proba[:, SEVERE_CLASS_IDX] >= threshold_severe,
        SEVERE_CLASS_IDX,
        np.argmax(proba[:, :2], axis=1),
    )


def score_with_xgboost(feat: pd.DataFrame) -> pd.DataFrame:
    model = _load_xgb_model()
    X = feat[FEATURE_ORDER].copy()
    proba = model.predict_proba(X)
    feat = feat.copy()
    feat["prob_normal"] = proba[:, 0]
    feat["prob_caution"] = proba[:, 1]
    feat["prob_severe"] = proba[:, 2]
    feat["prob_risk"] = proba[:, 1] + proba[:, 2]
    pred = _predict_with_threshold(proba)
    feat["predicted_class"] = pred
    feat["predicted_label"] = feat["predicted_class"].map({0: "정상", 1: "주의", 2: "심각"})
    return feat


# ---------------------------------------------------------------------------
# 작업4: 비용함수 스냅샷 그래프
# ---------------------------------------------------------------------------
def build_cost_graph(feat: pd.DataFrame) -> nx.DiGraph:
    G = _load_base_graph().copy()
    station_links = _load_station_links()
    link_to_seg = station_links.drop_duplicates(subset="link_id", keep="first").set_index("link_id")[
        ["segment_id", "direction"]].to_dict("index")
    risk_lookup = feat.set_index("segment_key")[["prob_risk", "y_hat_t30"]].to_dict("index")

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
        seg_key = f"{seg_id}_{direction}"
        risk_info = risk_lookup[seg_key]
        prob_risk = float(risk_info["prob_risk"])
        y_hat_t30 = risk_info["y_hat_t30"]

        base = data["static_cost"] if (seg_key in LOW_CONF_SEGMENTS or pd.isna(y_hat_t30)) else data["LENGTH"] / y_hat_t30

        data["segment_id"] = seg_id
        data["direction"] = direction
        data["prob_risk"] = prob_risk
        data["base"] = base
        data["cost"] = base * (1 + PENALTY_SCALE * prob_risk)

    return G


# ---------------------------------------------------------------------------
# 작업5: 최단경로 탐색 및 비교
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
def run_routing_scenario(origin_station: str, destination_station: str, replay_date: str, t0_time: str) -> dict:
    """정류장 O-D + 날짜/시각으로 정적경로 vs 예측반영경로를 비교한다.

    Returns
    -------
    dict with keys:
        static_path, predicted_path (노드ID 리스트),
        static_time_min, predicted_time_min (순수 이동시간, 분, penalty 제외),
        static_risk_exposure, predicted_risk_exposure (분 단위 위험노출량),
        avoided_links, new_links (LINK_ID/segment/prob_risk 딕셔너리 리스트),
        segment_prob_risk (90방향 {segment_key: prob_risk} 표),
        t0 (실제 사용된 리플레이 시각), date_fallback_applied (bool),
        origin / destination (역_좌표 조회 결과 dict, node_id 포함)
    """
    origin = resolve_station_node(origin_station, role="origin")
    destination = resolve_station_node(destination_station, role="destination")
    return _run_scenario_core(origin, destination, replay_date, t0_time)


def run_routing_scenario_from_coords(origin_lat: float, origin_lon: float,
                                      destination_lat: float, destination_lon: float,
                                      replay_date: str, t0_time: str,
                                      origin_label: str = "", destination_label: str = "") -> dict:
    """정류장 이름을 거치지 않고 임의의 (lat, lon) 좌표로 직접 시나리오를
    돌린다. resolve_coord_node()로 노드 스냅만 하고, 나머지(날짜 검증,
    피처 조립, XGBoost 스코어링, 비용함수 그래프, 경로 비교)는
    run_routing_scenario()와 완전히 동일한 _run_scenario_core()를 공유한다."""
    origin = resolve_coord_node(origin_lat, origin_lon, role="origin", label=origin_label)
    destination = resolve_coord_node(destination_lat, destination_lon, role="destination", label=destination_label)
    return _run_scenario_core(origin, destination, replay_date, t0_time)


def _run_scenario_core(origin: dict, destination: dict, replay_date: str, t0_time: str) -> dict:
    date_info = resolve_replay_datetime(replay_date, t0_time)
    t0 = date_info["t0"]
    if date_info["fallback_applied"]:
        for line in date_info.get("log", []):
            print(line)

    feat = assemble_features(t0)
    feat = score_with_xgboost(feat)

    G = build_cost_graph(feat)
    result = compare_paths(G, origin["node_id"], destination["node_id"])

    result.update(dict(
        t0=t0,
        requested_date=replay_date,
        requested_t0_time=t0_time,
        date_fallback_applied=date_info["fallback_applied"],
        origin=origin,
        destination=destination,
        segment_prob_risk=feat.set_index("segment_key")["prob_risk"].to_dict(),
        feature_table=feat,
        graph=G,
    ))
    return result
