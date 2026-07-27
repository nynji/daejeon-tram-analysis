# 라우팅 파이프라인 사용 설명서

혼잡 위험도를 반영한 정류장/좌표 O-D 최단경로 계산 모듈. 22개 피처 조립과
XGBoost 호출은 포함하지 않는다 — 대시보드 담당자가 이미 보유한
파이프라인으로 위험도를 계산해서 넘겨주면, 이 모듈이 경로를 계산한다.

## 1. 필요한 데이터 파일

| 파일 | 역할 | 비고 |
|---|---|---|
| `graph_daejeon.gpickle` | 대전 방향성 도로망 그래프(networkx DiGraph) | 필수 |
| `node_coords.csv` | 그래프 노드 위경도(EPSG:4326) | 필수 |
| `segment_link_mapping.csv` | LINK_ID → 구간(segment_id, direction) 매핑 | 필수, 이미 보유 중 |
| `역_좌표.csv` | 정류장 이름 → 좌표 | 정류장 이름으로 호출할 때만 필요, 좌표 직접입력이면 불필요 |
| `batch_validation_results.csv` | 17개 시나리오 배치 검증 결과 | 참고용, 기능 실행에는 불필요 |
| `penalty_scale_comparison.csv` | penalty_scale 2 vs 5 비교 | 참고용, 기능 실행에는 불필요 |
| `penalty_scale_gridsearch.csv` | penalty_scale 2~8 그리드서치 | 참고용, 기능 실행에는 불필요 |

## 2. 코드 구성 (`pipeline_dashboard.py`)

| 함수 | 역할 |
|---|---|
| `find_route()` | **메인 진입점.** O-D + 위험도 표를 받아 경로를 계산한다 |
| `build_cost_graph()` | 위험도 표를 그래프에 반영해 비용함수 그래프를 만든다 |
| `compare_paths()` | 정적경로 vs 예측반영경로를 탐색·비교한다 |
| `resolve_station_node()` | 정류장 이름 → 그래프 노드 |
| `resolve_coord_node()` | (lat, lon) 좌표 → 그래프 노드 |
| `LOW_CONF_SEGMENTS` | 위험도는 있지만 속도예측(y_hat_t30)은 못 믿는 18개 구간 목록 |
| `PENALTY_SCALE` | 위험도 페널티 강도 상수(기본 2, 권장 4 — 6절 참고) |

## 3. 통합 방법

대시보드 담당자가 이미 가진 피처조립+모델 파이프라인으로 특정
시각의 90개 구간 `prob_risk`(및 가능하면 `y_hat_t30`)를 계산해
`segment_risk_table` 형태로 만든 다음, `find_route()`에 그대로 넘기면
경로가 나온다. 날짜/시각 검증, 22개 피처 조립, XGBoost 호출은 전부 당신
쪽 파이프라인의 책임이고 이 모듈은 그 결과만 소비한다.

## 4. 함수 시그니처 / 반환값

```python
def find_route(
    origin,                          # str(정류장 이름) 또는 (lat, lon) 튜플
    destination,                     # str(정류장 이름) 또는 (lat, lon) 튜플
    segment_risk_table: dict,        # {(segment_id, direction): {"prob_risk": float, "y_hat_t30": float | None}}
    origin_is_coord: bool = False,       # origin이 좌표 튜플이면 True
    destination_is_coord: bool = False,  # destination이 좌표 튜플이면 True
    penalty_scale: float = PENALTY_SCALE,
) -> dict
```

| 반환 키 | 설명 |
|---|---|
| `static_path` | 정적(위험도 미반영) 경로의 노드ID 리스트 |
| `predicted_path` | 위험도 반영 경로의 노드ID 리스트 |
| `static_time_min` | 정적 경로 순수 이동시간(분, 페널티 제외) |
| `predicted_time_min` | 예측반영 경로 순수 이동시간(분, 페널티 제외) |
| `static_risk_exposure` | 정적 경로의 위험노출량(Σ prob_risk × 구간 이동시간) |
| `predicted_risk_exposure` | 예측반영 경로의 위험노출량 |
| `avoided_links` | 정적경로에만 있고 예측반영경로가 피한 링크 목록(dict 리스트) |
| `new_links` | 예측반영경로가 새로 지나가는 링크 목록(dict 리스트) |
| `same_path` | 두 경로가 완전히 동일한지 여부(bool) |

`segment_risk_table`에 없는 구간은 `prob_risk=0`(페널티 없음)으로,
`y_hat_t30`이 없거나 `LOW_CONF_SEGMENTS`에 속한 구간은 `static_cost`
(설계속도 기준)로 처리된다.

## 5. 최소 실행 예시

```python
import sys
sys.path.insert(0, "routing/scripts")
import pipeline_dashboard as pd_

# 더미 위험도 표 (실제로는 당신의 XGBoost 파이프라인이 채워줌)
segment_risk_table = {
    ("SEG_16_216_217", "BA"): {"prob_risk": 0.46, "y_hat_t30": None},
    ("SEG_17_217_218", "BA"): {"prob_risk": 0.10, "y_hat_t30": 21.3},
    # ... 90개 구간 전부 채우는 게 이상적이나, 없는 구간은 자동으로 prob_risk=0 처리됨
}

result = pd_.find_route("정부청사", "둔산", segment_risk_table)
print(result["static_time_min"], result["predicted_time_min"], result["same_path"])

# 좌표로 직접 호출
result2 = pd_.find_route(
    (36.3614, 127.3836), (36.3678, 127.3834), segment_risk_table,
    origin_is_coord=True, destination_is_coord=True,
)
```

## 6. 알려진 제약

- 스냅샷 근사(FIFO 아님) — 순수 이동시간 1~25분 범위 시나리오로 검증됨
- `penalty_scale` 권장값은 4(기본값은 2로 남아있음, 필요시 호출부에서 지정)
- 일부 구간은 이산적 트레이드오프(직진 아니면 큰 우회, 중간 경로 없음)일 수 있음
