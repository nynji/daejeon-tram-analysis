# 대전 트램 공사 구간 시공간 교통혼잡 예측 및 물류 대응  
## [모듈3] 매개중심성 네트워크 분석

> **전체 파이프라인에서 이 모듈의 위치**
>
> ```
> [전처리] 구간별 속도 피처 매트릭스
>        │
>        ├──▶ [모듈1] Prophet 시계열 예측  ──▶ [모듈2] XGBoost 분류
>        │                                         ▲
>        ├──▶ [모듈3] NetworkX 매개중심성  ────────┘  ← 이 저장소
>        │         (bc_its_realtime, bc_change, lane_remain_ratio + 기상·도로위계)
>        │
>        └──▶ [모듈4] DTW K-means 군집화  ──▶ [MCLP] AMR 거점 최적화
> ```
>
> 이 모듈의 핵심 질문: **"이 구간이 막히면 정체가 얼마나 넓게 퍼지는가?"**  
> 산출된 피처는 XGBoost 입력 행렬에 직접 결합됩니다.

---

## 📁 프로젝트 구조

```
C:\AI_Logistic\
├── README.md                            ← 이 파일
├── run.py                               ← 전체 분석 파이프라인 (원커맨드 실행)
├── visualize.py                         ← Folium 인터랙티브 지도 4종 생성
│
├── src/
│   ├── preprocess.py                    ← 데이터 로딩 + 정거장 구간(SEG_A_B) 매핑
│   ├── network_simulation.py            ← GIS 좌표 기반 공사 속도 패널티 (HCM 모델)
│   ├── analysis.py                      ← NetworkX 그래프 구성 + Edge BC 계산 (3모드)
│   ├── geocode_construction.py          ← 공사 통제 자연어 주소 → GIS 좌표 추출
│   ├── its_speed.py                     ← ITS 실측 링크 속도 처리
│   ├── weather_risk.py                  ← ASOS 실측 기상 → 위험 계수 변환
│   └── priority_score.py                ← 구간 우선순위 통합 스코어링 (0~100)
│
├── data/
│   ├── network/
│   │   ├── daejeon_link.geojson         ← 대전 표준 링크 (25,401개)
│   │   ├── daejeon_node.geojson         ← 대전 표준 노드 (18,165개, 교차로명 포함)
│   │   └── 역_좌표.csv                  ← 트램 45개 정거장 좌표
│   ├── daejeon_asos.csv                 ← 대전 ASOS 기상 (2024-10~2026-07, 15,624시간)
│   ├── ITS_대전_링크속도.csv            ← ITS 실측 속도 (14,221링크, 5분단위, 278만행)
│   ├── 트램_공구별_통제현황.xlsx        ← 14개 공구 공사 통제 현황 (19개 활성)
│   ├── 소방청_소방용수시설_20240207.csv ← 대전 소방용수 3,422개 (MCLP J_danger)
│   ├── 경찰청...택배소형화물...csv      ← 택배 주정차 허용 32개 구간 (MCLP 인센티브)
│   ├── 소상공인시장진흥공단_상가정보.csv ← 소상공인 상가 (MCLP 수요지점 I)
│   ├── 대전광역시_횡단보도.csv          ← 횡단보도 위치 (AMR 제약 10)
│   ├── 주차장 정보 표준데이터/          ← 5개 구 주차장 (MCLP 거점 후보 J)
│   ├── 어린이 보호구역 표준데이터/      ← 5개 구 (MCLP 제약 8)
│   └── 보행자 전용도로 표준데이터/      ← 5개 구 (AMR 이동 경로)
│
└── outputs/
    ├── network_betweenness.csv           ← ★ XGBoost 피처 (핵심 산출물)
    ├── segment_priority.csv              ← 96개 구간 우선순위
    ├── construction_zones_geocoded.csv   ← 19개 공사 구간 GIS 좌표
    ├── analysis_summary.md               ← 분석 요약 리포트
    ├── insight_report.md                 ← 보고서 인사이트 (도로 중요도 + 기상)
    ├── map1_free_flow_centrality.html    ← 🗺 공사 전 BC 분포
    ├── map2_risk_spillover.html          ← 🗺 정체 전이 위험 급증 구간
    ├── map3_before_after.html            ← 🗺 공사 전/후 레이어 비교
    └── map4_priority_dashboard.html      ← 🗺 ★ 우선순위 통합 대시보드
```

---

## 🛠️ 실행 방법

```bash
pip install networkx geopandas pandas openpyxl shapely folium
python run.py         # 전체 파이프라인 (BC 3종 + 기상 + 우선순위)
python visualize.py   # 지도 4종 생성
```

---

## 🔬 분석 방법론

### 1. 매개중심성 (Edge Betweenness Centrality)

$$C_B(e) = \sum_{s \neq t} \frac{\sigma_{st}(e)}{\sigma_{st}}$$

값이 높을수록 → 해당 링크 차단 시 우회 수요가 집중 → 정체 전이(Spill-over) 위험이 큼.

### 2. 그래프 설계 및 3가지 BC 시나리오

| 시나리오 | 엣지 가중치 (통행시간) | 용도 |
|---|---|---|
| `bc_free_flow` | LENGTH / MAX_SPD | 공사 전 기준선 (비교 기준) |
| `bc_under_construction` | LENGTH / (MAX_SPD × 패널티) | 공사 패널티 이론적 영향 |
| **`bc_its_realtime`** | LENGTH / **ITS 실측속도** | ★ 현실 반영 (가장 정확) |

- 방향성 가중 그래프 DiGraph (18,236 노드, 25,397 엣지)
- Dijkstra 최단경로 기반, 911개 노드 샘플 (5%, seed=42 고정, 재현성 보장)
- seed 간 Spearman rho=0.90, **Top 20 위험 구간 순위는 seed 무관 안정**
- ITS 실측 속도 14,221개 링크 적용 (56%), 나머지는 패널티 모델 폴백

### 3. 공사 패널티 산정 (잔여 차로 비율 + GIS 좌표)

**근거**: HCM 2010 (Highway Capacity Manual) + 대전시 교통대책 임계(20/15km/h)

| 잔여 차로 비율 | 패널티 | 실제 공사 예시 |
|:---:|:---:|---|
| > 75% | 0.75 | 도안대로 7공구 (왕복12→폐쇄2, 잔여83%) |
| 50~75% | 0.55 | 한밭대로 3공구 (왕복11→폐쇄3, 잔여73%) |
| 35~50% | 0.40 | 중앙로 13공구 (왕복4→폐쇄2, 잔여50%) |
| < 35% | 0.25 | 계족로 2공구 (편도4→폐쇄3, 잔여25%) |

**GIS 좌표 기반 세밀 적용** (v3):
- 공사 통제 자연어 주소 → 표준노드 교차로명 매칭 → 시점/종점 좌표 추출 (19/19 성공)
- 시점~종점 공간 버퍼 내 + 도로명 교차 링크만 패널티 → 도로 전체 오적용 방지

### 4. 기상 위험도 (ASOS 실측)

**데이터**: 대전 133번 관측소, 2024-10 ~ 2026-07 (15,624시간)

| 조건 | 위험 계수 | 근거 |
|---|:---:|---|
| 강수 ≥30mm/h | 0.80 | 속도 35% 저하, AMR 운영 불가 |
| 강수 20~29mm/h | 0.55 | 속도 25% 저하 |
| 강수 5~19mm/h | 0.30 | 속도 15% 저하 |
| 기온 < 0°C | +0.30 | 결빙 |
| 적설 ≥3cm | +0.40 | AMR 운영 불가 |

근거: 한국교통연구원(2019) 강수-속도 관계, HCM 2010

### 5. 우선순위 스코어 (0~100)

| 요소 | 가중치 | 출처 |
|---|:---:|---|
| BC 파급력 (bc_its_realtime) | 40% | 네트워크 분석 |
| 차로 감소 (1-lane_remain_ratio) | 30% | 공사 통제 현황 |
| 도로 위계 (ROAD_RANK) | 20% | 표준노드링크 |
| 기상 위험 (ASOS 실측) | 10% | 기상청 관측 |

등급: 🔴 심각(≥80) / 🟠 경고(60~79) / 🟡 주의(40~59) / 🟢 정상(<40)

---

## 📊 핵심 결과 (ITS 실측 기반)

### BC 최상위 도로 — 실제 파급력 순위

| 순위 | 도로명 | 위치 | BC(실측) | 교통적 의미 |
|:---:|---|---|:---:|---|
| 1 | **한밭대로** | 오정농수산물시장 | 0.068 | 대전 도심 중앙축. 트램 직접 경유. 차단 시 도시 이분 |
| 2 | **천변도시고속도로** | 둔산 | 0.065 | 갑천 변 남북 고속축. 공사 우회수요 최대 흡수처 |
| 3 | **대전천북로** | 오정 | 0.061 | 한밭대로 병행 우회로. 실제 부하 집중 확인 |

### ITS 실측 vs 패널티 모델 괴리

| 도로 | 실측 BC | 패널티 BC | 배율 | 해석 |
|---|:---:|:---:|:---:|---|
| 한밭대로 | 0.068 | 0.012 | **5.5×** | 패널티 모델이 위험 과소평가 |
| 천변도시고속도로 | 0.065 | 0.012 | **5.3×** | 실제 속도가 제한속도보다 훨씬 낮음 |
| 대전천북로 | 0.061 | 0.009 | **6.9×** | 실측 없으면 완전 누락되는 위험 |

> 이 차이가 XGBoost에 `bc_its_realtime`을 넣어야 하는 핵심 근거입니다.

---

## 🔗 XGBoost 피처 연결 명세

`outputs/network_betweenness.csv` 핵심 컬럼:

| 컬럼 | 역할 | 출처 |
|---|---|---|
| `bc_its_realtime` | ★ **실측 기반 구조적 파급력** | ITS 속도 → Dijkstra BC |
| `bc_its_vs_free` | 실측 vs 자유류 파급력 격차 | ITS BC − free BC |
| `bc_under_construction` | 패널티 모델 파급력 | MAX_SPD×패널티 → BC |
| `bc_change` | 공사 위험 증가분 | 패널티 BC − free BC |
| `lane_remain_ratio` | 직접 용량 지표 (0~1) | 통제현황 파싱 |
| `speed_penalty_multiplier` | 속도 패널티 배율 | HCM 모델 |
| `segment_id` | 구간 ID (`SEG_A_B`) | 정거장 쌍 매핑 |
| `nearest_station_name` | Prophet/군집 롤업 키 | 공간 매핑 |
| `ROAD_RANK` / `LANES` | 도로 위계/차로 수 | 표준노드링크 |

---

## ✅ 개선 이력

### v3 (현재 — 최종)

| # | 개선 내용 | 파일 |
|---|---|---|
| 5 | ASOS 실측 기상 데이터 통합 (651일, 15,624시간) | `src/weather_risk.py` |
| 6 | 구간 우선순위 스코어 (BC 40%+차로 30%+위계 20%+기상 10%) | `src/priority_score.py` |
| 7 | segment_id 진짜 구간 ID (SEG_A_B, 정거장 쌍 기반 96개) | `src/preprocess.py` |
| 8 | Map 4 우선순위 통합 대시보드 (등급별 레이어+Top10 패널) | `visualize.py` |
| 9 | 보고서 인사이트 자동 생성 (도로 중요도+기상 통계) | `run.py` |
| 10 | GIS 좌표 기반 세밀 패널티 (19/19 구간 좌표 추출) | `src/geocode_construction.py` |
| 11 | ITS 실측 링크 속도 추출 (1.7억행→278만행 대전 필터) | `src/its_speed.py` |
| 12 | **BC 3시나리오** (자유류/패널티/ITS실측) — 실측이 핵심 피처 | `src/analysis.py` |

### v2

| # | 개선 내용 |
|---|---|
| 1 | 잔여 차로 비율 기반 패널티 (HCM 2010 근거) |
| 2 | 복합 도로명 분리 파싱 (·, 괄호 처리) |
| 3 | 패널티 충돌 우선순위 (파싱값 > fallback) |
| 4 | `lane_remain_ratio` XGBoost 피처 추가 |

### v1

기본 BC 계산 + 키워드 3단계 패널티 + Folium 3종 시각화

---

## ⚠️ 한계 및 향후 과제

| 항목 | 현황 | 해결 방향 |
|---|---|---|
| ITS 속도 1일치 | 2026-07-07 하루만 보유 | 추가 기간 다운로드 시 즉시 적용 가능 (코드 완비) |
| GIS 세밀 패널티 커버리지 | 19구간 중 5개만 GIS 적용, 14개 도로명 폴백 | 시점≠종점인 구간만 GIS 가능 (데이터 한계) |
| BC 샘플링 | 5% (911노드), seed=42 고정 | seed 간 rho=0.90, Top20 안정. quantile 변환 권장 |
| 동광장로 미매칭 | 표준노드링크 미등재 | 중앙로 구간으로 간접 커버 확인 |
| 시변 패널티 | 공사 기간 내 단일 패널티 | Prophet changepoint 연동 예정 |

---

## 🤝 팀 연결 인터페이스

| 담당 모듈 | 이 모듈에서 공급하는 것 | 이 모듈이 받는 것 |
|---|---|---|
| [모듈1] Prophet | — | (미래 속도 예측 시 `bc_its_realtime`을 외생변수로 사용 가능) |
| [모듈2] XGBoost | `bc_its_realtime`, `bc_change`, `lane_remain_ratio`, `segment_id` | — |
| [모듈4] 군집분석 | `segment_id` (동일 롤업 키) | 상시정체형 구간 마스크 → MCLP 가중치 |
| MCLP 거점 최적화 | 고위험 구간 목록 (`segment_priority.csv`) | 주차장/상가/소방시설 데이터 |
| 대시보드 | `map4_priority_dashboard.html`, 모든 CSV | — |
