# [모듈5] 위험조정 최대 커버리지 입지 최적화 (MCLP)

대전 트램 공사 영향권 내 AMR-탑차 연계 거점의 최적 배치를 산출하는 수리최적화 모듈.

## 실행 방법

```bash
# 솔버 단독 실행 (CLI)
pip install pulp pyyaml pandas numpy
python -m src.mclp.run_mclp_v2 --mode SCREENING --p-max 15

# 웹 관제 시스템
pip install fastapi uvicorn pulp pyyaml pandas numpy xlrd openpyxl geopy pyproj
uvicorn webapp.main:app --port 8000          # 백엔드

cd webapp
npm install                                   # 프론트 의존성 (최초 1회)
npx vite --port 3000                          # 프론트엔드

# 접속: http://localhost:3000
```

## 핵심 산출물

| 파일 | 내용 |
|---|---|
| `outputs/latest/scenario_summary.csv` | P 탐색 + 27개 민감도 결과 |
| `outputs/latest/selected_anchors.csv` | 시나리오별 선택 거점 + 강건성 |
| `outputs/latest/assignments.csv` | 수요별 배정 거점, 거리, 위험도 |
| `outputs/latest/robustness_summary.csv` | 후보별 선택 빈도 |
| 웹 대시보드 | 12개 파라미터 실시간 제어 + 8개 지도 레이어 |

## 모형 개요

목적함수: 위험조정 가중 커버리지 극대화

```
max Σ w_i × (1 - β × route_risk_ij) × y_ij
```

- `w_i`: DTW 군집 기반 수요 가중치 (상시 저속형 ×1.37, RAPID ×1.10)
- `route_risk_ij`: 보행망 기반 경로 위험도 (보도폭 + 경사 + 횡단)
- `β`: 위험 감점 강도 (0.00~0.30)

제약식:
- 거점 수 ≤ P
- 단일 배정 (수요당 1거점)
- 배정 종속 (미오픈 거점 배정 차단)
- 최소 사용 (오픈 거점은 1개 이상 배정)
- 거점 이격 (D_min 이내 중복 오픈 금지)
- 전체 최소 커버리지 80%
- 자치구별 최소 커버리지 50%

## 데이터 의존성

솔버는 `data/mclp 입력 데이터/` 7개 정본 CSV를 입력으로 사용:

| 파일 | 행 수 | 역할 |
|---|---|---|
| demand_points.csv | 1,364 | 스크리닝 통과 수요 (250m 격자) |
| candidate_sites.csv | 562 | 거점 후보 (허용 205) |
| coverage_matrix.csv | 20,674 | 보행망 경로 + 위험도 |
| xgb_segment_risk.csv | 90 | XGBoost 정체 예측 |
| segment_priority.csv | 45 | DTW 군집 우선순위 |
| scenario_params.csv | 47 | 시나리오 파라미터 |
| exclusion_log.csv | 18,540 | 제외 사유 기록 |

## 핵심 결과

| 지표 | 값 |
|---|---|
| 최적 거점 수 P* | 15 |
| 가중 커버리지 (WCR) | 95.1% |
| 비가중 커버리지 (UCR) | 80.7% |
| 솔버 시간 | 500~700ms |
| 자치구별 최소 | 70.3% (대덕구) |
| 시나리오 (Optimal/Total) | 33/42 |

## 파일 구조

```
src/mclp/
├── README.md                이 문서
├── __init__.py              패키지 정의
├── config.py                YAML 설정 로딩 + 검증
├── data_loader.py           11개 데이터 소스 로딩
├── distance.py              Haversine 거리 행렬
├── weights.py               DTW 기반 수요 가중치
├── constraints.py           Hard 제약 엔진
├── solver_v2.py             Risk-MCLP ILP 솔버
├── run_mclp_v2.py           시나리오 파이프라인
├── xgb_integration.py       XGBoost 위험도 연동
├── output.py                CSV + Folium 출력
├── mclp_config.yaml         파라미터 설정
├── TD-CMCLP_기획서.md       수리모형 상세 문서
├── MCLP_최종보고서.md       실험 결과 보고서
└── webapp/
    ├── main.py              FastAPI 백엔드
    ├── package.json         프론트 의존성
    ├── vite.config.ts       Vite 설정
    ├── tsconfig.json        TypeScript 설정
    ├── index.html           HTML 진입점
    └── frontend_src/
        ├── App.tsx          메인 UI
        ├── styles.css       다크 테마
        └── main.tsx         React 진입점
```

## 웹 관제 시스템 기능

- 12개 파라미터 슬라이더/드롭다운 실시간 제어
- 파라미터 변경 → 500ms 내 재최적화 → 지도 즉시 갱신
- 돌발 사고 시뮬레이션 (지도 클릭 → 인근 거점 비활성화)
- 기상 마스킹 (적설/한파 시 전체 운행 중단)
- 트램 노선 + 정거장 + 공사 복도 + 간선도로 표시
- AMR 배송 경로 (거점→수요) + 탑차 경로 (창고→거점) 시각화
- 대전시 행정경계 표시

## 모듈 간 연결

| 공급 모듈 | 제공 데이터 | MCLP에서의 역할 |
|---|---|---|
| 네트워크 BC (모듈3) | betweenness, lane_ratio | 통합 위험도 스코어 |
| Prophet (모듈1) | y_hat_t30 | XGBoost 피처 (간접) |
| DTW (모듈4) | cluster_mult, rapid_mult | 수요 가중치 + 스크리닝 |
| XGBoost (모듈2) | prob_severe | 고위험 수요 선별 |
