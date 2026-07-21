# XGBoost 혼잡 예측 모델 — 실행/서빙용 인수인계 패키지

대전 트램 공사구간 교통혼잡 예측 XGBoost 모델을 다른 사람(대시보드 개발자 등)이 바로 로드해서 쓸 수 있도록 필요한 파일과 지식을 모은 폴더. 학습 과정 전체는 `../docs/xgb.md`(모델링 상세)와 `../docs/feature.md`(피처별 정의), 실제 학습 코드는 `../notebooks/xgb_model.ipynb`를 참고할 것 — 이 문서는 "이미 학습된 모델을 어떻게 쓰는가"에 집중한다.

---

## 1. 이 폴더에 들어있는 것

```
xgb/model/
├── README.md                                              이 문서
├── v3_modelB_alpha_threshold_tuned.json                   학습된 모델 (XGBoost 네이티브 포맷)
├── v3_modelB_alpha_threshold_tuned_metrics.json            Train/Val/Test 3-class 성능 지표
├── v3_modelB_alpha_threshold_tuned_binary_risk_metrics.json  "정상 vs 위험" 이진 집계 성능
├── v3_modelB_alpha_threshold_tuned_shap_importance.csv     클래스별 피처 중요도(mean|SHAP|)
└── reference_tables/
    ├── network_features.parquet             구간별 매개중심성·도로등급·차로수 (정적)
    ├── construction_lane_ratio_daily.parquet 구간x일자별 공사 차로 통제 비율
    └── is_bottleneck_slot_TRAIN_ONLY.csv     구간x시간대별 병목 여부 lookup
```

(상위 `xgb/` 폴더 구조 전체는 `../../README.md` 참고)

**여기 포함되지 않은 것**: 학습에 쓴 원본 8,068,423행짜리 피처 매트릭스(`output/features/xgb_feature_matrix_v3.parquet`, 445MB)나 실측 속도 원본(`output/features/speed_features.parquet`, 344MB), Prophet 예측 원본(`output/features/prophet_features.parquet`, 185MB) 등은 용량이 커서 이 폴더에 복사하지 않았다. 이 파일들은 "모델을 재학습/재현"할 때만 필요하고, "이미 학습된 모델로 새 데이터를 예측"할 때는 필요 없다(대신 실시간 서빙에서는 이 데이터들을 만든 것과 **같은 로직**으로 매번 새로 피처를 조립해야 하며, 그 로직은 4절에 정리했다).

---

## 2. 모델 로드 & 예측 코드

```python
import xgboost as xgb
import numpy as np

model = xgb.XGBClassifier()
model.load_model("v3_modelB_alpha_threshold_tuned.json")

BEST_THRESHOLD = 0.25       # 아래 3절 참고 — 반드시 같이 적용해야 함
SEVERE_CLASS_IDX = 2        # 클래스 순서: 0=정상, 1=주의, 2=심각

def predict_with_threshold(proba: np.ndarray, threshold_severe: float = BEST_THRESHOLD) -> np.ndarray:
    """모델 raw output(argmax)이 아니라 이 규칙을 반드시 적용해야
    검증된 성능(심각 Recall 0.72 등)이 재현된다."""
    return np.where(
        proba[:, SEVERE_CLASS_IDX] >= threshold_severe,
        SEVERE_CLASS_IDX,
        np.argmax(proba[:, :2], axis=1),
    )

proba = model.predict_proba(X_new)      # X_new: (n, 22) — 3절 피처 스펙 준수
pred = predict_with_threshold(proba)     # 최종 0/1/2 클래스
prob_risk = proba[:, 1] + proba[:, 2]    # "병목 여부"(정상 vs 위험) 이진 확률
```

**라이브러리 버전**: 학습에 쓴 `xgboost` 버전을 맞춰서 로드할 것을 권장한다(네이티브 JSON이 메이저 버전 간 100% 호환 보장은 아님). 학습 환경 버전은 원본 저장소에서 `pip show xgboost`로 확인 가능.

---

## 3. 입력 피처 스펙 (22개, 이름·순서·전처리 정확히 일치해야 함)

| 그룹 | 피처명 | dtype | 비고 |
|---|---|---|---|
| 실측 속도 | `V_segment` | float | 현재 속도 |
| | `speed_last_10min` | float | |
| | `speed_ma_30min` | float | 최근 30분 이동평균 |
| | `speed_ma_1h` | float | 최근 1시간 이동평균 |
| | `speed_change_rate` | float | |
| 시간 | `hour` | int | 0~23 |
| | `dow` | int | 요일 |
| | `is_weekend` | int8(0/1) | |
| | `is_bottleneck_slot` | int8(0/1) | `reference_tables/is_bottleneck_slot_TRAIN_ONLY.csv`에서 (구간, 시간대) join |
| 네트워크 | `betweenness_pre` | float | `reference_tables/network_features.parquet`에서 join |
| | `betweenness_during` | float | 〃 |
| | `road_rank` | int/float | 〃 |
| | `lanes` | int/float | 〃 |
| 공사 | `lane_remain_ratio` | float | `reference_tables/construction_lane_ratio_daily.parquet`에서 (구간, 날짜) join, 0~1 |
| 돌발 | `incident_flag` | int8(0/1) | 실시간 돌발정보 필요(6절 참고) |
| | `incident_count` | int | 〃 |
| 기상 | `precipitation_mm` | float | 실시간 기상 API 필요, 결측 허용(NaN) |
| | `is_weather_alert` | int8(0/1) | 강수≥30mm/h 또는 적설≥3cm |
| | `is_freezing` | int8(0/1) | 기온<0℃ |
| Prophet | `y_hat_t30` | float | 구간별 Prophet 모델 예측값, 결측 허용(18/90 구간 미제공) |
| | `y_hat_lower_t30` | float | 결측 허용 |
| | `y_hat_upper_t30` | float | 결측 허용 |

**주의사항**:
- `segment_key`, `timestamp`, `target_speed`, `label` 등은 입력 피처에서 **반드시 제외**해야 한다(학습 시 X에서 제외됨 — `segment_key`를 넣으면 다른 모델이 됨).
- bool 성격 컬럼(`is_weekend`, `is_bottleneck_slot`, `incident_flag`, `is_weather_alert`, `is_freezing`)은 원본 조인 과정에서 결측이 섞이면 pandas가 `object` dtype으로 읽는 경우가 있다. 반드시 아래처럼 캐스팅:
  ```python
  df[col] = df[col].astype("boolean").fillna(False).astype("int8")
  ```
- Prophet/기상 컬럼의 NaN은 채우지 말고 그대로 둘 것 — XGBoost가 결측을 자체적으로 분기 처리하도록 학습됐다.

---

## 4. α와 threshold가 무엇인지

- **α (클래스 가중치 지수, 이미 모델에 반영됨)**: 학습 시 `weight_class = balanced_weight_class ** α`로 클래스 불균형(정상 90.7%)을 보정한 정도. `α=0.3`으로 학습됨. **재학습해야만 바뀐다** — 이 폴더의 모델은 이미 이 값으로 고정된 결과물이라 신경 쓸 필요 없음.
- **threshold=0.25 (추론 시 반드시 적용해야 함)**: `predict_proba`가 뱉는 심각확률이 0.25 이상이면 심각으로 판정하는 규칙. 모델 raw argmax를 쓰면 심각 클래스를 거의 못 잡아낸다(원래 희소 클래스라서). **`predict_with_threshold()` 함수를 빼먹으면 검증된 성능이 재현되지 않는다** — 가장 흔히 발생할 실수 포인트.

---

## 5. 검증된 성능 (참고용, 대시보드 문구에 인용 가능)

**3-class (정상/주의/심각) — Test 기준**

| | Precision | Recall | F1 |
|---|---|---|---|
| 정상 | 0.93 | 0.93 | 0.93 |
| 주의 | 0.49 | 0.38 | 0.43 |
| 심각 | 0.45 | 0.72 | 0.55 |

accuracy 0.843, macro F1 0.637, log loss 0.346.

**이진 집계 (정상 vs 위험=주의+심각) — Test 기준**: Precision 0.691, Recall 0.702, F1 0.697.

**한계**: 주의 클래스 Recall(0.38)이 심각 Recall(0.72)보다 낮다 — 정상↔주의 경계는 별도로 튜닝되지 않고 모델 raw argmax에 맡겨져 있기 때문. 상세 원인·해석은 `../docs/xgb.md` 5.3~5.4절 참고.

---

## 6. SHAP 활용 (대시보드 "왜 이 등급인가" 설명)

`v3_modelB_alpha_threshold_tuned_shap_importance.csv`는 **Global SHAP**(Val/Test 샘플 평균 기여도)로, "이 모델은 전반적으로 무엇을 중요하게 보는가"를 보여준다. 모델 설명·보고서용이며 이미 계산돼 이 폴더에 포함되어 있다.

대시보드가 "지금 이 구간이 왜 이 등급(주의/심각)으로 예측됐는가"를 보여주려면 이것과 다른 **Local SHAP**(개별 구간에 대한 실시간 계산)이 필요하다 — Global SHAP 표를 그대로 대시보드에 띄우면 안 된다.

**핵심 주의점 — class_idx를 고정하면 안 됨**: 이 모델은 3-class라 SHAP도 클래스별로 3개(정상/주의/심각) 나온다. class=2(심각)로 고정해서 계산하면, 주의로 예측된 구간에도 "왜 심각인가"라는 엉뚱한 설명이 붙는다. **반드시 그 구간의 실제 예측 클래스(`pred`)를 넘겨야 한다.**

```python
import shap
import numpy as np

CLASS_NAMES = ["정상", "주의", "심각"]

def get_top_shap_features(shap_values, row_idx, class_idx, feature_cols, k=3):
    contributions = shap_values.values[row_idx, :, class_idx]
    order = np.argsort(-np.abs(contributions))[:k]
    return [
        {"feature": feature_cols[i], "impact": round(float(contributions[i]), 4)}
        for i in order
    ]

proba = model.predict_proba(X_current)      # X_current: 이번 추론 대상 구간들
pred = predict_with_threshold(proba)         # 구간별 예측 클래스(0/1/2)

alert_idx = np.where(pred != 0)[0]           # 주의/심각으로 예측된 구간만 SHAP 계산(정상은 설명 우선순위 낮음)
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_current.iloc[alert_idx])

results = []
for i, row_idx in enumerate(alert_idx):
    top3 = get_top_shap_features(shap_values, i, class_idx=pred[row_idx], feature_cols=feature_cols, k=3)
    results.append({
        "segment": segment_keys[row_idx],
        "predicted_class": CLASS_NAMES[pred[row_idx]],
        "prob_normal": float(proba[row_idx, 0]),
        "prob_caution": float(proba[row_idx, 1]),
        "prob_severe": float(proba[row_idx, 2]),
        "reason": top3,
    })
```

**계산량**: 90개 전체 구간이 아니라 그 순간 주의/심각으로 예측된 구간만 대상이라, 실무 규모(예: 경보 구간 40개 × 6시간 × 5분 단위 ≈ 2,880 샘플)에서는 `TreeExplainer`로 실시간 처리에 충분하다.

**표현 주의**: SHAP 값(`impact`)은 log-odds 스케일 기여도이지 "비율(%)"이 아니다. "차선감소 42%, 강수 31%"처럼 퍼센트로 표현하면 부정확하다 — `impact: 0.31`처럼 값 그대로 표기하거나 순위만 보여줄 것.

상세 논의는 `../docs/xgb.md` 6.3절·8.3절 참고.

---

## 7. 실시간 서빙 시 추가로 필요한 것 (중요 — 모델 파일만으로는 서빙 불가)

이 모델은 22개 피처가 **이미 갖춰진 벡터**를 받아야 예측할 수 있다. 그 피처를 실시간으로 조립하는 파이프라인은 이 폴더에 포함되어 있지 않다. 대시보드팀에 반드시 같이 전달해야 하는 것:

| 피처군 | 갱신 주기 | 필요한 것 |
|---|---|---|
| `speed_*` | 10분 | ITS 실시간 속도 피드 + rolling 계산 로직 |
| `is_bottleneck_slot` | 정적 | `reference_tables/is_bottleneck_slot_TRAIN_ONLY.csv` (이미 포함) |
| `betweenness_*`, `road_rank`, `lanes` | 정적 | `reference_tables/network_features.parquet` (이미 포함) |
| `lane_remain_ratio` | 일 단위 | `reference_tables/construction_lane_ratio_daily.parquet`(이미 포함, 예시 스냅샷) + 매일 갱신하는 공사 현황판 파싱 로직(원본 저장소 `../docs/xgb.md` 2.4절) |
| `incident_flag/count` | 실시간 | 국토교통부 돌발정보 API 연동 필요(미구현) |
| `precipitation_mm`, `is_weather_alert`, `is_freezing` | 실시간 | 기상청 API 연동 필요(미구현) |
| `y_hat_t30` 등 Prophet | 10분 | 구간별 Prophet 모델 90개(현재 72개만 존재) 실시간 재예측 필요(미구현) |

**핵심 결론**: 모델 자체의 추론 속도는 실시간 서빙에 충분하지만, 위 표의 "미구현" 항목들(돌발/기상 API 연동, Prophet 실시간 재예측)은 별도로 구축해야 한다. 상세 아키텍처 설계안은 `../docs/xgb.md` 8.2절("실시간 서빙 아키텍처") 참고.

---

## 8. 더 자세한 내용이 필요하면

원본 저장소(`물류/`) 기준:
- `../docs/xgb.md` — 모델링 설계 전체(피처 선정 이유, α/threshold 탐색 방법론, 성능 상세, SHAP 해석, 한계, 실시간 서빙 아키텍처)
- `../docs/feature.md` — 22개 피처 각각이 어떻게 만들어졌는지(원본 데이터, 전처리, 시행착오)
- `../notebooks/xgb_model.ipynb` — 학습/평가/SHAP 전체 실행 코드(재학습·비교실험 시 참고)
