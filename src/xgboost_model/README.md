# [모듈] XGBoost 혼잡 임계도달 분류 모델

**"개별 구간이 30분 후 대전시 정책 임계(20km/h, 15km/h) 아래로 떨어질지"**를 예측하는 구간 단위 조기경보 모델. 90개 구간(45구간×2방향)을 하나로 풀링한 단일 3-class 모델이며, 관제 대시보드 병목경보와 MCLP 고위험구간 스크리닝에 결과를 공급한다.

## 실행 방법

```bash
pip install xgboost shap scikit-learn polars pandas matplotlib
```

```
notebooks/ 안에서 순서대로 실행:
build_speed_features.ipynb → bottleneck_train_only.ipynb → network_features.ipynb
→ weather_features.ipynb → incident_flag.ipynb → prophet_features.ipynb
→ xgb_feature_matrix_ver3.ipynb   (최종 피처 매트릭스 조립, 약 8,068,423행)
→ xgb_model.ipynb                  (Model B 학습+튜닝+SHAP, 783만행 × 4α 기준 약 30분+)
```

## 핵심 산출물

| 파일 | 내용 | 용도 |
|---|---|---|
| `model/v3_modelB_alpha_threshold_tuned.json` | 학습된 XGBoost 모델(네이티브 JSON) | 추론 |
| `model/v3_modelB_alpha_threshold_tuned_metrics.json` | Train/Val/Test 3-class 성능 | 검증 |
| `model/v3_modelB_alpha_threshold_tuned_shap_importance.csv` | 클래스별 Global SHAP 중요도 | 보고서/대시보드 |
| `model/reference_tables/` | 실시간 추론 시 join용 정적 참조 테이블 3종 | 서빙 |
| `docs/xgb.md` | 설계·성능·SHAP·한계 전체 문서 | 보고서 |
| `docs/feature.md` | 22개 피처 각각의 산출 근거 | 보고서 |

## 분석 방법론

### 1. 데이터 grain(행 단위) 문제와 해결

원본 5분 ITS 데이터는 관측 시각이 불균일해, 실제 존재하는 시각만 쓰면(v1, exact join) 라벨 매칭 실패율 22.1%가 발생. **90개 구간 × 10분 정규 격자를 먼저 고정하고 as-of join**(v2)으로 전환해 2.2%까지 감소시켰다(v3는 여기에 `is_bottleneck_slot` leakage 제거까지 반영한 최종 확정본).

| 지표 | v1(exact join) | v2/v3(정규격자+as-of) |
|---|---|---|
| 라벨 매칭 실패 | 22.1% | 2.2% |
| Prophet 결측 | 92.2% | 20.4%(18/90 구간 미제공분과 거의 일치) |

### 2. 클래스 가중치(α) × 확률 임계값(threshold) 중첩 탐색

클래스 불균형(정상 90.7%/주의 7.2%/심각 2.1%) 보정을 sklearn `balanced`(15.8배) 그대로 쓰지 않고, `weight_class = balanced_weight_class ** α`로 지수 완화. α가 바뀌면 확률분포 자체가 달라져 최적 threshold도 바뀌므로 독립 탐색 대신 **중첩 grid search**로 동시 탐색.

| 파라미터 | 탐색 범위 | 선택 기준 |
|---|---|---|
| α | {0.3, 0.5, 0.7, 1.0} | Val 심각 Precision≥0.45를 만족하며 Recall 최대 |
| threshold | 0.05~0.50 | 재학습 없이 Val `predict_proba` 스윕(공짜) |

**채택: α=0.3, threshold=0.25**

### 3. SHAP 기반 해석 (Global + Local)

Test 클래스 층화추출(12,971건)로 `TreeExplainer` 계산. 대시보드용 "왜 이 구간이 위험한가" 설명은 Global(정적 저장)이 아니라 **Local SHAP**(구간별 실제 예측 클래스 기준 실시간 계산)이 맞으며, class를 고정하지 않고 `pred[row_idx]`를 그대로 넘기는 게 핵심(`docs/xgb.md` 6.3·8.3절).

## 핵심 결과

**3-class — Test 기준**

| | Precision | Recall | F1 |
|---|---|---|---|
| 정상 | 0.93 | 0.93 | 0.93 |
| 주의 | 0.49 | 0.38 | 0.43 |
| 심각 | 0.45 | 0.72 | 0.55 |

accuracy 0.843, macro F1 0.637, log loss 0.346.

**이진 집계(정상 vs 위험=주의+심각) — Test 기준**: Precision 0.691, Recall 0.702, F1 0.697

**튜닝 전(α=1) 대비**: Accuracy 0.727→0.843, 심각 Precision 0.381→0.451(Recall 0.748→0.720, 소폭 트레이드오프)

**SHAP 상위 피처(3클래스 평균)**: `speed_ma_30min`(0.624) > `speed_ma_1h`(0.241) > `y_hat_lower_t30`(0.238) > `y_hat_t30`(0.227) > `hour`(0.212). 심각 클래스에서는 `lane_remain_ratio`(0.327)가 3위로 상승 — 공사 강도가 주의→심각 전이의 결정 요인.

## 소스 코드

| 파일 | 역할 |
|---|---|
| `build_speed_features.ipynb` | 실측 속도 rolling 피처(4종) |
| `bottleneck_train_only.ipynb` | `is_bottleneck_slot` Train 전용 재계산(leakage 제거) |
| `network_features.ipynb` | 매개중심성·도로등급·차로수 재집계 |
| `weather_features.ipynb` | 기상 피처(문헌 기준 임계값) |
| `incident_flag.ipynb` | 돌발상황 10분 explode |
| `prophet_features.ipynb` | Prophet 팀 산출물 조립(72/90 구간) |
| `xgb_feature_matrix_ver2.ipynb` / `_ver3.ipynb` | 최종 매트릭스 조립(as-of join, v2→v3) |
| `xgb_model.ipynb` | Model B 학습/튜닝/평가/SHAP(최종 채택) |
| `xgb_model_modelA.ipynb` | Model A: `segment_key` 포함 비교 실험 |

## 데이터

| 데이터 | 규모 |
|---|---|
| 최종 피처 매트릭스(v3) | 8,068,423행 × 28컬럼, 90구간 × 10분 격자 |
| 기간 | 2024-10-01 ~ 2026-07-01(약 639일) |
| Train/Val/Test | 7,836,853 / 179,910 / 51,660 |
| 입력 피처 | 22개(`segment_key` 제외, Model B 기준) |
| SHAP 샘플 | Test 클래스 층화추출 12,971건 |

## 현재 한계

| 항목 | 상태 |
|---|---|
| Prophet 결측 | 18/90 구간 미제공(성능 미달 보류), 결측률 20.37% |
| Model A 비교 | 코드는 준비됐으나 재실행 미완료(`xgb_model_modelA.ipynb`) |
| 주의(class=1) Recall | 0.38로 심각 Recall(0.72)보다 낮음 — 정상↔주의 경계는 별도 튜닝 대상이 아니었음 |
| 본격 하이퍼파라미터 튜닝 | `max_depth`/`learning_rate` 등 baseline 고정값, Optuna 미실행 |
| 기상 공간 해상도 | 대전 단일 ASOS 관측소, 90구간 전부 동일값 |
| 실시간 서빙 | 모델 추론은 실시간 가능하나 피처 파이프라인은 배치 전용(구축 필요, `docs/xgb.md` 8.2절) |
