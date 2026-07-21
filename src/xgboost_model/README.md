# XGBoost 혼잡 예측 — 모델링 파트

대전 트램 공사구간 교통혼잡 예측 프로젝트의 XGBoost 모델링 파트 전체(피처 엔지니어링 → 데이터 조립 → 학습/튜닝 → SHAP 해석 → 문서화)를 담은 폴더. 팀 공유 저장소에 이 폴더 단위로 업로드한다.

## 폴더 구조

```
xgb/
├── README.md          이 문서 (시작점)
├── docs/               보고서용 설계 문서
│   ├── xgb.md           모델링 전체 설계·성능·해석·한계 (가장 먼저 읽을 문서)
│   └── feature.md       22개 피처 각각의 정의·산출 방식·EDA 인사이트
├── notebooks/          실행 코드 (Jupyter, 위에서 아래로 순서대로 실행)
│   ├── build_speed_features.ipynb      실측 속도 rolling 피처
│   ├── bottleneck_train_only.ipynb     is_bottleneck_slot Train 전용 재계산(leakage 제거)
│   ├── network_features.ipynb          매개중심성·도로등급·차로수
│   ├── weather_features.ipynb          기상 피처
│   ├── incident_flag.ipynb             돌발상황 피처
│   ├── prophet_features.ipynb          Prophet 예측치 조립(72/90 구간)
│   ├── xgb_feature_matrix.ipynb        최종 매트릭스 조립 v1 (exact join, 참고용)
│   ├── xgb_feature_matrix_ver2.ipynb   v2 (정규 10분 격자 + as-of join)
│   ├── xgb_feature_matrix_ver3.ipynb   v3 (v2 + is_bottleneck_slot Train 전용, 최종 확정본)
│   ├── xgb_model.ipynb                 Model B(최종 채택) 학습/튜닝/평가/SHAP
│   └── xgb_model_modelA.ipynb          Model A(segment_key 포함) 비교 실험
└── model/               학습된 모델과 실행에 필요한 참조 데이터 (인수인계용)
    ├── README.md          "이미 학습된 모델을 어떻게 쓰는가"에 집중한 사용 가이드
    ├── v3_modelB_alpha_threshold_tuned.json                    학습된 모델
    ├── v3_modelB_alpha_threshold_tuned_metrics.json             3-class 성능
    ├── v3_modelB_alpha_threshold_tuned_binary_risk_metrics.json 이진 집계 성능
    ├── v3_modelB_alpha_threshold_tuned_shap_importance.csv      SHAP 중요도
    └── reference_tables/   실시간 추론 시 join에 쓰는 정적/일 단위 참조 테이블
```

## 어디서부터 봐야 하는가

- **모델링 전체를 이해하려면**: `docs/xgb.md`부터. 설계 결정(α×threshold 튜닝, segment_key 제외 이유 등)과 최종 성능, SHAP 해석, 한계까지 다 정리되어 있다.
- **피처 하나하나의 근거가 궁금하면**: `docs/feature.md`.
- **모델을 그대로 가져다 쓰려면(대시보드 연동 등)**: `model/README.md`만 보면 된다 — 로드 코드, 입력 스펙, SHAP 활용법, 실시간 서빙 시 필요한 것까지 정리되어 있다. 학습 원리를 몰라도 이 문서만으로 충분하다.
- **학습을 재현하거나 이어서 실험하려면**: `notebooks/`를 순서대로 실행. `xgb_feature_matrix_ver3.ipynb`까지 돌려 `xgb_feature_matrix_v3.parquet`를 만든 뒤, `xgb_model.ipynb`(Model B) 또는 `xgb_model_modelA.ipynb`(Model A)를 실행한다.

## 여기 포함되지 않은 것 (용량 문제)

학습에 쓴 원본 피처 매트릭스(`xgb_feature_matrix_v3.parquet`, 445MB)와 중간 산출물(`speed_features.parquet` 344MB, `prophet_features.parquet` 185MB 등)은 GitHub에 올리기엔 너무 커서 이 폴더에 포함하지 않았다. 노트북을 재실행하면 로컬에서 다시 생성된다. `model/reference_tables/`에는 실시간 추론에 필요한 **작은** 참조 테이블만 별도로 포함해뒀다.

## 핵심 결과 요약

| | Precision | Recall | F1 |
|---|---|---|---|
| 정상 | 0.93 | 0.93 | 0.93 |
| 주의 | 0.49 | 0.38 | 0.43 |
| 심각 | 0.45 | 0.72 | 0.55 |

Test 기준 accuracy 0.843, macro F1 0.637. 상세 근거·해석은 `docs/xgb.md` 참고.
