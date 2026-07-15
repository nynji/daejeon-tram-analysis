# 모듈 간 인터페이스 스키마 (1주차 중 확정)

각 모듈이 다른 모듈에 결과를 넘길 때 사용할 데이터 포맷을 여기에 정의합니다.
담당자는 실제 구현 전에 이 문서부터 채우고, 다른 담당자는 이 스키마로 mock 데이터를 만들어 병행 개발합니다.

---

## 1. Prophet 출력 (진웅 → 현지, 진웅 → 대시보드)

```json
{
  "segment_id": "S12",
  "timestamp": "2026-07-20T08:30:00",
  "y_hat": 18.4,
  "y_hat_lower": 15.2,
  "y_hat_upper": 21.6
}
```

## 2. 매개중심성 출력 (대흥 → 현지)

```json
{
  "segment_id": "S12",
  "betweenness_centrality": 0.083,
  "updated_at": "2026-07-20T08:25:00"
}
```

## 3. 군집분석 출력 (현서 → 현서 스크리닝 → 대흥 MCLP)

```json
{
  "segment_id": "S12",
  "cluster_type": "공사_후_급격_악화형",
  "weight_incentive": 1.5
}
```
cluster_type 값: `상시_정체형` | `출퇴근_집중형` | `공사_후_급격_악화형` | `영향_미미형`

## 4. XGBoost 출력 (현지 → 현서 스크리닝, 현지 → 대시보드 경보패널)

```json
{
  "segment_id": "S12",
  "timestamp": "2026-07-20T08:30:00",
  "prob_normal": 0.1,
  "prob_caution": 0.3,
  "prob_severe": 0.6,
  "predicted_class": "심각",
  "shap": {
    "차선감소": 0.42,
    "강수": 0.31,
    "매개중심성": 0.27
  }
}
```

## 5. 종합 스크리닝 출력 (현서 → 대흥 MCLP)

```json
{
  "segment_id": "S12",
  "is_high_risk": true,
  "risk_score": 0.78,
  "screening_reason": ["xgboost_severe_prob>0.6", "cluster=공사_후_급격_악화형"]
}
```

## 6. MCLP 출력 (대흥 → 대시보드 MCLP 레이어)

```json
{
  "anchor_id": "A03",
  "lat": 36.3504,
  "lng": 127.3845,
  "covered_demand_points": ["D001", "D002", "D010"],
  "coverage_radius_m": 300,
  "reason": {
    "demand_weight": 120,
    "risk_score": 0.78,
    "cluster_type": "공사_후_급격_악화형"
  }
}
```

## 7. 우회경로 출력 (진웅 → 대시보드 경로탐색 UI)

```json
{
  "origin": "S05",
  "destination": "S22",
  "departure_time": "2026-07-20T08:00:00",
  "route": ["S05", "S06", "S09", "S22"],
  "estimated_duration_min": 24,
  "avoided_bottlenecks": ["S12"]
}
```

---

## TODO (1주차 중 각 담당자가 채울 것)
- [ ] 실제 segment_id 목록 및 명명 규칙 확정 (대흥 - 구간/공구 매핑 테이블 기준)
- [ ] 각 모듈 담당자가 위 스키마 검토 후 확정 서명
- [ ] mock 데이터셋 `data/mock/`에 각자 위 포맷대로 샘플 생성
