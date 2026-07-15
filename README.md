# 트램 공사 구간 시공간 교통혼잡 예측 및 물류 대응 통합 관제 대시보드

대전 도시철도 2호선(트램) 공사 구간 교통혼잡 예측 + AMR 거점 최적화 + 통합 관제 대시보드

## 

## 폴더 구조

```
repo/
├── data/               # 원본 데이터 (gitignore, 로컬/드라이브 별도 관리)
├── notebooks/          # 실험용 분석 노트북
├── src/
│   ├── forecasting/    # 진웅 - Prophet 모델링
│   ├── network/        # 대흥 - 매개중심성, MCLP
│   ├── clustering/     # 현서 - 군집분석, 스크리닝
│   └── xgboost\_model/  # 현지 - XGBoost, SHAP
├── dashboard/          # Streamlit + Folium 통합 대시보드
├── outputs/            # 모델 산출물, 캐시 (gitignore)
└── README.md
```

## 브랜치 전략

* `main` : 항상 실행되는 안정 버전만 (발표/제출용)
* `dev` : 통합 개발 브랜치, 모든 PR은 여기로
* `feature/작업명` : 개인 작업 브랜치 (예: `feature/xgboost-shap`, `feature/dashboard-alert-panel`)

## 개발 환경 세팅

```bash
git clone <repo-url>
cd repo
python -m venv venv
source venv/bin/activate   # Windows: venv\\Scripts\\activate
pip install -r requirements.txt
```

