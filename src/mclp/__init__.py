"""
MCLP(Maximum Covering Location Problem) 최적화 모듈
==================================================
대전 트램 공사 구간 AMR-탑차 연계 거점 최적 배치

Modules:
    config.py       - 파라미터 설정 관리
    data_loader.py  - 데이터 로딩/전처리
    distance.py     - 거리 행렬 산출
    weights.py      - 수요 가중치 산출
    constraints.py  - 12대 제약 조건 엔진
    solver.py       - CMCLP ILP 솔버
    output.py       - CSV/Folium 출력
"""

__version__ = "1.0.0"
