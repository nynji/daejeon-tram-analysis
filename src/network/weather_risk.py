"""
weather_risk.py
기상 위험도 가중치 계산 모듈 v2 — 실측 ASOS 데이터 기반

데이터: 대전 ASOS 관측소 (지점 133번, 대전)
파일:   data/daejeon_asos.csv.csv  (cp949 인코딩)
기간:   2024-10-01 ~ 2026-07-14 (약 651일, 15,624시간)

패널티 변환 근거:
  강수-속도 저하 관계 (한국교통연구원, 2019 / HCM 2010):
    강수량 0       mm/h  → 속도영향 0%   → risk 0.00
    강수량 1~4    mm/h  → 속도영향 5%   → risk 0.10
    강수량 5~19   mm/h  → 속도영향 15%  → risk 0.30
    강수량 20~29  mm/h  → 속도영향 25%  → risk 0.55
    강수량 ≥30    mm/h  → 속도영향 35%  → risk 0.80

  기온·적설 기반 결빙/적설 위험:
    기온 0~3°C + 강수: 결빙 위험 → risk +0.20 (도로살빙 발생 임계)
    기온 < 0°C        : 결빙 확정 → risk +0.30
    적설 1~2cm        → risk +0.20
    적설 ≥3cm         → risk +0.40 (AMR 운영 불가 기준)

  최종 weather_risk = min(강수위험 + 결빙위험 + 적설위험, 1.0)
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


ASOS_PATH = os.path.join("data", "daejeon_asos.csv")


# ─────────────────────────────────────────────────────────────
# ASOS 데이터 로드 & 전처리
# ─────────────────────────────────────────────────────────────

def load_asos(path: str = ASOS_PATH) -> pd.DataFrame:
    """
    ASOS 시간 단위 데이터를 로드하고 분석에 필요한 컬럼만 정제한다.

    Returns
    -------
    df : DataFrame
        컬럼: datetime, temp_c, rain_mm, snow_cm, wind_ms
        index: datetime (시간 단위)
    """
    df = pd.read_csv(path, encoding='cp949')
    df = df.rename(columns={
        '일시':       'datetime',
        '기온(°C)':   'temp_c',
        '강수량(mm)':  'rain_mm',
        '적설(cm)':   'snow_cm',
        '풍속(m/s)':  'wind_ms',
    })
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    # 결측치 처리
    df['rain_mm'] = df['rain_mm'].fillna(0.0)
    df['snow_cm'] = df['snow_cm'].fillna(0.0)
    df['temp_c']  = df['temp_c'].interpolate(method='linear').bfill().ffill()
    df['wind_ms'] = df['wind_ms'].fillna(0.0)

    df = df.set_index('datetime')[['temp_c', 'rain_mm', 'snow_cm', 'wind_ms']]
    return df


# ─────────────────────────────────────────────────────────────
# 단일 시간 → 위험 계수 변환
# ─────────────────────────────────────────────────────────────

def _hour_to_risk(rain: float, temp: float, snow: float) -> dict:
    """
    1시간 관측값으로 위험 계수를 산출한다.

    Returns
    -------
    dict: rain_risk, freeze_risk, snow_risk, total_risk, amr_operable, scenario
    """
    # 강수 위험
    if rain >= 30:
        rain_risk = 0.80
        rain_scenario = "집중호우(≥30mm)"
    elif rain >= 20:
        rain_risk = 0.55
        rain_scenario = "강한비(20~29mm)"
    elif rain >= 5:
        rain_risk = 0.30
        rain_scenario = "중간비(5~19mm)"
    elif rain >= 1:
        rain_risk = 0.10
        rain_scenario = "약한비(1~4mm)"
    else:
        rain_risk = 0.00
        rain_scenario = "맑음/흐림"

    # 결빙 위험
    if temp < 0:
        freeze_risk = 0.30
    elif temp <= 3 and rain > 0:
        freeze_risk = 0.20   # 도로 살빙 임계 구간
    else:
        freeze_risk = 0.00

    # 적설 위험
    if snow >= 3:
        snow_risk = 0.40
    elif snow >= 1:
        snow_risk = 0.20
    else:
        snow_risk = 0.00

    total = min(rain_risk + freeze_risk + snow_risk, 1.0)

    # AMR 운영 가능 여부
    amr_operable = not (rain >= 30 or snow >= 3 or temp < -10)

    return {
        'rain_risk':    rain_risk,
        'freeze_risk':  freeze_risk,
        'snow_risk':    snow_risk,
        'total_risk':   total,
        'amr_operable': amr_operable,
        'scenario':     rain_scenario,
    }


# ─────────────────────────────────────────────────────────────
# 월별 실측 집계 (Prophet 외생변수 / 리포트용)
# ─────────────────────────────────────────────────────────────

def compute_monthly_stats(df_asos: pd.DataFrame) -> pd.DataFrame:
    """
    월별 기상 위험 통계를 산출한다.

    Returns
    -------
    DataFrame: month, rain_sum, rain_hours, heavy_rain_hours,
               freeze_hours, snow_hours, avg_risk, max_risk
    """
    records = []
    for dt, row in df_asos.iterrows():
        risk = _hour_to_risk(row['rain_mm'], row['temp_c'], row['snow_cm'])
        records.append({
            'datetime': dt,
            'total_risk': risk['total_risk'],
            'amr_operable': risk['amr_operable'],
        })
    df_risk = pd.DataFrame(records).set_index('datetime')

    combined = df_asos.join(df_risk)
    combined['month'] = combined.index.to_period('M')

    monthly = combined.groupby('month').agg(
        rain_sum        = ('rain_mm',    'sum'),
        rain_hours      = ('rain_mm',    lambda x: (x > 0).sum()),
        heavy_rain_hours= ('rain_mm',    lambda x: (x >= 20).sum()),
        freeze_hours    = ('temp_c',     lambda x: (x <= 0).sum()),
        snow_hours      = ('snow_cm',    lambda x: (x >= 1).sum()),
        avg_risk        = ('total_risk', 'mean'),
        max_risk        = ('total_risk', 'max'),
        amr_inoperable  = ('amr_operable', lambda x: (~x).sum()),
    ).reset_index()
    monthly['month'] = monthly['month'].astype(str)
    return monthly


# ─────────────────────────────────────────────────────────────
# 특정 시점 / 기간 위험 계수 조회
# ─────────────────────────────────────────────────────────────

def get_weather_risk_at(df_asos: pd.DataFrame,
                         target_dt: datetime = None) -> dict:
    """
    특정 시각의 기상 위험 계수를 반환한다.
    target_dt=None이면 데이터의 가장 최근 시각 사용.
    """
    if target_dt is None:
        target_dt = df_asos.index.max()

    # 가장 가까운 관측 시각
    idx = df_asos.index.get_indexer([target_dt], method='nearest')[0]
    row = df_asos.iloc[idx]
    risk = _hour_to_risk(row['rain_mm'], row['temp_c'], row['snow_cm'])
    risk['datetime'] = str(df_asos.index[idx])
    risk['temp_c']   = float(row['temp_c'])
    risk['rain_mm']  = float(row['rain_mm'])
    risk['snow_cm']  = float(row['snow_cm'])
    return risk


def get_period_avg_risk(df_asos: pd.DataFrame,
                         start: str, end: str) -> float:
    """
    특정 기간의 평균 기상 위험 계수를 반환한다.
    Prophet 외생변수 학습 기간 설정용.
    """
    mask = (df_asos.index >= start) & (df_asos.index <= end)
    sub = df_asos[mask]
    risks = [_hour_to_risk(r['rain_mm'], r['temp_c'], r['snow_cm'])['total_risk']
             for _, r in sub.iterrows()]
    return float(np.mean(risks)) if risks else 0.0


def get_monthly_weather_risk(month: int = None,
                              df_asos: pd.DataFrame = None) -> dict:
    """
    월 기준 실측 평균 기상 위험 계수를 반환한다.
    run.py / priority_score.py 인터페이스 호환용.

    Parameters
    ----------
    month : int  (1~12)
    df_asos : DataFrame  (None이면 파일에서 자동 로드)

    Returns
    -------
    dict: month, rain_risk_score, scenario, amr_operable, note, source
    """
    if df_asos is None:
        df_asos = load_asos()

    if month is None:
        month = datetime.now().month

    # 해당 월 전체 행의 평균 위험 계수
    month_rows = df_asos[df_asos.index.month == month]

    if len(month_rows) == 0:
        # 데이터 없으면 보수적 기본값
        return {
            'month': month,
            'rain_risk_score': 0.25,
            'base_weather_penalty': 0.25,
            'weather_speed_multiplier': 0.75,
            'scenario': '데이터 없음(기본값)',
            'amr_operable': True,
            'note': 'ASOS 데이터 해당 월 없음',
            'source': 'fallback',
        }

    risks = [_hour_to_risk(r['rain_mm'], r['temp_c'], r['snow_cm']) for _, r in month_rows.iterrows()]
    avg_risk  = float(np.mean([r['total_risk'] for r in risks]))
    max_risk  = float(np.max([r['total_risk'] for r in risks]))
    heavy_hrs = sum(1 for r in risks if r['rain_risk'] >= 0.55)
    freeze_hrs = sum(1 for r in risks if r['freeze_risk'] > 0)
    inop_hrs  = sum(1 for r in risks if not r['amr_operable'])

    # 대표 시나리오 결정
    if avg_risk >= 0.40:
        scenario = "집중호우/결빙 고위험"
    elif avg_risk >= 0.20:
        scenario = "강수/결빙 중위험"
    elif avg_risk >= 0.05:
        scenario = "약한강수"
    else:
        scenario = "맑음/흐림"

    note = (f"실측 데이터 기반 | "
            f"집중호우 {heavy_hrs}시간 · 결빙위험 {freeze_hrs}시간 · "
            f"AMR불가 {inop_hrs}시간")

    return {
        'month': month,
        'rain_risk_score': round(avg_risk, 4),
        'max_risk': round(max_risk, 4),
        'base_weather_penalty': round(avg_risk, 4),
        'weather_speed_multiplier': round(1.0 - avg_risk, 4),
        'scenario': scenario,
        'amr_operable': inop_hrs == 0,
        'note': note,
        'source': 'ASOS 실측 (대전 133번, 시간자료)',
        'heavy_rain_hours': heavy_hrs,
        'freeze_hours': freeze_hrs,
        'amr_inoperable_hours': inop_hrs,
    }


def get_required_data_note() -> str:
    return """
## 기상 데이터 현황 및 추가 수집 권장 목록

### 현재 보유
- **대전 ASOS 133번 시간자료** (2024-10-01 ~ 2026-07-14, 15,624시간)
  - 기온, 강수량, 적설, 풍속 포함
  - Prophet 외생변수 및 우선순위 스코어 기상 가중치로 직접 활용 가능

### 추가 수집 권장

| 데이터 | 출처 | 활용 |
|--------|------|------|
| 기상특보 발령 이력 | 기상청 API허브 | AMR 운영불가일 J_danger 마스킹 |
| 동네예보 (3일) | 기상청 동네예보 API | 사전 경보, 거점 재배치 알림 |
| 도로결빙 센서 데이터 | 대전시 도로관리원 | 결빙 구간 공간적 특정 |
"""
