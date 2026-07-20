"""
its_speed.py
ITS 실측 링크 속도 데이터 처리 모듈

데이터: data/ITS_대전_링크속도.csv
  - CREATDE: 날짜 (20260707)
  - CREATHM: 시간 (0, 5, 10, ... 2355 → 5분 단위)
  - LINKID: 링크 ID (대전 표준노드링크와 동일 체계)
  - PASNGSPED: 통행속도 (km/h)

역할:
  1. BC 계산 시 MAX_SPD 대신 실측 속도로 엣지 가중치 산출
  2. 시간대별 속도 프로파일 생성 (Prophet 종속변수)
  3. 공사 패널티 검증 (실측속도 vs MAX_SPD×패널티)
"""

import os
import pandas as pd
import numpy as np


ITS_PATH = os.path.join("data", "ITS_대전_링크속도.csv")


def load_its_speed(path: str = ITS_PATH) -> pd.DataFrame:
    """
    ITS 실측 속도 데이터를 로드한다.

    Returns
    -------
    df : DataFrame
        컬럼: LINKID(int), hour(int 0~23), speed_mean, speed_min, speed_max
        링크별 시간대 집계 (5분 → 1시간 평균)
    """
    print("[ITS] Loading ITS speed data ...")
    df = pd.read_csv(path, encoding='utf-8-sig')
    df['LINKID'] = df['LINKID'].astype(int)

    # CREATHM → hour 변환 (0→0시, 100→1시, ..., 2355→23시)
    df['hour'] = df['CREATHM'] // 100

    # 시간대별 링크 평균속도
    hourly = df.groupby(['LINKID', 'hour']).agg(
        speed_mean=('PASNGSPED', 'mean'),
        speed_min=('PASNGSPED', 'min'),
        speed_max=('PASNGSPED', 'max'),
        obs_count=('PASNGSPED', 'count'),
    ).reset_index()

    print(f"  원본: {len(df):,}행 → 시간대 집계: {len(hourly):,}행")
    print(f"  링크 수: {hourly['LINKID'].nunique():,}개")
    print(f"  시간대: {hourly['hour'].min()}~{hourly['hour'].max()}")
    return hourly


def get_link_speed_dict(df_hourly: pd.DataFrame, target_hour: int = None) -> dict:
    """
    특정 시간대의 링크별 실측 평균 속도를 딕셔너리로 반환한다.

    Parameters
    ----------
    df_hourly : DataFrame (load_its_speed 결과)
    target_hour : int (0~23). None이면 전 시간대 평균.

    Returns
    -------
    dict: {LINKID: speed_km/h}
    """
    if target_hour is not None:
        sub = df_hourly[df_hourly['hour'] == target_hour]
    else:
        sub = df_hourly.groupby('LINKID').agg(speed_mean=('speed_mean', 'mean')).reset_index()

    return dict(zip(sub['LINKID'].astype(int), sub['speed_mean']))


def get_daily_avg_speed(df_hourly: pd.DataFrame) -> dict:
    """일평균 실측 속도 딕셔너리"""
    avg = df_hourly.groupby('LINKID')['speed_mean'].mean()
    return avg.to_dict()


def compute_speed_profile_stats(df_hourly: pd.DataFrame) -> pd.DataFrame:
    """
    링크별 24시간 속도 프로파일 통계. 군집분석(DTW) 입력용.

    Returns
    -------
    DataFrame: LINKID, speed_mean_daily, speed_std, peak_hour, trough_hour,
               speed_at_peak, speed_at_trough, speed_ratio (trough/peak)
    """
    stats = df_hourly.groupby('LINKID').agg(
        speed_mean_daily=('speed_mean', 'mean'),
        speed_std=('speed_mean', 'std'),
        speed_min_all=('speed_min', 'min'),
        speed_max_all=('speed_max', 'max'),
    ).reset_index()

    # 최고/최저 시간대
    peak_hours = df_hourly.loc[df_hourly.groupby('LINKID')['speed_mean'].idxmax()][['LINKID','hour','speed_mean']]
    peak_hours = peak_hours.rename(columns={'hour': 'peak_hour', 'speed_mean': 'speed_at_peak'})

    trough_hours = df_hourly.loc[df_hourly.groupby('LINKID')['speed_mean'].idxmin()][['LINKID','hour','speed_mean']]
    trough_hours = trough_hours.rename(columns={'hour': 'trough_hour', 'speed_mean': 'speed_at_trough'})

    stats = stats.merge(peak_hours, on='LINKID', how='left')
    stats = stats.merge(trough_hours, on='LINKID', how='left')
    stats['speed_ratio'] = stats['speed_at_trough'] / stats['speed_at_peak'].clip(lower=1)

    return stats
