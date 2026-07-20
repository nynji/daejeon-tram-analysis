"""
priority_score.py
구간 우선순위 통합 스코어링 모듈

역할:
  네트워크 분석(BC), 공사 패널티(lane_remain_ratio),
  기상 위험도, 도로 위계를 결합하여
  대시보드에 표출할 '위험 우선순위 스코어'를 산출한다.

스코어 설계 원칙:
  - 0~100 정규화 (관제 담당자가 직관적으로 읽을 수 있도록)
  - 구성 요소별 가중치는 AHP 유사 방식으로 설정
    · BC(파급력): 40%  — "막히면 얼마나 퍼지는가"
    · 차로 감소:  30%  — "직접 용량 손실"
    · 도로 위계:  20%  — "간선도로일수록 파급 크다"
    · 기상 위험:  10%  — "기상 악화 시 가중"
  - 정거장 구간(segment_id) 단위로 집계 → Prophet/군집분석 단위와 일치

위험 등급 기준:
  80 이상 → 🔴 심각 (즉시 대응)
  60~79  → 🟠 경고 (사전 준비)
  40~59  → 🟡 주의 (모니터링)
  40 미만 → 🟢 정상
"""

import pandas as pd
import numpy as np


# ─────────────────────────────────────────
# 도로 위계 가중치 (ROAD_RANK 코드 기준)
# ITS 표준노드링크 도로 위계 코드:
#   101: 고속국도  102: 일반국도  103: 특별광역시도
#   104: 지방도    105: 시군구도  106: 고속도로  107: 기타
# ─────────────────────────────────────────
ROAD_RANK_WEIGHT = {
    '101': 1.00,  # 고속국도 — 최고 위계
    '102': 0.90,  # 일반국도
    '106': 0.95,  # 고속도로 (천변도시고속도로 포함)
    '103': 0.85,  # 특별광역시도 (계룡로, 대전로 등 광역 간선)
    '104': 0.60,  # 지방도 (일반 시내 도로)
    '105': 0.40,  # 시군구도
    '107': 0.30,  # 기타
}


def compute_priority_scores(
    df: pd.DataFrame,
    weather_risk: float = 0.20,
    weight_bc: float = 0.40,
    weight_lane: float = 0.30,
    weight_rank: float = 0.20,
    weight_weather: float = 0.10
) -> pd.DataFrame:
    """
    링크 단위 우선순위 스코어를 산출하고 정거장 구간 단위로 집계한다.

    Parameters
    ----------
    df : DataFrame
        outputs/network_betweenness.csv 내용
    weather_risk : float
        기상 위험 계수 (0~1). weather_risk.py에서 공급.
    weight_* : float
        각 요소의 가중치. 합계 = 1.0

    Returns
    -------
    df_segment : DataFrame
        segment_id 단위 집계 결과 + 우선순위 스코어 + 등급
    df_link : DataFrame
        링크 단위 스코어 (세부 드릴다운용)
    """
    df = df.copy()

    # ── 1. 각 요소 0~1 정규화 ──────────────────────────────
    # BC (공사 중) — 99퍼센타일 기준 상한 클리핑 후 정규화
    bc_max = df['bc_under_construction'].quantile(0.99)
    df['bc_norm'] = (df['bc_under_construction'] / bc_max).clip(0, 1)

    # BC 변화량 — 양수만 의미있음 (음수는 0 처리)
    bc_chg_max = df['bc_change'].clip(lower=0).quantile(0.99)
    df['bc_change_norm'] = (df['bc_change'].clip(lower=0) / bc_chg_max).clip(0, 1) if bc_chg_max > 0 else 0

    # 최종 BC 점수: 절대값(70%) + 변화량(30%) 혼합
    df['bc_score'] = 0.70 * df['bc_norm'] + 0.30 * df['bc_change_norm']

    # 차로 감소 점수: 잔여 비율의 역수 (잔여가 적을수록 위험)
    df['lane_score'] = 1.0 - df['lane_remain_ratio'].clip(0, 1)

    # 도로 위계 점수
    df['ROAD_RANK_str'] = df['ROAD_RANK'].astype(str).str.strip()
    df['rank_score'] = df['ROAD_RANK_str'].map(ROAD_RANK_WEIGHT).fillna(0.30)

    # 기상 위험 (전 링크 동일 적용)
    df['weather_score'] = float(weather_risk)

    # ── 2. 가중합 스코어 (0~1) ─────────────────────────────
    df['raw_score'] = (
        weight_bc      * df['bc_score']      +
        weight_lane    * df['lane_score']     +
        weight_rank    * df['rank_score']     +
        weight_weather * df['weather_score']
    )

    # 0~100 스케일
    df['priority_score'] = (df['raw_score'] * 100).round(1)

    # ── 3. 위험 등급 부여 ──────────────────────────────────
    def grade(score):
        if score >= 80:   return "🔴 심각"
        elif score >= 60: return "🟠 경고"
        elif score >= 40: return "🟡 주의"
        else:             return "🟢 정상"

    df['risk_grade'] = df['priority_score'].apply(grade)

    # ── 4. segment_id 단위 집계 ────────────────────────────
    seg_agg = df.groupby('segment_id').agg(
        nearest_station_name=('nearest_station_name', 'first'),
        second_station_name=('second_station_name', 'first'),
        link_count=('LINK_ID', 'count'),
        priority_score=('priority_score', 'max'),     # 구간 내 최악 링크 기준
        priority_score_mean=('priority_score', 'mean'),
        bc_under_construction=('bc_under_construction', 'mean'),
        bc_change=('bc_change', 'mean'),
        lane_remain_ratio=('lane_remain_ratio', 'min'),  # 구간 내 최악 차로 기준
        speed_penalty_multiplier=('speed_penalty_multiplier', 'min'),
        ROAD_RANK=('ROAD_RANK', lambda x: x.mode()[0] if len(x) > 0 else '104'),
    ).reset_index()

    seg_agg['risk_grade'] = seg_agg['priority_score'].apply(grade)

    # 구간명 (두 정거장 이름으로 표현)
    seg_agg['segment_name'] = (
        seg_agg['nearest_station_name'] + " ↔ " + seg_agg['second_station_name']
    )

    seg_agg = seg_agg.sort_values('priority_score', ascending=False).reset_index(drop=True)
    seg_agg['rank'] = seg_agg.index + 1

    return seg_agg, df


def get_top_risk_segments(seg_df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """우선순위 상위 N개 구간 반환"""
    cols = ['rank', 'segment_id', 'segment_name', 'priority_score',
            'risk_grade', 'bc_under_construction', 'bc_change',
            'lane_remain_ratio', 'speed_penalty_multiplier', 'link_count']
    return seg_df.head(n)[cols]
