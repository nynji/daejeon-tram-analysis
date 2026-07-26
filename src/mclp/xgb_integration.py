"""
XGBoost-MCLP 통합 모듈
======================
XGBoost 모델의 reference tables를 활용하여 구간별 위험도를 산출하고
MCLP 수요 가중치에 반영.

XGBoost 핵심 정보:
- is_bottleneck_slot: 구간×시간대별 병목 여부 (Train 기반 통계)
- construction_lane_ratio_daily: 구간×날짜별 잔여 차로 비율
- network_features: 구간별 매개중심성 변화 (pre vs during)
- SHAP: lane_remain_ratio가 심각 클래스 3위 (0.327) → 공사 강도가 주의→심각 전이 결정

산출물:
- segment별 xgb_risk_score (0~1): 병목 빈도 + BC 악화율 + 공사 강도
- MCLP weights에 곱해지는 xgb_multiplier (1.0~2.0)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
XGB_MODEL_DIR = BASE_DIR / "repo" / "src" / "xgboost_model" / "model"
REF_TABLES_DIR = XGB_MODEL_DIR / "reference_tables"


def load_xgb_risk_scores() -> pd.DataFrame:
    """XGBoost reference tables로부터 구간별 위험 스코어 산출.

    3가지 관점을 결합:
    1. 병목 빈도 (is_bottleneck_slot 비율) — "이 구간이 얼마나 자주 병목인가"
    2. BC 악화율 (during/pre 비율) — "공사로 구조적 부담이 얼마나 증가했나"
    3. 공사 강도 (1 - lane_remain_ratio) — "차로가 얼마나 줄었나"

    Returns:
        DataFrame with columns: [segment_id, xgb_risk_score, xgb_multiplier,
                                  bottleneck_rate, bc_change_ratio, construction_severity]
    """
    if not REF_TABLES_DIR.exists():
        logger.warning(
            f"XGBoost reference tables 없음: {REF_TABLES_DIR}\n"
            "  → XGB 위험도 미반영, multiplier=1.0 사용"
        )
        return pd.DataFrame(columns=[
            "segment_id", "xgb_risk_score", "xgb_multiplier",
            "bottleneck_rate", "bc_change_ratio", "construction_severity"
        ])

    # ─── 1. 병목 빈도 ───
    bn_path = REF_TABLES_DIR / "is_bottleneck_slot_TRAIN_ONLY.csv"
    if bn_path.exists():
        bn = pd.read_csv(bn_path)
        # segment_key에서 방향 제거하여 segment_id로 집계
        bn["segment_id"] = bn["segment_key"].str.rsplit("_", n=1).str[0]
        bottleneck_rate = bn.groupby("segment_id")["is_bottleneck_slot"].mean().reset_index()
        bottleneck_rate.columns = ["segment_id", "bottleneck_rate"]
        # severity_score 평균도 가져오기
        severity = bn.groupby("segment_id")["severity_score"].mean().reset_index()
        severity.columns = ["segment_id", "avg_severity_score"]
        bottleneck_rate = bottleneck_rate.merge(severity, on="segment_id", how="left")
    else:
        logger.warning("is_bottleneck_slot 없음")
        bottleneck_rate = pd.DataFrame(columns=["segment_id", "bottleneck_rate", "avg_severity_score"])

    # ─── 2. BC 악화율 ───
    nf_path = REF_TABLES_DIR / "network_features.parquet"
    if nf_path.exists():
        nf = pd.read_parquet(nf_path)
        nf["segment_id"] = nf["segment_key"].str.rsplit("_", n=1).str[0]
        # BC 변화: during이 pre보다 클수록 공사로 인해 부담 집중
        # 또는 pre가 크고 during이 작아지면 우회 발생 (주변 구간에 부담 전이)
        nf["bc_change_ratio"] = np.where(
            nf["betweenness_pre"] > 0,
            (nf["betweenness_during"] - nf["betweenness_pre"]) / nf["betweenness_pre"],
            0
        )
        # 양방향 중 더 나쁜 쪽 사용
        bc_risk = nf.groupby("segment_id").agg(
            bc_change_ratio=("bc_change_ratio", lambda x: x.abs().max()),
            betweenness_max=("betweenness_pre", "max"),
        ).reset_index()
    else:
        logger.warning("network_features 없음")
        bc_risk = pd.DataFrame(columns=["segment_id", "bc_change_ratio", "betweenness_max"])

    # ─── 3. 공사 강도 (최신 날짜 기준) ───
    cl_path = REF_TABLES_DIR / "construction_lane_ratio_daily.parquet"
    if cl_path.exists():
        cl = pd.read_parquet(cl_path)
        # 가장 최근 날짜의 lane_remain_ratio
        latest = cl.sort_values("date").groupby("segment_id").last().reset_index()
        construction = latest[["segment_id", "lane_remain_ratio"]].copy()
        construction["construction_severity"] = 1.0 - construction["lane_remain_ratio"]
    else:
        logger.warning("construction_lane_ratio 없음")
        construction = pd.DataFrame(columns=["segment_id", "lane_remain_ratio", "construction_severity"])

    # ─── 4. 통합 스코어 산출 ───
    # 모든 segment_id 수집
    all_segments = set()
    for df in [bottleneck_rate, bc_risk, construction]:
        if len(df) > 0:
            all_segments.update(df["segment_id"].unique())

    result = pd.DataFrame({"segment_id": sorted(all_segments)})

    # 병합
    result = result.merge(bottleneck_rate[["segment_id", "bottleneck_rate"]], on="segment_id", how="left")
    result = result.merge(bc_risk[["segment_id", "bc_change_ratio"]], on="segment_id", how="left")
    result = result.merge(construction[["segment_id", "construction_severity"]], on="segment_id", how="left")

    # 결측 채우기
    result["bottleneck_rate"] = result["bottleneck_rate"].fillna(0)
    result["bc_change_ratio"] = result["bc_change_ratio"].fillna(0)
    result["construction_severity"] = result["construction_severity"].fillna(0)

    # ─── 통합 위험 스코어 ───
    # 가중 합산: 병목빈도(40%) + 공사강도(40%) + BC변화(20%)
    # 각각 0~1 정규화 후 합산
    bn_norm = result["bottleneck_rate"] / max(result["bottleneck_rate"].max(), 0.001)
    cs_norm = result["construction_severity"]  # 이미 0~1
    bc_norm = result["bc_change_ratio"].clip(0, 1)  # 0~1로 클리핑

    result["xgb_risk_score"] = (
        0.4 * bn_norm
        + 0.4 * cs_norm
        + 0.2 * bc_norm
    ).clip(0, 1)

    # ─── XGB multiplier: risk_score → 가중치 배수 (1.0 ~ 2.0) ───
    # 위험도 0 → multiplier 1.0 (변화 없음)
    # 위험도 1 → multiplier 2.0 (수요 가중치 2배)
    result["xgb_multiplier"] = 1.0 + result["xgb_risk_score"]

    logger.info(
        f"XGB 위험 스코어 산출: {len(result)}개 구간, "
        f"risk 범위 [{result['xgb_risk_score'].min():.3f}, {result['xgb_risk_score'].max():.3f}], "
        f"multiplier 범위 [{result['xgb_multiplier'].min():.3f}, {result['xgb_multiplier'].max():.3f}]"
    )

    # 상위 10개 출력
    top10 = result.nlargest(10, "xgb_risk_score")
    for _, r in top10.iterrows():
        logger.info(
            f"  {r['segment_id']}: risk={r['xgb_risk_score']:.3f} "
            f"(bn={r['bottleneck_rate']:.2f}, cs={r['construction_severity']:.2f}, "
            f"bc={r['bc_change_ratio']:.2f}) → mult={r['xgb_multiplier']:.2f}"
        )

    return result


def get_xgb_multiplier_map() -> dict:
    """segment_id → xgb_multiplier 딕셔너리 반환.

    MCLP weights.py에서 사용.
    """
    df = load_xgb_risk_scores()
    if len(df) == 0:
        return {}
    return dict(zip(df["segment_id"], df["xgb_multiplier"]))
