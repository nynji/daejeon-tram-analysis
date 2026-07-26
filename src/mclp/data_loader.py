"""
데이터 로딩/전처리 모듈
======================
주차장, 소상공인, 소방시설, 보호구역, 횡단보도, 공사구간 등 로딩
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
import logging
import os

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"


def _try_read_csv(path: Path, encodings=("utf-8", "utf-8-sig", "cp949", "euc-kr"), **kwargs) -> pd.DataFrame:
    """여러 인코딩 시도하여 CSV 읽기."""
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"파일 읽기 실패 (모든 인코딩 시도): {path}")


def load_parking_candidates() -> pd.DataFrame:
    """거점 후보지 (주차장) 로딩 - 5개 구 통합.

    Returns:
        DataFrame with columns: [parking_id, name, lat, lon, capacity, address, district]
    """
    parking_dir = DATA_DIR / "주차장 정보 표준데이터"
    if not parking_dir.exists():
        raise FileNotFoundError(f"주차장 데이터 폴더 없음: {parking_dir}")

    files = list(parking_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"주차장 CSV 파일 없음: {parking_dir}")

    dfs = []
    for f in files:
        df = _try_read_csv(f)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    # 정규화
    result = pd.DataFrame({
        "parking_id": range(len(combined)),
        "name": combined["주차장명"],
        "lat": pd.to_numeric(combined["위도"], errors="coerce"),
        "lon": pd.to_numeric(combined["경도"], errors="coerce"),
        "capacity": pd.to_numeric(combined["주차구획수"], errors="coerce").fillna(0).astype(int),
        "address": combined.get("소재지도로명주소", combined.get("소재지지번주소", "")),
        "parking_type": combined["주차장구분"],
    })

    # 좌표 누락 제거
    before = len(result)
    result = result.dropna(subset=["lat", "lon"])
    result = result[(result["lat"] > 0) & (result["lon"] > 0)]
    dropped = before - len(result)
    if dropped > 0:
        logger.warning(f"주차장 좌표 누락 {dropped}건 제외")

    result = result.reset_index(drop=True)
    result["parking_id"] = range(len(result))

    logger.info(f"주차장 후보지 로딩: {len(result)}개 (5개 구)")
    return result


def load_demand_points() -> pd.DataFrame:
    """수요지점 (소상공인 상가 + 전통시장) 로딩.

    Returns:
        DataFrame with columns: [demand_id, name, lat, lon, type, dong]
    """
    # 소상공인 상가
    shop_path = DATA_DIR / "소상공인시장진흥공단_상가(상권)정보.csv"
    if not shop_path.exists():
        raise FileNotFoundError(f"상가 데이터 없음: {shop_path}")

    shops = pd.read_csv(shop_path, usecols=[
        "상호명", "시도코드", "위도", "경도", "행정동명", "상권업종대분류명"
    ])

    # 대전광역시 필터링 (시도코드 30)
    shops = shops[shops["시도코드"] == 30].copy()

    shop_df = pd.DataFrame({
        "name": shops["상호명"],
        "lat": pd.to_numeric(shops["위도"], errors="coerce"),
        "lon": pd.to_numeric(shops["경도"], errors="coerce"),
        "type": "상가",
        "dong": shops["행정동명"],
        "category": shops["상권업종대분류명"],
    })

    # 전통시장
    market_path = DATA_DIR / "소상공인시장진흥공단_전통시장.csv"
    if market_path.exists():
        markets = _try_read_csv(market_path)
        # 대전 필터링
        markets = markets[
            markets["소재지도로명주소"].str.contains("대전", na=False)
        ].copy()

        market_df = pd.DataFrame({
            "name": markets["시장명"],
            "lat": pd.to_numeric(markets["위도"], errors="coerce"),
            "lon": pd.to_numeric(markets["경도"], errors="coerce"),
            "type": "전통시장",
            "dong": "",
            "category": "전통시장",
        })

        combined = pd.concat([shop_df, market_df], ignore_index=True)
    else:
        logger.warning("전통시장 데이터 없음, 상가만 사용")
        combined = shop_df

    # 좌표 누락 제거
    before = len(combined)
    combined = combined.dropna(subset=["lat", "lon"])
    combined = combined[(combined["lat"] > 0) & (combined["lon"] > 0)]
    dropped = before - len(combined)
    if dropped > 0:
        logger.warning(f"수요지점 좌표 누락 {dropped}건 제외")

    combined = combined.reset_index(drop=True)
    combined["demand_id"] = range(len(combined))

    logger.info(f"수요지점 로딩: {len(combined)}개 (상가 + 전통시장)")
    return combined


def load_fire_hydrants() -> pd.DataFrame:
    """소방용수시설 위치 로딩 (대전광역시).

    Returns:
        DataFrame with columns: [fire_id, lat, lon, type]
    """
    fire_path = DATA_DIR / "소방청_소방용수시설_20240207.csv"
    if not fire_path.exists():
        raise FileNotFoundError(f"소방용수시설 데이터 없음: {fire_path}")

    fire = _try_read_csv(fire_path)

    # 대전 필터링
    fire = fire[fire["시도명"].str.contains("대전", na=False)].copy()

    result = pd.DataFrame({
        "fire_id": range(len(fire)),
        "lat": pd.to_numeric(fire["위도"], errors="coerce"),
        "lon": pd.to_numeric(fire["경도"], errors="coerce"),
        "type": fire.get("시설유형코드", ""),
    })

    result = result.dropna(subset=["lat", "lon"])
    result = result.reset_index(drop=True)
    result["fire_id"] = range(len(result))

    logger.info(f"소방용수시설 로딩: {len(result)}개")
    return result


def load_school_zones() -> pd.DataFrame:
    """어린이보호구역 로딩 (5개 구 통합).

    Returns:
        DataFrame with columns: [zone_id, name, lat, lon, road_width]
    """
    zone_dir = DATA_DIR / "어린이 보호구역 표준데이터"
    if not zone_dir.exists():
        raise FileNotFoundError(f"어린이보호구역 폴더 없음: {zone_dir}")

    files = list(zone_dir.glob("*.csv"))
    dfs = []
    for f in files:
        df = _try_read_csv(f)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    result = pd.DataFrame({
        "zone_id": range(len(combined)),
        "name": combined["대상시설명"],
        "lat": pd.to_numeric(combined["위도"], errors="coerce"),
        "lon": pd.to_numeric(combined["경도"], errors="coerce"),
        "road_width": pd.to_numeric(combined.get("보호구역도로폭", 0), errors="coerce"),
    })

    result = result.dropna(subset=["lat", "lon"])
    result = result.reset_index(drop=True)
    result["zone_id"] = range(len(result))

    logger.info(f"어린이보호구역 로딩: {len(result)}개")
    return result


def load_crosswalks() -> pd.DataFrame:
    """횡단보도 위치 로딩.

    Returns:
        DataFrame with columns: [crosswalk_id, lat, lon, width, lanes]
    """
    cw_path = DATA_DIR / "대전광역시_횡단보도.csv"
    if not cw_path.exists():
        raise FileNotFoundError(f"횡단보도 데이터 없음: {cw_path}")

    cw = _try_read_csv(cw_path)

    result = pd.DataFrame({
        "crosswalk_id": range(len(cw)),
        "lat": pd.to_numeric(cw["위도"], errors="coerce"),
        "lon": pd.to_numeric(cw["경도"], errors="coerce"),
        "width": pd.to_numeric(cw.get("횡단보도폭", 0), errors="coerce"),
        "lanes": pd.to_numeric(cw.get("차로수", 0), errors="coerce"),
    })

    result = result.dropna(subset=["lat", "lon"])
    result = result.reset_index(drop=True)
    result["crosswalk_id"] = range(len(result))

    logger.info(f"횡단보도 로딩: {len(result)}개")
    return result


def load_construction_zones() -> pd.DataFrame:
    """공사구간 좌표 로딩.

    Returns:
        DataFrame with columns: [zone_name, start_lat, start_lon, end_lat, end_lon]
    """
    cz_path = OUTPUT_DIR / "construction_zones_geocoded.csv"
    if not cz_path.exists():
        raise FileNotFoundError(f"공사구간 좌표 없음: {cz_path}")

    cz = pd.read_csv(cz_path)

    result = pd.DataFrame({
        "zone_name": cz["공구"].astype(str) + "_" + cz["노선명"],
        "start_lat": pd.to_numeric(cz["start_lat"], errors="coerce"),
        "start_lon": pd.to_numeric(cz["start_lon"], errors="coerce"),
        "end_lat": pd.to_numeric(cz["end_lat"], errors="coerce"),
        "end_lon": pd.to_numeric(cz["end_lon"], errors="coerce"),
    })

    logger.info(f"공사구간 로딩: {len(result)}개")
    return result


def load_segment_priority() -> pd.DataFrame:
    """통합 segment priority 로딩.

    Returns:
        DataFrame with segment_id, priority_score, integrated_score, risk_grade
    """
    sp_path = OUTPUT_DIR / "segment_priority.csv"
    if not sp_path.exists():
        raise FileNotFoundError(f"segment_priority 없음: {sp_path}")

    sp = pd.read_csv(sp_path, encoding="utf-8-sig")
    logger.info(f"Segment priority 로딩: {len(sp)}개 구간")
    return sp


def load_pedestrian_roads() -> pd.DataFrame:
    """보행자전용도로 로딩 (5개 구 통합).

    Returns:
        DataFrame with start/end coordinates and width
    """
    ped_dir = DATA_DIR / "보행자 전용도로 표준데이터"
    if not ped_dir.exists():
        raise FileNotFoundError(f"보행자전용도로 폴더 없음: {ped_dir}")

    files = list(ped_dir.glob("*.csv"))
    dfs = []
    for f in files:
        df = _try_read_csv(f)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    result = pd.DataFrame({
        "name": combined["보행자전용도로명"],
        "start_lat": pd.to_numeric(combined["보행자전용도로시작점위도"], errors="coerce"),
        "start_lon": pd.to_numeric(combined["보행자전용도로시작점경도"], errors="coerce"),
        "end_lat": pd.to_numeric(combined["보행자전용도로종료점위도"], errors="coerce"),
        "end_lon": pd.to_numeric(combined["보행자전용도로종료점경도"], errors="coerce"),
        "width": pd.to_numeric(combined.get("보행자전용도로폭", 0), errors="coerce"),
    })

    result = result.dropna(subset=["start_lat", "start_lon"])
    result = result.reset_index(drop=True)

    logger.info(f"보행자전용도로 로딩: {len(result)}개")
    return result


def load_parking_zones() -> pd.DataFrame:
    """택배 주정차 허용 구간 로딩.

    Returns:
        DataFrame with columns: [start_name, end_name, length_km, time_start, time_end]
    """
    pz_path = DATA_DIR / "경찰청 대전광역시경찰청_대전지역 주정차 허용현황_택배소형화물차량_20251127.csv"
    if not pz_path.exists():
        raise FileNotFoundError(f"주정차 허용현황 없음: {pz_path}")

    pz = _try_read_csv(pz_path, encodings=("cp949", "utf-8", "euc-kr"))

    result = pd.DataFrame({
        "start_name": pz["시점"],
        "end_name": pz["종점"],
        "length_km": pd.to_numeric(pz["연장(Km)"], errors="coerce"),
        "permission_type": pz["허용구분"],
        "time_info": pz["허용시간(상시)"],
    })

    logger.info(f"택배 주정차 허용구간 로딩: {len(result)}개")
    return result


def load_station_coords() -> pd.DataFrame:
    """역 좌표 로딩 (segment 매핑용).

    Returns:
        DataFrame with station_no, station_name, lat, lon
    """
    st_path = DATA_DIR / "network" / "역_좌표.csv"
    if not st_path.exists():
        raise FileNotFoundError(f"역 좌표 데이터 없음: {st_path}")

    st = pd.read_csv(st_path)
    result = pd.DataFrame({
        "station_no": st["station_no"],
        "station_name": st["station_name"],
        "lat": pd.to_numeric(st["lat"], errors="coerce"),
        "lon": pd.to_numeric(st["lon"], errors="coerce"),
    })

    logger.info(f"역 좌표 로딩: {len(result)}개")
    return result
