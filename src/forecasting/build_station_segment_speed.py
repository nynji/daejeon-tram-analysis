"""
정류장 구간(segment_id) x 방향(AB/BA) x 10분 단위 속도(y_hat)를 원시 속도
로그에서 직접 산출한다.

문제 배경: 원본 속도 로그는 하루 288개(10분 기준) 슬롯 중 상당수를 구조적으로
비워두고 배포한다. 5분 단위로 먼저 집계한 뒤 두 슬롯을 짝지어 10분으로
리샘플링하는 방식은 이 구조적 공백과 만나 결측이 과도하게 커진다.

이 스크립트는 중간 리샘플링 단계 없이, 10분 창 안에 실제로 찍힌 모든 원시
레코드를 곧바로 풀링해서 길이가중평균한다. 두 가지 풀링 방식을 함께 계산해서
비교한다:
- A(레코드가중): 같은 링크가 창 안에서 여러 번 찍혔으면 레코드 각각을 개별
  항으로 분자/분모에 포함하는 표준적인 레코드 단위 가중평균.
- B(링크가중, 기본 채택 y_hat): 링크별로 먼저 평균 낸 뒤, 그 창에서 관측된
  distinct 링크만 길이가중평균(중복 관측이 이중 가중되지 않음).

coverage는 B와 동일하게 distinct 관측 링크 길이 / 그 시점 이론적으로 관측
가능한 링크 길이 총합으로 계산하고, 임계값(COVERAGE_MIN) 미만이면 결측
처리한다. "이론적으로 관측 가능한 링크"는 link_observation_status.csv의
상태(정상형 / 연계중단형 / 미관측형)를 반영한 동적 분모로 계산한다 —
연계가 중단된 링크는 중단 시점 이후로는 분모에서 빠지고, 애초에 미관측형인
링크는 처음부터 분모에서 제외된다(고정 분모를 쓰면 애초에 못 볼 링크
때문에 coverage가 영구적으로 낮게 잡히는 문제를 방지).

필요 입력 파일:
- dataset/network/정류장_구간_링크.csv   (segment_id, direction, link_id, length_m)
- dataset/network/link_observation_status.csv  (LINK_ID, status, last_observed_date)
- dataset/modeling/대전_속도_2024_2026.csv.gz   (원시 속도 로그, 컬럼 없음:
  date_str, time_str, link_id, col4, speed, flag)
"""

import numpy as np
import pandas as pd

STATION_LINKS_CSV = "dataset/network/정류장_구간_링크.csv"
STATUS_CSV = "dataset/network/link_observation_status.csv"
SPEED_GZ = "dataset/modeling/대전_속도_2024_2026.csv.gz"
OUT_CSV = "dataset/modeling/station_segment_speed_10min_pooled.csv"

DATA_START = "2024-10-01"
DATA_END_EXCLUSIVE = "2026-07-01"  # 마지막 온전한 10분 슬롯: 하루 전 23:50
COVERAGE_MIN = 0.6
CHUNKSIZE = 5_000_000
RAW_COLUMNS = ["date_str", "time_str", "link_id", "col4", "speed", "flag"]


def load_station_links() -> pd.DataFrame:
    df = pd.read_csv(STATION_LINKS_CSV, dtype=str)
    df["length_m"] = df["length_m"].astype(float)
    return df


def build_dynamic_daily_denominator(station_links: pd.DataFrame, status_map: dict,
                                     day_index: pd.DatetimeIndex) -> dict:
    """(segment_id, direction) -> 일별 동적 분모(이론적 관측가능 링크 길이합) Series.

    링크 상태에 따라:
    - 미관측형: 분모에서 항상 제외
    - 연계중단형: last_observed_date까지만 분모에 포함, 이후는 제외
    - 그 외(정상형): 항상 분모에 포함
    """
    result = {}
    for (seg, direction), grp in station_links.groupby(["segment_id", "direction"]):
        denom = pd.Series(0.0, index=day_index)
        for _, r in grp.iterrows():
            lid, length = r["link_id"], r["length_m"]
            meta = status_map.get(lid)
            if meta is None or meta["status"] == "미관측형":
                continue
            elif meta["status"] == "연계중단형":
                last_dt = pd.Timestamp(meta["last_observed_date"])
                denom = denom + length * (day_index <= last_dt).astype(float)
            else:
                denom = denom + length
        result[(seg, direction)] = denom
    return result


def stream_10min_pooled(station_links: pd.DataFrame) -> pd.DataFrame:
    """원시 속도 로그를 청크 단위로 스트리밍하며 10분 창 단위로 직접 풀링 집계한다."""
    link_map = station_links[["link_id", "segment_id", "direction", "length_m"]].copy()
    target_links = set(link_map["link_id"])
    print(f"[10분 풀링] 대상 LINK_ID {len(target_links)}개")

    chunk_A = []        # segment,direction,ts10 -> sum_speed_len_A, sum_len_A (레코드 단위)
    chunk_perlink = []   # segment,direction,ts10,link_id -> speed_sum, speed_cnt, length_m (B/coverage용 중간)

    rows_scanned = 0
    chunk_idx = 0
    reader = pd.read_csv(
        SPEED_GZ, header=None, names=RAW_COLUMNS,
        dtype={"date_str": str, "time_str": str, "link_id": str, "col4": str, "flag": str},
        usecols=["date_str", "time_str", "link_id", "speed"], chunksize=CHUNKSIZE,
    )
    for chunk in reader:
        chunk_idx += 1
        rows_scanned += len(chunk)
        chunk["speed"] = pd.to_numeric(chunk["speed"], errors="coerce")
        filtered = chunk[chunk["link_id"].isin(target_links) & chunk["speed"].notna()]
        if not filtered.empty:
            filtered = filtered.copy()
            minute = filtered["time_str"].str[2:4].astype(int)
            floor_min = (minute // 10) * 10
            filtered["ts10_str"] = filtered["date_str"] + filtered["time_str"].str[:2] + floor_min.astype(str).str.zfill(2)

            rows = filtered.merge(link_map, on="link_id", how="inner")
            rows["speed_x_len"] = rows["speed"] * rows["length_m"]

            agg_a = rows.groupby(["segment_id", "direction", "ts10_str"], as_index=False).agg(
                sum_speed_len_A=("speed_x_len", "sum"), sum_len_A=("length_m", "sum"))
            chunk_A.append(agg_a)

            per_link = rows.groupby(["segment_id", "direction", "ts10_str", "link_id"], as_index=False).agg(
                speed_sum=("speed", "sum"), speed_cnt=("speed", "size"), length_m=("length_m", "first"))
            chunk_perlink.append(per_link)

        if chunk_idx % 20 == 0:
            nA = sum(len(a) for a in chunk_A)
            print(f"  chunk {chunk_idx}, 누적 스캔 {rows_scanned:,}행, 누적 A집계행 {nA:,}")

    print(f"[10분 풀링] 총 스캔 {rows_scanned:,}행")

    full_A = pd.concat(chunk_A, ignore_index=True)
    final_A = full_A.groupby(["segment_id", "direction", "ts10_str"], as_index=False).agg(
        sum_speed_len_A=("sum_speed_len_A", "sum"), sum_len_A=("sum_len_A", "sum"))

    full_pl = pd.concat(chunk_perlink, ignore_index=True)
    final_pl = full_pl.groupby(["segment_id", "direction", "ts10_str", "link_id"], as_index=False).agg(
        speed_sum=("speed_sum", "sum"), speed_cnt=("speed_cnt", "sum"), length_m=("length_m", "first"))
    final_pl["link_avg_speed"] = final_pl["speed_sum"] / final_pl["speed_cnt"]
    final_pl["speed_len_B"] = final_pl["link_avg_speed"] * final_pl["length_m"]

    final_B = final_pl.groupby(["segment_id", "direction", "ts10_str"], as_index=False).agg(
        sum_speed_len_B=("speed_len_B", "sum"), sum_len_distinct=("length_m", "sum"))

    merged = final_A.merge(final_B, on=["segment_id", "direction", "ts10_str"], how="outer")
    merged["ts"] = pd.to_datetime(merged["ts10_str"], format="%Y%m%d%H%M")
    print(f"[10분 풀링] 10분 집계 최종 {len(merged):,}행")
    return merged[["segment_id", "direction", "ts", "sum_speed_len_A", "sum_len_A", "sum_speed_len_B", "sum_len_distinct"]]


def main():
    print("=== 메타데이터 로드 ===")
    station_links = load_station_links()
    status_df = pd.read_csv(STATUS_CSV, dtype=str)
    status_map = status_df.set_index("LINK_ID")[["status", "last_observed_date"]].to_dict("index")

    day_index = pd.date_range(DATA_START, DATA_END_EXCLUSIVE, freq="D", inclusive="left")
    dynamic_denom_daily = build_dynamic_daily_denominator(station_links, status_map, day_index)

    print("\n=== 10분 창 직접 풀링 스트리밍 ===")
    agg10 = stream_10min_pooled(station_links)
    agg10_by_key = {k: v for k, v in agg10.groupby(["segment_id", "direction"])}

    ten_min_index = pd.date_range(DATA_START, DATA_END_EXCLUSIVE, freq="10min", inclusive="left")
    day_of_10min = ten_min_index.floor("D")

    print("\n=== 세그먼트별 재조립 및 저장 ===")
    all_segments = sorted(station_links["segment_id"].unique())
    first_write = True
    compare_rows = []
    method_diff_rows = []

    for seg in all_segments:
        for direction in ["AB", "BA"]:
            key = (seg, direction)
            daily_denom = dynamic_denom_daily.get(key)
            if daily_denom is None:
                continue
            denom_10min = daily_denom.reindex(day_of_10min).values

            grp = agg10_by_key.get(key)
            if grp is None:
                s = pd.DataFrame({"sum_speed_len_A": np.nan, "sum_len_A": np.nan,
                                   "sum_speed_len_B": np.nan, "sum_len_distinct": np.nan}, index=ten_min_index)
            else:
                s = grp.set_index("ts")[["sum_speed_len_A", "sum_len_A", "sum_speed_len_B", "sum_len_distinct"]].reindex(ten_min_index)

            with np.errstate(divide="ignore", invalid="ignore"):
                coverage = s["sum_len_distinct"].values / denom_10min
            coverage = pd.Series(coverage, index=ten_min_index)

            y_A = s["sum_speed_len_A"] / s["sum_len_A"]
            y_B = s["sum_speed_len_B"] / s["sum_len_distinct"]
            valid = coverage >= COVERAGE_MIN
            y_A = y_A.where(valid)
            y_B = y_B.where(valid)

            out = pd.DataFrame({
                "timestamp": ten_min_index,
                "segment_id": seg,
                "direction": direction,
                "y_hat": y_B.values,
                "y_hat_record_weighted": y_A.values,
                "coverage": coverage.values,
            })
            out.to_csv(OUT_CSV, mode="w" if first_write else "a", header=first_write, index=False, encoding="utf-8-sig")
            first_write = False

            both_valid = y_A.notna() & y_B.notna()
            if both_valid.sum() > 0:
                diff = (y_A[both_valid] - y_B[both_valid]).abs()
                method_diff_rows.append(dict(segment_id=seg, direction=direction,
                                              mean_abs_diff=round(diff.mean(), 3), max_abs_diff=round(diff.max(), 3)))

            compare_rows.append(dict(segment_id=seg, direction=direction,
                                      nan_rate=round(out["y_hat"].isna().mean(), 4)))
        print(f"  {seg} 완료")

    print(f"\n저장 완료: {OUT_CSV}")

    cmp_df = pd.DataFrame(compare_rows)
    pd.set_option("display.width", 160)
    print("\n=== segment_id x direction별 결측률 ===")
    print(cmp_df.sort_values("nan_rate", ascending=False).to_string(index=False))
    print(f"\n전체 평균 결측률: {cmp_df['nan_rate'].mean():.4f}")

    print("\n=== 방식 A(레코드가중) vs B(링크가중) 차이 ===")
    diff_df = pd.DataFrame(method_diff_rows)
    if len(diff_df):
        print(diff_df.sort_values("mean_abs_diff", ascending=False).head(20).to_string(index=False))
        print(f"\n전체 평균 절대차이: {diff_df['mean_abs_diff'].mean():.4f} km/h, 최대: {diff_df['max_abs_diff'].max():.4f} km/h")
    else:
        print("두 방식이 동시에 유효한 데이터 없음")

    print("\n=== 잔여 결측 시간대 패턴 (전체 세그먼트 통합) ===")
    full = pd.read_csv(OUT_CSV, parse_dates=["timestamp"])
    full["tod"] = full["timestamp"].dt.strftime("%H:%M")
    tod_nan = full.groupby("tod")["y_hat"].apply(lambda s: s.isna().mean())
    print("시각대별 결측률 (상위 15개):")
    print(tod_nan.sort_values(ascending=False).head(15).to_string())
    print("\n시각대별 결측률 (하위 15개, 가장 양호):")
    print(tod_nan.sort_values().head(15).to_string())


if __name__ == "__main__":
    main()
