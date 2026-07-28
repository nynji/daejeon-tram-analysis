#!/usr/bin/env python3
"""Aggregate filtered 5-minute link speeds to 15-minute segment speed indices."""

from __future__ import annotations

import csv
import datetime as dt
import gzip
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, TextIO

from link_version_bridge import load_link_history


HERE = Path(__file__).resolve().parent
MAPPING_PATH = HERE / "outputs" / "01_mapping_qc" / "segment_link_mapping.csv"
INPUT_PATH = HERE / "outputs" / "02_speed_5min_qc" / "speed_5min_filtered.csv.gz"
OUTPUT_DIR = HERE / "outputs" / "03_speed_index_15min"
PREINDEX_PATH = OUTPUT_DIR / "_segment_speed_15min_preindex.csv.gz"
OUTPUT_PATH = OUTPUT_DIR / "segment_speed_index_15min.csv.gz"

EXCLUDED_KEY = ("SEG_07_207_208", "AB")
MIN_LENGTH_COVERAGE = 0.50
FREE_FLOW_CALIBRATION_COVERAGE = 0.95
FREE_FLOW_PERCENTILE = 0.85
MIN_FREE_FLOW_OBSERVATIONS = 1_000
MIN_VALID_DAYS_FOR_SPATIAL_QC = 180
MIN_VALID_BINS_PER_DAY = 90
HISTOGRAM_STEP_KMH = 0.1
HISTOGRAM_MAX_KMH = 150.0
HISTOGRAM_SIZE = int(HISTOGRAM_MAX_KMH / HISTOGRAM_STEP_KMH) + 1
START_DATE = dt.date(2024, 10, 1)
END_DATE = dt.date(2026, 7, 1)


def load_routes() -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    if not MAPPING_PATH.exists():
        raise FileNotFoundError(MAPPING_PATH)
    with MAPPING_PATH.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    history = load_link_history(rows)

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["segment_id"], row["direction"])].append(row)

    routes: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    for key, route_rows in grouped.items():
        route_rows.sort(key=lambda row: int(row["link_sequence"]))
        first = route_rows[0]
        if key == EXCLUDED_KEY:
            excluded.append(
                {
                    "segment_id": key[0],
                    "direction": key[1],
                    "from_station_no": first["from_station_no"],
                    "from_station_name": first["from_station_name"],
                    "to_station_no": first["to_station_no"],
                    "to_station_name": first["to_station_name"],
                    "excluded_from": "15min_aggregation_and_dtw_training",
                    "reason": (
                        "2025-11-01 이후 표준 링크 분할과 ITS 제공 링크 미동기화로 "
                        "관측 길이 커버리지 54.9%"
                    ),
                }
            )
            continue
        parent_ids = {
            history[row["link_id"]]["historical_link_id"]
            for row in route_rows
            if history[row["link_id"]]["historical_link_id"]
        }
        grouped_links: dict[str, dict[str, Any]] = {}
        for row in route_rows:
            link_id = row["link_id"]
            length_m = float(row["length_m"])
            historical_link_id = history[link_id]["historical_link_id"]
            group_id = historical_link_id or link_id
            group = grouped_links.setdefault(
                group_id,
                {
                    "group_id": group_id,
                    "historical_alias_link_id": group_id if group_id in parent_ids else "",
                    "current_links": [],
                    "total_length_m": 0.0,
                },
            )
            group["current_links"].append((link_id, length_m))
            group["total_length_m"] += length_m
        links = [(row["link_id"], float(row["length_m"])) for row in route_rows]
        routes.append(
            {
                "segment_id": key[0],
                "segment_order": int(first["segment_order"]),
                "direction": key[1],
                "branch": first["branch"],
                "from_station_no": first["from_station_no"],
                "from_station_name": first["from_station_name"],
                "to_station_no": first["to_station_no"],
                "to_station_name": first["to_station_name"],
                "links": links,
                "link_version_groups": list(grouped_links.values()),
                "total_length_m": sum(length for _, length in links),
            }
        )
    routes.sort(key=lambda row: (row["segment_order"], row["direction"]))
    if len(routes) != 89 or len(excluded) != 1:
        raise RuntimeError(f"Expected 89 included and 1 excluded spatial units, got {len(routes)} and {len(excluded)}")
    return routes, excluded


def new_link_bins() -> tuple[list[float], list[int], list[float], list[float]]:
    return ([0.0] * 96, [0] * 96, [math.inf] * 96, [-math.inf] * 96)


def update_link_bin(
    state: tuple[list[float], list[int], list[float], list[float]],
    bin_index: int,
    value: float,
) -> None:
    totals, counts, minima, maxima = state
    totals[bin_index] += value
    counts[bin_index] += 1
    if value < minima[bin_index]:
        minima[bin_index] = value
    if value > maxima[bin_index]:
        maxima[bin_index] = value


def median_from_state(
    state: tuple[list[float], list[int], list[float], list[float]], bin_index: int
) -> float | None:
    totals, counts, minima, maxima = state
    count = counts[bin_index]
    if count == 0:
        return None
    if count == 1:
        return totals[bin_index]
    if count == 2:
        return totals[bin_index] / 2.0
    if count == 3:
        return totals[bin_index] - minima[bin_index] - maxima[bin_index]
    # Defensive fallback: duplicates should already have been collapsed in the
    # 5-minute stage, but a mean remains stable if an unexpected fourth value occurs.
    return totals[bin_index] / count


def hhmm_from_bin(bin_index: int) -> str:
    minute = bin_index * 15
    return f"{minute // 60:02d}{minute % 60:02d}"


def add_histogram(histogram: list[int], value: float) -> None:
    index = int(round(value / HISTOGRAM_STEP_KMH))
    index = max(0, min(index, HISTOGRAM_SIZE - 1))
    histogram[index] += 1


def histogram_percentile(histogram: list[int], percentile: float) -> float:
    total = sum(histogram)
    if total == 0:
        raise ValueError("Cannot calculate a percentile from an empty histogram")
    target = max(1, math.ceil(total * percentile))
    cumulative = 0
    for index, count in enumerate(histogram):
        cumulative += count
        if cumulative >= target:
            return index * HISTOGRAM_STEP_KMH
    return HISTOGRAM_MAX_KMH


def write_dict_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def process_day(
    service_date: str,
    link_bins: dict[str, tuple[list[float], list[int], list[float], list[float]]],
    routes: list[dict[str, Any]],
    output: TextIO,
    stats: dict[tuple[str, str], dict[str, Any]],
) -> int:
    median_cache: dict[str, list[float | None]] = {}
    count_cache: dict[str, list[int]] = {}
    for link_id, state in link_bins.items():
        median_cache[link_id] = [median_from_state(state, index) for index in range(96)]
        count_cache[link_id] = state[1]

    written = 0
    for route in routes:
        key = (route["segment_id"], route["direction"])
        route_stats = stats[key]
        pass_bins_for_day = 0
        for bin_index in range(96):
            observed_length = 0.0
            travel_time_hours = 0.0
            observed_link_count = 0
            source_5min_count = 0
            historical_alias_group_count = 0
            for group in route["link_version_groups"]:
                group_observed_length = 0.0
                group_travel_time_hours = 0.0
                group_observed_link_count = 0
                group_source_count = 0
                for link_id, length_m in group["current_links"]:
                    values = median_cache.get(link_id)
                    speed = values[bin_index] if values is not None else None
                    if speed is None or speed <= 0:
                        continue
                    group_observed_length += length_m
                    group_travel_time_hours += (length_m / 1000.0) / speed
                    group_observed_link_count += 1
                    group_source_count += count_cache[link_id][bin_index]

                group_total_length = group["total_length_m"]
                group_coverage = (
                    group_observed_length / group_total_length if group_total_length else 0.0
                )
                alias_link_id = group["historical_alias_link_id"]
                alias_values = median_cache.get(alias_link_id) if alias_link_id else None
                alias_speed = alias_values[bin_index] if alias_values is not None else None
                if (
                    group_coverage < MIN_LENGTH_COVERAGE
                    and alias_speed is not None
                    and alias_speed > 0
                ):
                    # Before a standard link is split, the historical parent ID
                    # represents the full length of its present-day child group.
                    observed_length += group_total_length
                    travel_time_hours += (group_total_length / 1000.0) / alias_speed
                    observed_link_count += 1
                    source_5min_count += count_cache[alias_link_id][bin_index]
                    historical_alias_group_count += 1
                else:
                    observed_length += group_observed_length
                    travel_time_hours += group_travel_time_hours
                    observed_link_count += group_observed_link_count
                    source_5min_count += group_source_count

            total_length = route["total_length_m"]
            coverage = observed_length / total_length if total_length else 0.0
            if observed_link_count == 0:
                quality_status = "NO_DATA"
                segment_speed = None
                route_stats["no_data_bins"] += 1
            elif coverage < MIN_LENGTH_COVERAGE:
                quality_status = "LOW_LINK_COVERAGE"
                segment_speed = None
                route_stats["low_coverage_bins"] += 1
            else:
                quality_status = "PASS"
                segment_speed = (observed_length / 1000.0) / travel_time_hours
                route_stats["pass_bins"] += 1
                pass_bins_for_day += 1
                route_stats["pass_bins_using_historical_alias"] += (
                    historical_alias_group_count > 0
                )
                add_histogram(route_stats["all_histogram"], segment_speed)
                route_stats["all_histogram_count"] += 1
                if coverage >= FREE_FLOW_CALIBRATION_COVERAGE:
                    add_histogram(route_stats["high_coverage_histogram"], segment_speed)
                    route_stats["high_coverage_histogram_count"] += 1

            route_stats["coverage_sum"] += coverage
            fields = [
                service_date,
                hhmm_from_bin(bin_index),
                str(bin_index),
                route["segment_id"],
                route["direction"],
                route["from_station_no"],
                route["from_station_name"],
                route["to_station_no"],
                route["to_station_name"],
                "" if segment_speed is None else f"{segment_speed:.3f}",
                f"{total_length:.3f}",
                f"{observed_length:.3f}",
                f"{coverage:.6f}",
                str(observed_link_count),
                str(source_5min_count),
                str(historical_alias_group_count),
                quality_status,
            ]
            output.write(",".join(map(str, fields)) + "\n")
            written += 1
        route_stats["days_with_96_pass_bins"] += pass_bins_for_day == 96
        route_stats["days_with_at_least_90_pass_bins"] += (
            pass_bins_for_day >= MIN_VALID_BINS_PER_DAY
        )
    return written


def initialize_stats(routes: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for route in routes:
        result[(route["segment_id"], route["direction"])] = {
            "route": route,
            "pass_bins": 0,
            "no_data_bins": 0,
            "low_coverage_bins": 0,
            "coverage_sum": 0.0,
            "days_with_96_pass_bins": 0,
            "days_with_at_least_90_pass_bins": 0,
            "all_histogram": [0] * HISTOGRAM_SIZE,
            "all_histogram_count": 0,
            "high_coverage_histogram": [0] * HISTOGRAM_SIZE,
            "high_coverage_histogram_count": 0,
            "pass_bins_using_historical_alias": 0,
        }
    return result


def calculate_preindex(
    routes: list[dict[str, Any]], stats: dict[tuple[str, str], dict[str, Any]]
) -> tuple[int, int]:
    header = [
        "service_date", "time_bin", "bin_index", "segment_id", "direction",
        "from_station_no", "from_station_name", "to_station_no", "to_station_name",
        "segment_speed_kmh", "mapped_length_m", "observed_length_m",
        "length_coverage", "observed_link_count", "source_5min_count",
        "historical_alias_group_count", "quality_status",
    ]
    input_rows = 0
    output_rows = 0
    current_date = ""
    link_bins: dict[str, tuple[list[float], list[int], list[float], list[float]]] = {}
    all_dates = [
        (START_DATE + dt.timedelta(days=offset)).strftime("%Y%m%d")
        for offset in range((END_DATE - START_DATE).days + 1)
    ]
    date_position = {value: index for index, value in enumerate(all_dates)}
    last_completed_position = -1
    with gzip.open(INPUT_PATH, "rt", encoding="utf-8", newline="") as source, gzip.open(
        PREINDEX_PATH, "wt", encoding="utf-8", newline="", compresslevel=1
    ) as output:
        first_line = source.readline().rstrip("\r\n")
        if not first_line.startswith("service_date,time_hhmm,link_id,speed_kmh"):
            raise RuntimeError(f"Unexpected input header: {first_line}")
        output.write(",".join(header) + "\n")
        for line in source:
            input_rows += 1
            service_date, time_hhmm, link_id, speed_text, *_ = line.rstrip("\r\n").split(",")
            if service_date not in date_position:
                raise RuntimeError(f"Input date is outside the modeling period: {service_date}")
            if current_date and service_date != current_date:
                output_rows += process_day(current_date, link_bins, routes, output, stats)
                last_completed_position = date_position[current_date]
                for missing_date in all_dates[last_completed_position + 1 : date_position[service_date]]:
                    output_rows += process_day(missing_date, {}, routes, output, stats)
                    last_completed_position = date_position[missing_date]
                if (last_completed_position + 1) % 50 == 0:
                    print(f"Aggregated through {current_date}", flush=True)
                link_bins = {}
            elif not current_date:
                for missing_date in all_dates[: date_position[service_date]]:
                    output_rows += process_day(missing_date, {}, routes, output, stats)
                    last_completed_position = date_position[missing_date]
            current_date = service_date
            minute = int(time_hhmm[:2]) * 60 + int(time_hhmm[2:])
            bin_index = minute // 15
            state = link_bins.get(link_id)
            if state is None:
                state = new_link_bins()
                link_bins[link_id] = state
            update_link_bin(state, bin_index, float(speed_text))
        if current_date:
            output_rows += process_day(current_date, link_bins, routes, output, stats)
            last_completed_position = date_position[current_date]
        for missing_date in all_dates[last_completed_position + 1 :]:
            output_rows += process_day(missing_date, {}, routes, output, stats)
    return input_rows, output_rows


def calculate_free_flow(
    routes: list[dict[str, Any]], stats: dict[tuple[str, str], dict[str, Any]]
) -> tuple[dict[tuple[str, str], float], list[dict[str, Any]]]:
    free_flow: dict[tuple[str, str], float] = {}
    rows: list[dict[str, Any]] = []
    for route in routes:
        key = (route["segment_id"], route["direction"])
        values = stats[key]
        if values["high_coverage_histogram_count"] >= MIN_FREE_FLOW_OBSERVATIONS:
            histogram = values["high_coverage_histogram"]
            count = values["high_coverage_histogram_count"]
            calibration_rule = "length_coverage_ge_0.95"
        else:
            histogram = values["all_histogram"]
            count = values["all_histogram_count"]
            calibration_rule = "fallback_all_pass_bins"
        p50 = histogram_percentile(histogram, 0.50)
        p85 = histogram_percentile(histogram, FREE_FLOW_PERCENTILE)
        free_flow[key] = p85
        rows.append(
            {
                "segment_id": key[0],
                "direction": key[1],
                "from_station_no": route["from_station_no"],
                "from_station_name": route["from_station_name"],
                "to_station_no": route["to_station_no"],
                "to_station_name": route["to_station_name"],
                "calibration_observation_count": count,
                "calibration_rule": calibration_rule,
                "median_speed_kmh": f"{p50:.1f}",
                "free_flow_percentile": FREE_FLOW_PERCENTILE,
                "free_flow_speed_kmh": f"{p85:.1f}",
            }
        )
    return free_flow, rows


def write_final_index(free_flow: dict[tuple[str, str], float]) -> tuple[int, int, int]:
    output_header = [
        "service_date", "time_bin", "bin_index", "segment_id", "direction",
        "from_station_no", "from_station_name", "to_station_no", "to_station_name",
        "segment_speed_kmh", "free_flow_speed_kmh", "speed_index_raw", "speed_index",
        "mapped_length_m", "observed_length_m", "length_coverage", "observed_link_count",
        "source_5min_count", "historical_alias_group_count", "quality_status",
    ]
    rows = 0
    pass_rows = 0
    invalid_index_rows = 0
    with gzip.open(PREINDEX_PATH, "rt", encoding="utf-8", newline="") as source, gzip.open(
        OUTPUT_PATH, "wt", encoding="utf-8", newline="", compresslevel=1
    ) as output:
        source.readline()
        output.write(",".join(output_header) + "\n")
        for line in source:
            fields = line.rstrip("\r\n").split(",")
            rows += 1
            key = (fields[3], fields[4])
            ffs = free_flow[key]
            speed_text = fields[9]
            status = fields[16]
            if status == "PASS":
                speed = float(speed_text)
                raw_index = speed / ffs
                speed_index = min(max(raw_index, 0.0), 1.0)
                pass_rows += 1
                if not 0.0 <= speed_index <= 1.0:
                    invalid_index_rows += 1
                raw_index_text = f"{raw_index:.6f}"
                speed_index_text = f"{speed_index:.6f}"
            else:
                raw_index_text = ""
                speed_index_text = ""
            output_fields = fields[:10] + [
                f"{ffs:.1f}", raw_index_text, speed_index_text,
                fields[10], fields[11], fields[12], fields[13], fields[14], fields[15], status,
            ]
            output.write(",".join(output_fields) + "\n")
    return rows, pass_rows, invalid_index_rows


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    routes, excluded = load_routes()
    stats = initialize_stats(routes)

    spatial_rows = [
        {
            "segment_id": route["segment_id"],
            "segment_order": route["segment_order"],
            "direction": route["direction"],
            "branch": route["branch"],
            "from_station_no": route["from_station_no"],
            "from_station_name": route["from_station_name"],
            "to_station_no": route["to_station_no"],
            "to_station_name": route["to_station_name"],
            "mapped_link_count": len(route["links"]),
            "mapped_length_m": f"{route['total_length_m']:.3f}",
        }
        for route in routes
    ]
    write_dict_rows(OUTPUT_DIR / "spatial_units_89.csv", spatial_rows)
    write_dict_rows(OUTPUT_DIR / "excluded_spatial_units.csv", excluded)

    print(f"Starting 15-minute aggregation for {len(routes)} spatial units", flush=True)
    input_rows, preindex_rows = calculate_preindex(routes, stats)
    free_flow, free_flow_rows = calculate_free_flow(routes, stats)
    write_dict_rows(OUTPUT_DIR / "segment_direction_free_flow_speed.csv", free_flow_rows)
    final_rows, pass_rows, invalid_index_rows = write_final_index(free_flow)

    expected_bins_per_unit = preindex_rows // len(routes)
    quality_rows: list[dict[str, Any]] = []
    for route in routes:
        key = (route["segment_id"], route["direction"])
        values = stats[key]
        valid_rate = values["pass_bins"] / expected_bins_per_unit
        qc_status = (
            "PASS"
            if values["days_with_at_least_90_pass_bins"] >= MIN_VALID_DAYS_FOR_SPATIAL_QC
            and values["all_histogram_count"] >= MIN_FREE_FLOW_OBSERVATIONS
            else "REVIEW"
        )
        quality_rows.append(
            {
                "segment_id": key[0],
                "direction": key[1],
                "from_station_no": route["from_station_no"],
                "from_station_name": route["from_station_name"],
                "to_station_no": route["to_station_no"],
                "to_station_name": route["to_station_name"],
                "expected_15min_bin_count": expected_bins_per_unit,
                "pass_bin_count": values["pass_bins"],
                "pass_bin_rate": f"{valid_rate:.6f}",
                "no_data_bin_count": values["no_data_bins"],
                "low_coverage_bin_count": values["low_coverage_bins"],
                "mean_length_coverage": f"{values['coverage_sum'] / expected_bins_per_unit:.6f}",
                "days_with_96_pass_bins": values["days_with_96_pass_bins"],
                "days_with_at_least_90_pass_bins": values["days_with_at_least_90_pass_bins"],
                "pass_bins_using_historical_alias": values[
                    "pass_bins_using_historical_alias"
                ],
                "free_flow_speed_kmh": f"{free_flow[key]:.1f}",
                "quality_status": qc_status,
            }
        )
    write_dict_rows(OUTPUT_DIR / "segment_direction_15min_quality.csv", quality_rows)

    review_rows = [row for row in quality_rows if row["quality_status"] != "PASS"]
    alias_pass_bins = sum(
        int(row["pass_bins_using_historical_alias"]) for row in quality_rows
    )
    report = [
        "# 15분 단위 속도지수 산출 결과",
        "",
        f"- DTW 학습 대상 공간 단위: {len(routes)}개",
        f"- 제외 공간 단위: {excluded[0]['segment_id']} {excluded[0]['direction']}",
        f"- 입력 5분 관측치: {input_rows:,}개",
        f"- 전체 15분 격자: {final_rows:,}개",
        f"- 유효 속도지수: {pass_rows:,}개 ({pass_rows / final_rows:.1%})",
        f"- 결측 또는 링크 길이 커버리지 미달: {final_rows - pass_rows:,}개",
        f"- 과거 링크 버전 브리지 사용 유효 격자: {alias_pass_bins:,}개",
        f"- 구간·방향 품질 PASS: {len(quality_rows) - len(review_rows)}/{len(quality_rows)}",
        f"- 공간 단위 PASS 기준: 일 96개 중 {MIN_VALID_BINS_PER_DAY}개 이상 유효한 날 {MIN_VALID_DAYS_FOR_SPATIAL_QC}일 이상",
        f"- 속도지수 범위 오류: {invalid_index_rows:,}개",
        "",
        "## 산식",
        "",
        "- 링크별 15분 속도: 해당 구간에 포함된 유효 5분 속도의 중앙값",
        f"- 링크 버전 전환: 현재 분할 링크 묶음의 길이 커버리지가 {MIN_LENGTH_COVERAGE:.0%} 미만이면 HISTREMARK 과거 상위 링크를 묶음 전체 길이의 대체 관측으로 사용",
        "- 구간 15분 속도: 관측 링크 길이로 계산한 길이 가중 조화평균",
        f"- 유효 구간 기준: 매핑 길이 커버리지 {MIN_LENGTH_COVERAGE:.0%} 이상",
        f"- 자유류 속도: 길이 커버리지 {FREE_FLOW_CALIBRATION_COVERAGE:.0%} 이상인 유효 구간속도의 {FREE_FLOW_PERCENTILE:.0%} 백분위수",
        "- 속도지수 원값: 구간속도 / 자유류속도",
        "- 모델 입력 속도지수: 원값을 0~1 범위로 절단",
        "- 결측값은 보간하지 않음",
        "",
    ]
    if review_rows:
        report.extend(["## 추가 검토 대상", ""])
        for row in review_rows:
            report.append(
                f"- {row['segment_id']} {row['direction']}: pass_bin_rate="
                f"{float(row['pass_bin_rate']):.1%}, "
                f"days_with_at_least_{MIN_VALID_BINS_PER_DAY}_bins="
                f"{row['days_with_at_least_90_pass_bins']}"
            )
    else:
        report.extend(["## 판정", "", "89개 공간 단위가 모두 15분 품질 기준을 통과했다."])
    (OUTPUT_DIR / "speed_index_15min_qc_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    PREINDEX_PATH.unlink(missing_ok=True)

    if preindex_rows != final_rows or invalid_index_rows:
        raise RuntimeError("15-minute output validation failed")
    print(f"Wrote {OUTPUT_DIR}", flush=True)
    print(
        f"rows={final_rows} pass={pass_rows} spatial_qc_pass={len(quality_rows) - len(review_rows)}/89",
        flush=True,
    )


if __name__ == "__main__":
    main()
