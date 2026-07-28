#!/usr/bin/env python3
"""Create non-overlapping daily 96-bin DTW input windows."""

from __future__ import annotations

import csv
import datetime as dt
import gzip
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
INPUT_PATH = (
    HERE
    / "outputs"
    / "03_speed_index_15min"
    / "segment_speed_index_15min.csv.gz"
)
OUTPUT_DIR = HERE / "outputs" / "04_daily_96_windows"
WINDOW_PATH = OUTPUT_DIR / "daily_speed_index_windows.csv.gz"
QUALITY_PATH = OUTPUT_DIR / "daily_window_quality.csv.gz"
UNIT_SUMMARY_PATH = OUTPUT_DIR / "daily_window_unit_summary.csv"
REPORT_PATH = OUTPUT_DIR / "daily_window_qc_report.md"

START_DATE = dt.date(2024, 10, 1)
END_DATE = dt.date(2026, 7, 1)
EXPECTED_DAYS = (END_DATE - START_DATE).days + 1
EXPECTED_SPATIAL_UNITS = 89
EXPECTED_BINS = 96
EXPECTED_CANDIDATE_WINDOWS = EXPECTED_DAYS * EXPECTED_SPATIAL_UNITS
MIN_COMPLETE_WINDOWS_PER_UNIT = 90

TIME_LABELS = [f"{minute // 60:02d}{minute % 60:02d}" for minute in range(0, 1440, 15)]
SPEED_COLUMNS = [f"speed_index_{label}" for label in TIME_LABELS]
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

WINDOW_HEADER = [
    "sample_id",
    "service_date",
    "year_month",
    "day_of_week",
    "day_name",
    "is_weekend",
    "segment_id",
    "direction",
    "from_station_no",
    "from_station_name",
    "to_station_no",
    "to_station_name",
    "window_start",
    "window_end",
    "interval_minutes",
    "bin_count",
] + SPEED_COLUMNS

QUALITY_HEADER = [
    "sample_id",
    "service_date",
    "segment_id",
    "direction",
    "from_station_no",
    "from_station_name",
    "to_station_no",
    "to_station_name",
    "expected_bin_count",
    "valid_bin_count",
    "missing_bin_count",
    "valid_bin_rate",
    "no_data_bin_count",
    "low_link_coverage_bin_count",
    "longest_missing_run",
    "window_status",
    "exclusion_reason",
]


def window_key(row: dict[str, str]) -> tuple[str, str, str]:
    return row["service_date"], row["segment_id"], row["direction"]


def iter_windows(rows: Iterable[dict[str, str]]) -> Iterable[list[dict[str, str]]]:
    current_key: tuple[str, str, str] | None = None
    current_rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        key = window_key(row)
        if current_key is None:
            current_key = key
        elif key != current_key:
            if current_key in seen:
                raise RuntimeError(f"Non-contiguous duplicate window key: {current_key}")
            seen.add(current_key)
            yield current_rows
            current_key = key
            current_rows = []
        current_rows.append(row)
    if current_rows:
        assert current_key is not None
        if current_key in seen:
            raise RuntimeError(f"Duplicate window key: {current_key}")
        yield current_rows


def longest_true_run(values: list[bool]) -> int:
    longest = 0
    current = 0
    for value in values:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def parse_service_date(value: str) -> dt.date:
    date = dt.datetime.strptime(value, "%Y%m%d").date()
    if not START_DATE <= date <= END_DATE:
        raise RuntimeError(f"Date outside modeling period: {value}")
    return date


def validate_window(rows: list[dict[str, str]]) -> None:
    if len(rows) != EXPECTED_BINS:
        raise RuntimeError(f"Window {window_key(rows[0])} has {len(rows)} rows")
    expected_indices = list(range(EXPECTED_BINS))
    actual_indices = [int(row["bin_index"]) for row in rows]
    if actual_indices != expected_indices:
        raise RuntimeError(f"Invalid bin order for {window_key(rows[0])}")
    actual_times = [row["time_bin"] for row in rows]
    if actual_times != TIME_LABELS:
        raise RuntimeError(f"Invalid 15-minute labels for {window_key(rows[0])}")


def write_unit_summary(rows: list[dict[str, Any]]) -> None:
    with UNIT_SUMMARY_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    status_counts: Counter[str] = Counter()
    complete_dates: set[str] = set()
    unit_stats: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    candidate_count = 0
    complete_count = 0
    duplicate_sample_ids: set[str] = set()
    sample_ids: set[str] = set()

    with gzip.open(INPUT_PATH, "rt", encoding="utf-8", newline="") as source, gzip.open(
        WINDOW_PATH, "wt", encoding="utf-8", newline="", compresslevel=1
    ) as window_handle, gzip.open(
        QUALITY_PATH, "wt", encoding="utf-8", newline="", compresslevel=1
    ) as quality_handle:
        reader = csv.DictReader(source)
        window_writer = csv.writer(window_handle)
        quality_writer = csv.writer(quality_handle)
        window_writer.writerow(WINDOW_HEADER)
        quality_writer.writerow(QUALITY_HEADER)

        for rows in iter_windows(reader):
            validate_window(rows)
            candidate_count += 1
            first = rows[0]
            service_date = first["service_date"]
            date = parse_service_date(service_date)
            segment_id = first["segment_id"]
            direction = first["direction"]
            unit_key = (segment_id, direction)
            sample_id = f"{service_date}_{segment_id}_{direction}"
            if sample_id in sample_ids:
                duplicate_sample_ids.add(sample_id)
            sample_ids.add(sample_id)

            missing = [
                row["quality_status"] != "PASS" or row["speed_index"] == ""
                for row in rows
            ]
            missing_count = sum(missing)
            valid_count = EXPECTED_BINS - missing_count
            source_statuses = Counter(row["quality_status"] for row in rows)
            if missing_count == 0:
                window_status = "PASS_COMPLETE"
                exclusion_reason = ""
                complete_count += 1
                complete_dates.add(service_date)
                unit_stats[unit_key]["complete"] += 1
                vector = [row["speed_index"] for row in rows]
                if any(value == "" or not 0.0 <= float(value) <= 1.0 for value in vector):
                    raise RuntimeError(f"Invalid speed-index vector: {sample_id}")
                window_writer.writerow(
                    [
                        sample_id,
                        service_date,
                        service_date[:6],
                        date.weekday(),
                        DAY_NAMES[date.weekday()],
                        int(date.weekday() >= 5),
                        segment_id,
                        direction,
                        first["from_station_no"],
                        first["from_station_name"],
                        first["to_station_no"],
                        first["to_station_name"],
                        "0000",
                        "2345",
                        15,
                        EXPECTED_BINS,
                    ]
                    + vector
                )
            elif missing_count == EXPECTED_BINS:
                window_status = "EXCLUDE_NO_DATA"
                exclusion_reason = "ALL_96_BINS_MISSING"
                unit_stats[unit_key]["all_missing"] += 1
            else:
                window_status = "EXCLUDE_INCOMPLETE"
                exclusion_reason = "MISSING_15MIN_BINS"
                unit_stats[unit_key]["incomplete"] += 1

            status_counts[window_status] += 1
            unit_stats[unit_key]["candidate"] += 1
            unit_stats[unit_key]["valid_bins"] += valid_count
            unit_stats[unit_key]["missing_bins"] += missing_count
            quality_writer.writerow(
                [
                    sample_id,
                    service_date,
                    segment_id,
                    direction,
                    first["from_station_no"],
                    first["from_station_name"],
                    first["to_station_no"],
                    first["to_station_name"],
                    EXPECTED_BINS,
                    valid_count,
                    missing_count,
                    f"{valid_count / EXPECTED_BINS:.6f}",
                    source_statuses["NO_DATA"],
                    source_statuses["LOW_LINK_COVERAGE"],
                    longest_true_run(missing),
                    window_status,
                    exclusion_reason,
                ]
            )

            if candidate_count % 10_000 == 0:
                print(f"Processed {candidate_count:,} candidate windows", flush=True)

    if duplicate_sample_ids:
        raise RuntimeError(f"Duplicate sample IDs: {sorted(duplicate_sample_ids)[:5]}")
    if candidate_count != EXPECTED_CANDIDATE_WINDOWS:
        raise RuntimeError(
            f"Expected {EXPECTED_CANDIDATE_WINDOWS:,} windows, got {candidate_count:,}"
        )
    if len(unit_stats) != EXPECTED_SPATIAL_UNITS:
        raise RuntimeError(f"Expected 89 spatial units, got {len(unit_stats)}")

    unit_rows: list[dict[str, Any]] = []
    for (segment_id, direction), values in sorted(unit_stats.items()):
        complete_windows = values["complete"]
        unit_rows.append(
            {
                "segment_id": segment_id,
                "direction": direction,
                "candidate_window_count": values["candidate"],
                "complete_window_count": complete_windows,
                "complete_window_rate": f"{complete_windows / values['candidate']:.6f}",
                "incomplete_window_count": values["incomplete"],
                "all_missing_window_count": values["all_missing"],
                "valid_bin_count": values["valid_bins"],
                "missing_bin_count": values["missing_bins"],
                "window_qc_status": (
                    "PASS"
                    if complete_windows >= MIN_COMPLETE_WINDOWS_PER_UNIT
                    else "REVIEW"
                ),
            }
        )
    write_unit_summary(unit_rows)

    review_units = [row for row in unit_rows if row["window_qc_status"] != "PASS"]
    minimum_complete = min(int(row["complete_window_count"]) for row in unit_rows)
    maximum_complete = max(int(row["complete_window_count"]) for row in unit_rows)
    report = [
        "# 96개 비중첩 일별 윈도우 생성 결과",
        "",
        f"- 기간: {START_DATE.isoformat()} ~ {END_DATE.isoformat()} ({EXPECTED_DAYS}일)",
        f"- DTW 학습 대상 공간 단위: {len(unit_rows)}개",
        f"- 전체 후보 일별 윈도우: {candidate_count:,}개",
        f"- 완전한 96칸 DTW 입력 윈도우: {complete_count:,}개 ({complete_count / candidate_count:.1%})",
        f"- 일부 15분 격자 결측으로 제외: {status_counts['EXCLUDE_INCOMPLETE']:,}개",
        f"- 96개 격자 전체 결측으로 제외: {status_counts['EXCLUDE_NO_DATA']:,}개",
        f"- 완전 윈도우가 하나 이상 있는 날짜: {len(complete_dates)}일",
        f"- 공간 단위별 완전 윈도우: 최소 {minimum_complete}개, 최대 {maximum_complete}개",
        f"- 공간 단위 PASS 기준: 완전 윈도우 {MIN_COMPLETE_WINDOWS_PER_UNIT}개 이상",
        f"- 공간 단위 품질 PASS: {len(unit_rows) - len(review_units)}/{len(unit_rows)}",
        "",
        "## 생성 규칙",
        "",
        "- 하나의 윈도우는 동일한 service_date × segment_id × direction의 00:00~23:45 15분 속도지수 96개로 구성",
        "- 날짜 간 또는 공간 단위 간 중첩 없음",
        "- 96개 속도지수가 모두 유효한 윈도우만 DTW 입력 파일에 저장",
        "- 하나라도 결측이면 보간하지 않고 품질 파일에 제외 사유와 결측 개수를 기록",
        "- 속도지수 열 순서는 speed_index_0000부터 speed_index_2345까지 고정",
        "",
        "## 판정",
        "",
    ]
    if review_units:
        report.append(
            f"{len(review_units)}개 공간 단위가 완전 윈도우 최소 기준에 미달해 후속 검토가 필요하다."
        )
    else:
        report.append("89개 공간 단위가 모두 일별 윈도우 표본 수 기준을 통과했다.")
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"Wrote {OUTPUT_DIR}", flush=True)
    print(
        f"candidates={candidate_count} complete={complete_count} "
        f"spatial_qc_pass={len(unit_rows) - len(review_units)}/{len(unit_rows)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
