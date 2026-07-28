#!/usr/bin/env python3
"""Filter 5-minute ITS speeds for the validated tram-link mapping."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import os
import re
import shutil
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import median
from typing import Any

from link_version_bridge import (
    build_bridge_rows,
    historical_alias_speeds,
    load_link_history,
)


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
MAPPING = HERE / "outputs" / "01_mapping_qc" / "segment_link_mapping.csv"
AWK_SCRIPT = HERE / "filter_5min.awk"
OUTPUT_DIR = HERE / "outputs" / "02_speed_5min_qc"
START_DATE = dt.date(2024, 10, 1)
END_DATE = dt.date(2026, 7, 1)
MIN_SOURCE_COVERAGE = 0.80
MIN_DAILY_SLOTS = 144

PASS_HEADER = "service_date,time_hhmm,link_id,speed_kmh,speed_cap_kmh,duplicate_count,quality_status\n"
REJECT_HEADER = "service_date,time_hhmm,link_id,raw_speed_kmh,reject_reason\n"
DAILY_HEADER = [
    "service_date", "link_id", "source_slots", "valid_before_temporal_filter",
    "physical_reject_count", "temporal_spike_count", "kept_slot_count",
    "coverage_of_source_slots", "coverage_of_288_slots", "daily_qc_pass",
]
FILE_HEADER = [
    "service_date", "raw_row_count", "mapped_row_count", "source_slot_count",
    "mapped_link_count", "daily_pass_link_count", "kept_observation_count",
    "rejected_observation_count", "physical_reject_count", "temporal_spike_count",
    "low_coverage_reject_count", "duplicate_row_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 2))
    return parser.parse_args()


def locate_speed_files() -> list[tuple[str, Path]]:
    pattern = re.compile(r"^(\d{8})_daejeon\.csv\.gz$")
    files: list[tuple[str, Path]] = []
    for path in (WORKSPACE / "raw_data").rglob("*_daejeon.csv.gz"):
        match = pattern.match(path.name)
        if not match:
            continue
        service_date = dt.datetime.strptime(match.group(1), "%Y%m%d").date()
        if START_DATE <= service_date <= END_DATE:
            files.append((match.group(1), path))
    files.sort()
    expected_days = (END_DATE - START_DATE).days + 1
    if len(files) != expected_days:
        present = {date_text for date_text, _ in files}
        missing = [
            (START_DATE + dt.timedelta(days=offset)).strftime("%Y%m%d")
            for offset in range(expected_days)
            if (START_DATE + dt.timedelta(days=offset)).strftime("%Y%m%d") not in present
        ]
        raise RuntimeError(f"Expected {expected_days} daily files, found {len(files)}; missing={missing[:10]}")
    return files


def load_mapping() -> tuple[
    list[dict[str, str]],
    dict[str, dict[str, float]],
    list[dict[str, Any]],
    set[str],
]:
    if not MAPPING.exists():
        raise FileNotFoundError(f"Run 01_validate_segment_link_mapping.py first: {MAPPING}")
    with MAPPING.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    link_info: dict[str, dict[str, float]] = {}
    for row in rows:
        link_id = row["link_id"]
        max_speed = float(row["max_speed_kmh"] or 0)
        length_m = float(row["length_m"])
        info = link_info.setdefault(link_id, {"max_speed": 0.0, "length_m": length_m})
        info["max_speed"] = max(info["max_speed"], max_speed)
    for info in link_info.values():
        info["cap"] = min(150.0, max(80.0, info["max_speed"] * 1.5))
    metadata = load_link_history(rows)
    bridge_rows = build_bridge_rows(rows, metadata)
    alias_speeds = historical_alias_speeds(rows, metadata)
    current_ids = set(link_info)
    external_aliases = set(alias_speeds) - current_ids
    for link_id, max_speed in alias_speeds.items():
        info = link_info.setdefault(link_id, {"max_speed": max_speed, "length_m": 0.0})
        info["max_speed"] = max(info["max_speed"], max_speed)
        info["cap"] = min(150.0, max(80.0, info["max_speed"] * 1.5))
    return rows, link_info, bridge_rows, external_aliases


def write_caps(path: Path, link_info: dict[str, dict[str, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("link_id\tspeed_cap_kmh\n")
        for link_id in sorted(link_info):
            handle.write(f"{link_id}\t{link_info[link_id]['cap']:.1f}\n")


def run_one_day(date_text: str, source: Path, caps: Path, chunks: Path) -> tuple[str, int]:
    pass_path = chunks / f"{date_text}_pass.csv.gz"
    reject_path = chunks / f"{date_text}_reject.csv"
    daily_path = chunks / f"{date_text}_daily.csv"
    file_path = chunks / f"{date_text}_file.csv"
    awk_command = [
        "awk",
        "-v", f"expected_date={date_text}",
        "-v", f"reject_path={reject_path}",
        "-v", f"daily_summary_path={daily_path}",
        "-v", f"file_summary_path={file_path}",
        "-v", f"min_source_coverage={MIN_SOURCE_COVERAGE}",
        "-v", f"min_daily_slots={MIN_DAILY_SLOTS}",
        "-f", str(AWK_SCRIPT),
        str(caps),
        "-",
    ]
    with pass_path.open("wb") as pass_handle:
        decompress = subprocess.Popen(["gzip", "-cd", str(source)], stdout=subprocess.PIPE)
        assert decompress.stdout is not None
        filter_process = subprocess.Popen(awk_command, stdin=decompress.stdout, stdout=subprocess.PIPE)
        decompress.stdout.close()
        assert filter_process.stdout is not None
        compress = subprocess.Popen(["gzip", "-c"], stdin=filter_process.stdout, stdout=pass_handle)
        filter_process.stdout.close()
        compress_status = compress.wait()
        filter_status = filter_process.wait()
        decompress_status = decompress.wait()
    if decompress_status or filter_status or compress_status:
        raise RuntimeError(
            f"Pipeline failed for {date_text}: gzip={decompress_status}, awk={filter_status}, output_gzip={compress_status}"
        )
    return date_text, pass_path.stat().st_size


def append_gzip_members(destination: Path, header: str, sources: list[Path]) -> None:
    with destination.open("wb") as output:
        output.write(gzip.compress(header.encode("utf-8")))
        for source in sources:
            with source.open("rb") as input_handle:
                shutil.copyfileobj(input_handle, output)


def aggregate_text_to_gzip(destination: Path, header: str, sources: list[Path]) -> None:
    with gzip.open(destination, "wt", encoding="utf-8", newline="") as output:
        output.write(header)
        for source in sources:
            if source.exists():
                with source.open(encoding="utf-8", newline="") as input_handle:
                    shutil.copyfileobj(input_handle, output)


def aggregate_csv(destination: Path, header: list[str], sources: list[Path]) -> None:
    with destination.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(header)
        for source in sources:
            with source.open(encoding="utf-8", newline="") as input_handle:
                for row in csv.reader(input_handle):
                    writer.writerow(row)


def summarize_links(daily_gz: Path, total_days: int) -> list[dict[str, Any]]:
    aggregate: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    with gzip.open(daily_gz, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            link = row["link_id"]
            values = aggregate[link]
            kept = int(row["kept_slot_count"])
            values["days_present"] += kept > 0
            values["days_pass"] += int(row["daily_qc_pass"])
            values["kept_slots"] += kept if int(row["daily_qc_pass"]) else 0
            values["physical_rejects"] += int(row["physical_reject_count"])
            values["temporal_spikes"] += int(row["temporal_spike_count"])
            values["coverage_source_sum"] += float(row["coverage_of_source_slots"])
    rows = []
    for link_id, values in sorted(aggregate.items()):
        rows.append(
            {
                "link_id": link_id,
                "total_days": total_days,
                "days_with_observations": int(values["days_present"]),
                "days_qc_pass": int(values["days_pass"]),
                "day_pass_rate": values["days_pass"] / total_days,
                "kept_slot_count": int(values["kept_slots"]),
                "physical_reject_count": int(values["physical_rejects"]),
                "temporal_spike_count": int(values["temporal_spikes"]),
                "mean_source_slot_coverage": values["coverage_source_sum"] / total_days,
            }
        )
    return rows


def summarize_segments(
    mapping_rows: list[dict[str, str]], link_rows: list[dict[str, Any]], total_days: int
) -> list[dict[str, Any]]:
    link_stats = {row["link_id"]: row for row in link_rows}
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in mapping_rows:
        groups[(row["segment_id"], row["direction"])].append(row)
    result = []
    for (segment_id, direction), rows in sorted(groups.items()):
        total_length = sum(float(row["length_m"]) for row in rows)
        any_length = sum(
            float(row["length_m"])
            for row in rows
            if link_stats[row["link_id"]]["days_with_observations"] > 0
        )
        stable_length = sum(
            float(row["length_m"])
            for row in rows
            if link_stats[row["link_id"]]["days_qc_pass"] / total_days >= 0.80
        )
        first = rows[0]
        result.append(
            {
                "segment_id": segment_id,
                "direction": direction,
                "from_station_no": first["from_station_no"],
                "from_station_name": first["from_station_name"],
                "to_station_no": first["to_station_no"],
                "to_station_name": first["to_station_name"],
                "mapped_link_count": len(rows),
                "mapped_length_m": total_length,
                "ever_observed_length_coverage": any_length / total_length if total_length else 0,
                "stable_length_coverage": stable_length / total_length if total_length else 0,
                "speed_coverage_qc": "PASS" if any_length / total_length >= 0.80 else "REVIEW",
            }
        )
    return result


def write_dict_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    files = locate_speed_files()
    mapping_rows, link_info, bridge_rows, external_aliases = load_mapping()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    chunks = OUTPUT_DIR / "_chunks"
    if chunks.exists():
        shutil.rmtree(chunks)
    chunks.mkdir()
    caps = OUTPUT_DIR / "mapped_link_speed_caps.tsv"
    write_caps(caps, link_info)
    write_dict_rows(OUTPUT_DIR / "link_version_bridge.csv", bridge_rows)

    print(
        f"Starting 5-minute filtering: days={len(files)} current_links="
        f"{len(link_info) - len(external_aliases)} historical_aliases={len(external_aliases)} "
        f"filter_links={len(link_info)} workers={args.workers}",
        flush=True,
    )
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_one_day, date_text, path, caps, chunks): date_text
            for date_text, path in files
        }
        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed % 20 == 0 or completed == len(files):
                print(f"Processed {completed}/{len(files)} days", flush=True)

    dates = [date_text for date_text, _ in files]
    append_gzip_members(
        OUTPUT_DIR / "speed_5min_filtered.csv.gz",
        PASS_HEADER,
        [chunks / f"{date}_pass.csv.gz" for date in dates],
    )
    aggregate_text_to_gzip(
        OUTPUT_DIR / "speed_5min_rejected.csv.gz",
        REJECT_HEADER,
        [chunks / f"{date}_reject.csv" for date in dates],
    )
    daily_path = OUTPUT_DIR / "daily_link_quality.csv.gz"
    aggregate_text_to_gzip(
        daily_path,
        ",".join(DAILY_HEADER) + "\n",
        [chunks / f"{date}_daily.csv" for date in dates],
    )
    file_path = OUTPUT_DIR / "daily_file_quality.csv"
    aggregate_csv(file_path, FILE_HEADER, [chunks / f"{date}_file.csv" for date in dates])

    link_rows = summarize_links(daily_path, len(files))
    segment_rows = summarize_segments(mapping_rows, link_rows, len(files))
    write_dict_rows(OUTPUT_DIR / "link_speed_quality_summary.csv", link_rows)
    write_dict_rows(OUTPUT_DIR / "segment_direction_speed_coverage.csv", segment_rows)

    with file_path.open(encoding="utf-8-sig", newline="") as handle:
        file_rows = list(csv.DictReader(handle))
    totals = {
        field: sum(int(row[field]) for row in file_rows)
        for field in FILE_HEADER[1:]
        if field != "source_slot_count"
    }
    source_slots = [int(row["source_slot_count"]) for row in file_rows]
    review_segments = [row for row in segment_rows if row["speed_coverage_qc"] != "PASS"]
    pass_links = sum(row["days_qc_pass"] > 0 for row in link_rows)
    stable_links = sum(row["day_pass_rate"] >= 0.80 for row in link_rows)
    report = [
        "# 5분 단위 속도 품질 필터링 결과",
        "",
        f"- 기간: {START_DATE.isoformat()} ~ {END_DATE.isoformat()} ({len(files):,}일)",
        f"- 현재 매핑 고유 링크: {len(link_info) - len(external_aliases):,}개",
        f"- 추가 필터링한 외부 과거 링크 ID: {len(external_aliases):,}개",
        f"- 현재-과거 링크 이력 연결: {len(bridge_rows):,}행",
        f"- 한 번 이상 일별 QC 통과 링크: {pass_links:,}개",
        f"- 전체 기간의 80% 이상 일별 QC 통과 링크: {stable_links:,}개",
        f"- 원시 행: {totals['raw_row_count']:,}개",
        f"- 매핑 링크 원시 행: {totals['mapped_row_count']:,}개",
        f"- 최종 보존 관측치: {totals['kept_observation_count']:,}개",
        f"- 물리·형식 오류 제거: {totals['physical_reject_count']:,}개",
        f"- 고립 시간 스파이크 제거: {totals['temporal_spike_count']:,}개",
        f"- 낮은 일별 링크 완전성으로 제거: {totals['low_coverage_reject_count']:,}개",
        f"- 중복 원시 행(평균 병합): {totals['duplicate_row_count']:,}개",
        f"- 원천 일별 5분 슬롯 수: min={min(source_slots)}, median={median(source_slots):g}, max={max(source_slots)} (이론상 288)",
        f"- 속도 커버리지 PASS 구간·방향: {len(segment_rows) - len(review_segments):,}/90",
        "",
        "## 적용 규칙",
        "",
        "- 파일 날짜와 CREATDE 일치, HHMM이 유효한 5분 격자인지 확인",
        "- 속도는 0 초과이며 링크 제한속도 기반 동적 상한(max(80, 1.5×제한속도), 최대 150km/h) 이하",
        "- 동일 링크·시각 중복값은 평균으로 병합",
        "- 앞뒤 관측이 10km/h 이내로 안정적인데 가운데 값만 35km/h 초과 이탈하면 고립 스파이크로 제거",
        f"- 링크·일 관측은 원천 제공 슬롯의 {MIN_SOURCE_COVERAGE:.0%} 이상이면서 최소 {MIN_DAILY_SLOTS}슬롯일 때만 보존",
        "- 결측 슬롯은 보간하지 않음",
        "- 표준 링크 분할 이전 기간을 복원할 수 있도록 HISTREMARK의 과거 상위 링크도 동일 규칙으로 필터링",
        "",
    ]
    if review_segments:
        report.extend(["## 속도 커버리지 추가 검토 대상", ""])
        for row in review_segments:
            report.append(
                f"- {row['segment_id']} {row['direction']}: ever_observed_length_coverage="
                f"{float(row['ever_observed_length_coverage']):.1%}"
            )
    else:
        report.extend(["## 판정", "", "90개 구간·방향 모두 매핑 길이의 80% 이상에서 속도가 관측됐다."])
    (OUTPUT_DIR / "speed_5min_qc_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    shutil.rmtree(chunks)
    print(f"Wrote {OUTPUT_DIR}", flush=True)
    print(
        f"kept={totals['kept_observation_count']} rejected={totals['rejected_observation_count']} "
        f"coverage_pass={len(segment_rows) - len(review_segments)}/90",
        flush=True,
    )


if __name__ == "__main__":
    main()
