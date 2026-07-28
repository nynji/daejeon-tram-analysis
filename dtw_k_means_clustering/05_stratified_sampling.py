#!/usr/bin/env python3
"""Create a balanced, deterministic stratified sample for DTW K-means."""

from __future__ import annotations

import csv
import gzip
import hashlib
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
INPUT_PATH = (
    HERE / "outputs" / "04_daily_96_windows" / "daily_speed_index_windows.csv.gz"
)
OUTPUT_DIR = HERE / "outputs" / "05_stratified_sampling"
TRAIN_PATH = OUTPUT_DIR / "dtw_training_sample.csv.gz"
HOLDOUT_PATH = OUTPUT_DIR / "dtw_time_holdout.csv.gz"
MANIFEST_PATH = OUTPUT_DIR / "sampling_manifest.csv.gz"
UNIT_SUMMARY_PATH = OUTPUT_DIR / "sampling_unit_summary.csv"
STRATUM_SUMMARY_PATH = OUTPUT_DIR / "sampling_stratum_summary.csv"
DISTRIBUTION_PATH = OUTPUT_DIR / "sampling_distribution_summary.csv"
REPORT_PATH = OUTPUT_DIR / "stratified_sampling_qc_report.md"

TIME_HOLDOUT_START = "20260401"
TARGET_TRAIN_SAMPLE = 8_000
SAMPLING_SEED = 20_260_715
EXPECTED_POPULATION = 36_154
EXPECTED_SPATIAL_UNITS = 89
MIN_TRAIN_SAMPLE_PER_UNIT = 30
MIN_HOLDOUT_PER_UNIT = 10

SAMPLING_FIELDS = [
    "dataset_split",
    "sampling_stratum",
    "sampling_seed",
    "train_pool_stratum_count",
    "sampled_stratum_count",
    "inclusion_probability",
]


def unit_key(row: dict[str, str]) -> tuple[str, str]:
    return row["segment_id"], row["direction"]


def season(month: int) -> str:
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    if month in (9, 10, 11):
        return "autumn"
    return "winter"


def stratum_key(row: dict[str, str]) -> tuple[tuple[str, str], str, str]:
    return unit_key(row), season(int(row["service_date"][4:6])), row["is_weekend"]


def stratum_label(row: dict[str, str]) -> str:
    key = stratum_key(row)
    return f"{key[0][0]}|{key[0][1]}|{key[1]}|weekend_{key[2]}"


def deterministic_rank(sample_id: str) -> str:
    value = f"{SAMPLING_SEED}|{sample_id}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def allocate_balanced_unit_quotas(
    capacities: dict[tuple[str, str], int], target: int
) -> dict[tuple[str, str], int]:
    if target > sum(capacities.values()):
        raise RuntimeError("Training sample target exceeds the training pool")
    quotas = {key: 0 for key in capacities}
    remaining = target
    ordered = sorted(capacities)
    while remaining:
        progressed = False
        for key in ordered:
            if remaining == 0:
                break
            if quotas[key] >= capacities[key]:
                continue
            quotas[key] += 1
            remaining -= 1
            progressed = True
        if not progressed:
            raise RuntimeError("Unable to allocate the full training sample")
    return quotas


def allocate_proportional_strata(
    capacities: dict[tuple[tuple[str, str], str, str], int], target: int
) -> dict[tuple[tuple[str, str], str, str], int]:
    if target > sum(capacities.values()):
        raise RuntimeError("Stratum target exceeds its unit pool")
    keys = sorted(capacities)
    quotas = {key: 0 for key in keys}
    if target == 0:
        return quotas

    # Every non-empty seasonal/weekend stratum receives one record when possible.
    if target >= len(keys):
        for key in keys:
            quotas[key] = 1
        remaining = target - len(keys)
    else:
        for key in sorted(keys, key=lambda value: (-capacities[value], value))[:target]:
            quotas[key] = 1
        return quotas

    remaining_capacity = {key: capacities[key] - quotas[key] for key in keys}
    total_remaining_capacity = sum(remaining_capacity.values())
    if remaining == 0:
        return quotas

    exact: dict[tuple[tuple[str, str], str, str], float] = {}
    assigned = 0
    for key in keys:
        value = remaining * remaining_capacity[key] / total_remaining_capacity
        exact[key] = value
        addition = min(remaining_capacity[key], math.floor(value))
        quotas[key] += addition
        assigned += addition

    left = remaining - assigned
    for key in sorted(
        keys,
        key=lambda value: (-(exact[value] - math.floor(exact[value])), value),
    ):
        if left == 0:
            break
        if quotas[key] >= capacities[key]:
            continue
        quotas[key] += 1
        left -= 1

    if left:
        for key in keys:
            while left and quotas[key] < capacities[key]:
                quotas[key] += 1
                left -= 1
    if left or sum(quotas.values()) != target:
        raise RuntimeError("Proportional stratum allocation failed")
    return quotas


def insert_sampling_fields(
    row: dict[str, str],
    original_header: list[str],
    split: str,
    pool_count: int | None,
    sampled_count: int | None,
) -> list[str]:
    probability = (
        ""
        if pool_count is None or sampled_count is None
        else f"{sampled_count / pool_count:.6f}"
    )
    metadata = [
        split,
        stratum_label(row),
        str(SAMPLING_SEED),
        "" if pool_count is None else str(pool_count),
        "" if sampled_count is None else str(sampled_count),
        probability,
    ]
    return (
        [row[field] for field in original_header[:16]]
        + metadata
        + [row[field] for field in original_header[16:]]
    )


def write_dict_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def distribution_rows(
    train_pool: list[dict[str, str]],
    train_sample: list[dict[str, str]],
    holdout: list[dict[str, str]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    dimensions = {
        "season": lambda row: season(int(row["service_date"][4:6])),
        "is_weekend": lambda row: row["is_weekend"],
    }
    for dimension, getter in dimensions.items():
        pool_counts = Counter(getter(row) for row in train_pool)
        sample_counts = Counter(getter(row) for row in train_sample)
        holdout_counts = Counter(getter(row) for row in holdout)
        categories = sorted(set(pool_counts) | set(sample_counts) | set(holdout_counts))
        for category in categories:
            pool_rate = pool_counts[category] / len(train_pool)
            sample_rate = sample_counts[category] / len(train_sample)
            result.append(
                {
                    "dimension": dimension,
                    "category": category,
                    "train_pool_count": pool_counts[category],
                    "train_pool_rate": f"{pool_rate:.6f}",
                    "train_sample_count": sample_counts[category],
                    "train_sample_rate": f"{sample_rate:.6f}",
                    "absolute_rate_difference": f"{abs(sample_rate - pool_rate):.6f}",
                    "time_holdout_count": holdout_counts[category],
                    "time_holdout_rate": f"{holdout_counts[category] / len(holdout):.6f}",
                }
            )
    return result


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with gzip.open(INPUT_PATH, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        original_header = list(reader.fieldnames or [])
        population = list(reader)
    if len(population) != EXPECTED_POPULATION:
        raise RuntimeError(f"Expected {EXPECTED_POPULATION:,} windows, got {len(population):,}")
    if len({row["sample_id"] for row in population}) != len(population):
        raise RuntimeError("Population contains duplicate sample IDs")

    train_pool = [row for row in population if row["service_date"] < TIME_HOLDOUT_START]
    holdout = [row for row in population if row["service_date"] >= TIME_HOLDOUT_START]
    train_by_unit: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    holdout_by_unit: Counter[tuple[str, str]] = Counter()
    for row in train_pool:
        train_by_unit[unit_key(row)].append(row)
    for row in holdout:
        holdout_by_unit[unit_key(row)] += 1
    if len(train_by_unit) != EXPECTED_SPATIAL_UNITS or len(holdout_by_unit) != EXPECTED_SPATIAL_UNITS:
        raise RuntimeError("Every spatial unit must be represented in train pool and holdout")

    unit_quotas = allocate_balanced_unit_quotas(
        {key: len(rows) for key, rows in train_by_unit.items()}, TARGET_TRAIN_SAMPLE
    )
    stratum_pool: dict[
        tuple[tuple[str, str], str, str], list[dict[str, str]]
    ] = defaultdict(list)
    for row in train_pool:
        stratum_pool[stratum_key(row)].append(row)

    stratum_quotas: dict[tuple[tuple[str, str], str, str], int] = {}
    for key, unit_rows in sorted(train_by_unit.items()):
        capacities = {
            stratum: len(rows)
            for stratum, rows in stratum_pool.items()
            if stratum[0] == key
        }
        stratum_quotas.update(
            allocate_proportional_strata(capacities, unit_quotas[key])
        )

    selected_ids: set[str] = set()
    selected_by_stratum: dict[
        tuple[tuple[str, str], str, str], list[dict[str, str]]
    ] = {}
    for key, rows in sorted(stratum_pool.items()):
        ordered = sorted(rows, key=lambda row: (deterministic_rank(row["sample_id"]), row["sample_id"]))
        selected = ordered[: stratum_quotas[key]]
        selected_by_stratum[key] = selected
        selected_ids.update(row["sample_id"] for row in selected)
    train_sample = sorted(
        (row for row in train_pool if row["sample_id"] in selected_ids),
        key=lambda row: (row["service_date"], row["segment_id"], row["direction"]),
    )
    holdout.sort(key=lambda row: (row["service_date"], row["segment_id"], row["direction"]))
    if len(train_sample) != TARGET_TRAIN_SAMPLE or len(selected_ids) != TARGET_TRAIN_SAMPLE:
        raise RuntimeError("Training sample size or uniqueness validation failed")

    output_header = original_header[:16] + SAMPLING_FIELDS + original_header[16:]
    with gzip.open(
        TRAIN_PATH, "wt", encoding="utf-8", newline="", compresslevel=1
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(output_header)
        for row in train_sample:
            key = stratum_key(row)
            writer.writerow(
                insert_sampling_fields(
                    row,
                    original_header,
                    "TRAIN_SAMPLE",
                    len(stratum_pool[key]),
                    stratum_quotas[key],
                )
            )
    with gzip.open(
        HOLDOUT_PATH, "wt", encoding="utf-8", newline="", compresslevel=1
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(output_header)
        for row in holdout:
            writer.writerow(
                insert_sampling_fields(
                    row, original_header, "TIME_HOLDOUT", None, None
                )
            )

    manifest_header = [
        "sample_id", "service_date", "segment_id", "direction", "season",
        "is_weekend", "sampling_stratum", "dataset_split", "sampling_seed",
        "train_pool_stratum_count", "sampled_stratum_count", "inclusion_probability",
    ]
    with gzip.open(
        MANIFEST_PATH, "wt", encoding="utf-8", newline="", compresslevel=1
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(manifest_header)
        for row in sorted(
            population,
            key=lambda value: (value["service_date"], value["segment_id"], value["direction"]),
        ):
            if row["service_date"] >= TIME_HOLDOUT_START:
                split = "TIME_HOLDOUT"
                pool_count = sampled_count = probability = ""
            else:
                key = stratum_key(row)
                split = "TRAIN_SAMPLE" if row["sample_id"] in selected_ids else "TRAIN_POOL_UNSAMPLED"
                pool_count = str(len(stratum_pool[key]))
                sampled_count = str(stratum_quotas[key])
                probability = f"{stratum_quotas[key] / len(stratum_pool[key]):.6f}"
            writer.writerow(
                [
                    row["sample_id"], row["service_date"], row["segment_id"],
                    row["direction"], season(int(row["service_date"][4:6])),
                    row["is_weekend"], stratum_label(row), split, SAMPLING_SEED,
                    pool_count, sampled_count, probability,
                ]
            )

    train_sample_by_unit = Counter(unit_key(row) for row in train_sample)
    population_by_unit = Counter(unit_key(row) for row in population)
    unit_rows: list[dict[str, Any]] = []
    for key in sorted(population_by_unit):
        train_count = train_sample_by_unit[key]
        holdout_count = holdout_by_unit[key]
        qc_status = (
            "PASS"
            if train_count >= MIN_TRAIN_SAMPLE_PER_UNIT
            and holdout_count >= MIN_HOLDOUT_PER_UNIT
            else "REVIEW"
        )
        unit_rows.append(
            {
                "segment_id": key[0],
                "direction": key[1],
                "population_count": population_by_unit[key],
                "train_pool_count": len(train_by_unit[key]),
                "train_sample_count": train_count,
                "train_inclusion_rate": f"{train_count / len(train_by_unit[key]):.6f}",
                "train_pool_unsampled_count": len(train_by_unit[key]) - train_count,
                "time_holdout_count": holdout_count,
                "sampling_qc_status": qc_status,
            }
        )
    write_dict_rows(UNIT_SUMMARY_PATH, unit_rows)

    stratum_rows: list[dict[str, Any]] = []
    for key in sorted(stratum_pool):
        pool_count = len(stratum_pool[key])
        sampled_count = stratum_quotas[key]
        stratum_rows.append(
            {
                "segment_id": key[0][0],
                "direction": key[0][1],
                "season": key[1],
                "is_weekend": key[2],
                "train_pool_count": pool_count,
                "train_sample_count": sampled_count,
                "inclusion_probability": f"{sampled_count / pool_count:.6f}",
                "stratum_represented": int(sampled_count > 0),
            }
        )
    write_dict_rows(STRATUM_SUMMARY_PATH, stratum_rows)
    global_distribution = distribution_rows(train_pool, train_sample, holdout)
    write_dict_rows(DISTRIBUTION_PATH, global_distribution)

    review_units = [row for row in unit_rows if row["sampling_qc_status"] != "PASS"]
    unrepresented_strata = [row for row in stratum_rows if not row["stratum_represented"]]
    max_rate_difference = max(
        float(row["absolute_rate_difference"]) for row in global_distribution
    )
    train_ids = {row["sample_id"] for row in train_sample}
    holdout_ids = {row["sample_id"] for row in holdout}
    overlap_count = len(train_ids & holdout_ids)
    report = [
        "# 층화 샘플링 결과",
        "",
        f"- 완전 일별 윈도우 모집단: {len(population):,}개",
        f"- 시간 홀드아웃 시작일: {TIME_HOLDOUT_START[:4]}-{TIME_HOLDOUT_START[4:6]}-{TIME_HOLDOUT_START[6:]}",
        f"- 학습 후보 풀: {len(train_pool):,}개",
        f"- 층화 학습 표본: {len(train_sample):,}개 ({len(train_sample) / len(train_pool):.1%} of train pool)",
        f"- 학습 풀 미추출: {len(train_pool) - len(train_sample):,}개",
        f"- 시간 홀드아웃: {len(holdout):,}개",
        f"- 공간 단위별 학습 표본: 최소 {min(train_sample_by_unit.values())}개, 최대 {max(train_sample_by_unit.values())}개",
        f"- 공간 단위별 시간 홀드아웃: 최소 {min(holdout_by_unit.values())}개, 최대 {max(holdout_by_unit.values())}개",
        f"- 공간 단위 품질 PASS: {len(unit_rows) - len(review_units)}/{len(unit_rows)}",
        f"- 비어 있지 않은 층의 미대표 개수: {len(unrepresented_strata)}개",
        f"- 계절·주말 전역 구성비 최대 절대 차이: {max_rate_difference:.2%}",
        f"- 학습–홀드아웃 sample_id 중복: {overlap_count}개",
        "",
        "## 샘플링 규칙",
        "",
        "- 시간 누수를 막기 위해 2026-04-01 이후 완전 윈도우는 무작위 추출하지 않고 시간 홀드아웃으로 전부 보존",
        "- 학습 후보 풀에서는 segment_id × direction별 표본 수가 가능한 한 같도록 1차 할당",
        "- 각 공간 단위 안에서는 계절(spring/summer/autumn/winter) × 평일·주말 비율에 따라 비례 할당",
        "- 비어 있지 않은 각 층은 최소 1개 이상 포함",
        f"- SHA-256 고정 순위와 시드 {SAMPLING_SEED}을 사용한 비복원 추출",
        "- 전체 모집단의 분할과 층별 포함확률을 manifest에 기록",
        "",
        "## 판정",
        "",
    ]
    if review_units or unrepresented_strata or overlap_count:
        report.append("층화 표본에 추가 검토 항목이 있어 DTW 학습 전에 보완이 필요하다.")
    else:
        report.append("89개 공간 단위와 모든 비어 있지 않은 계절·주말 층이 대표되며 시간 누수는 없다.")
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")

    if review_units or unrepresented_strata or overlap_count:
        raise RuntimeError("Stratified sampling QC failed")
    print(f"Wrote {OUTPUT_DIR}", flush=True)
    print(
        f"population={len(population)} train_pool={len(train_pool)} "
        f"train_sample={len(train_sample)} holdout={len(holdout)} spatial_qc=89/89",
        flush=True,
    )


if __name__ == "__main__":
    main()
