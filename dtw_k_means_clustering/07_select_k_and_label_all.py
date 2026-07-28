#!/usr/bin/env python3
"""Select the final DTW K-means model and label all complete daily windows."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import shutil
import sys
import time
import warnings
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
LOCAL_ML_DEPS = HERE / ".deps_ml"
if LOCAL_ML_DEPS.exists():
    sys.path.insert(0, str(LOCAL_ML_DEPS))

warnings.filterwarnings("ignore", message="h5py not installed")
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import joblib
    import numpy as np
    from sklearn.metrics import adjusted_rand_score, silhouette_samples, silhouette_score
    from tslearn.metrics import cdist_dtw
except ImportError as exc:  # pragma: no cover - environment guidance
    raise RuntimeError(
        "ML dependencies are missing. Install requirements-ml.txt into .deps_ml."
    ) from exc


TRAIN_PATH = HERE / "outputs" / "05_stratified_sampling" / "dtw_training_sample.csv.gz"
HOLDOUT_PATH = HERE / "outputs" / "05_stratified_sampling" / "dtw_time_holdout.csv.gz"
MANIFEST_PATH = HERE / "outputs" / "05_stratified_sampling" / "sampling_manifest.csv.gz"
ALL_WINDOWS_PATH = HERE / "outputs" / "04_daily_96_windows" / "daily_speed_index_windows.csv.gz"
TRAINING_DIR = HERE / "outputs" / "06_dtw_kmeans_training"
OUTPUT_DIR = HERE / "outputs" / "07_final_k_selection"

KS = (4, 5, 6)
SEEDS = (20_260_715, 20_260_716, 20_260_717)
EXPECTED_TRAIN_ROWS = 8_000
EXPECTED_HOLDOUT_ROWS = 7_474
EXPECTED_ALL_ROWS = 36_154
EXPECTED_UNITS = 89
EXPECTED_BINS = 96
SILHOUETTE_SAMPLE_SIZE = 2_000
SILHOUETTE_SEED = 20_260_715
SAKOE_CHIBA_RADIUS = 4
MIN_CLUSTER_SHARE = 0.05
MIN_MEAN_ARI = 0.60
DEFAULT_WORKERS = min(8, os.cpu_count() or 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--rebuild-distance-cache",
        action="store_true",
        help="Recompute the common 2,000-sample DTW distance matrix (slow).",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_rank(seed: int, namespace: str, value: str) -> str:
    return hashlib.sha256(f"{seed}|{namespace}|{value}".encode()).hexdigest()


def load_windows(
    path: Path,
    expected_rows: int,
    retain_all_metadata: bool = False,
) -> tuple[list[dict[str, str]], np.ndarray, list[str]]:
    metadata: list[dict[str, str]] = []
    vectors: list[list[float]] = []
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        speed_columns = [field for field in fields if field.startswith("speed_index_")]
        if len(speed_columns) != EXPECTED_BINS:
            raise RuntimeError(f"Expected {EXPECTED_BINS} speed columns in {path}")
        metadata_columns = [field for field in fields if field not in speed_columns]
        for row in reader:
            vector = [float(row[column]) for column in speed_columns]
            if not np.all(np.isfinite(vector)) or min(vector) < 0 or max(vector) > 1:
                raise RuntimeError(f"Invalid speed index vector: {row['sample_id']}")
            if retain_all_metadata:
                metadata.append({column: row[column] for column in metadata_columns})
            else:
                metadata.append(
                    {
                        "sample_id": row["sample_id"],
                        "service_date": row["service_date"],
                        "segment_id": row["segment_id"],
                        "direction": row["direction"],
                    }
                )
            vectors.append(vector)
    if len(metadata) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows:,} rows in {path}, got {len(metadata):,}")
    if len({row["sample_id"] for row in metadata}) != expected_rows:
        raise RuntimeError(f"Duplicate sample IDs in {path}")
    return metadata, np.asarray(vectors, dtype=np.float64)[:, :, np.newaxis], speed_columns


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_dict_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def allocate_balanced_silhouette_indices(train_meta: list[dict[str, str]]) -> np.ndarray:
    by_unit: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(train_meta):
        by_unit[(row["segment_id"], row["direction"])].append(index)
    if len(by_unit) != EXPECTED_UNITS:
        raise RuntimeError(f"Expected {EXPECTED_UNITS} training spatial units")

    base, remainder = divmod(SILHOUETTE_SAMPLE_SIZE, len(by_unit))
    unit_order = sorted(
        by_unit,
        key=lambda unit: deterministic_rank(SILHOUETTE_SEED, "silhouette-unit", "|".join(unit)),
    )
    allocations = {unit: base + int(rank < remainder) for rank, unit in enumerate(unit_order)}
    selected: list[int] = []
    for unit in sorted(by_unit):
        candidates = sorted(
            by_unit[unit],
            key=lambda index: deterministic_rank(
                SILHOUETTE_SEED, "silhouette-sample", train_meta[index]["sample_id"]
            ),
        )
        selected.extend(candidates[: allocations[unit]])
    result = np.asarray(sorted(selected), dtype=np.int32)
    if len(result) != SILHOUETTE_SAMPLE_SIZE or len(np.unique(result)) != len(result):
        raise RuntimeError("Invalid silhouette sample allocation")
    return result


def load_or_build_distance_cache(
    train_meta: list[dict[str, str]],
    train_x: np.ndarray,
    workers: int,
    rebuild: bool,
) -> tuple[np.ndarray, np.ndarray, float]:
    indices_path = OUTPUT_DIR / "silhouette_training_indices.npy"
    manifest_path = OUTPUT_DIR / "silhouette_sample_manifest.csv"
    cache_path = OUTPUT_DIR / "silhouette_dtw_distance_matrix.npz"

    if not rebuild and indices_path.exists() and manifest_path.exists() and cache_path.exists():
        indices = np.load(indices_path)
        cached = np.load(cache_path)
        matrix = cached["distance_matrix"]
        elapsed = float(np.ravel(cached["elapsed_seconds"])[0])
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            manifest = list(csv.DictReader(handle))
        expected_ids = [train_meta[int(index)]["sample_id"] for index in indices]
        if [row["sample_id"] for row in manifest] != expected_ids:
            raise RuntimeError("Silhouette sample manifest does not match training indices")
        if matrix.shape != (SILHOUETTE_SAMPLE_SIZE, SILHOUETTE_SAMPLE_SIZE):
            raise RuntimeError("Invalid cached DTW distance matrix shape")
        if not np.allclose(matrix, matrix.T) or not np.allclose(np.diag(matrix), 0):
            raise RuntimeError("Cached DTW distance matrix is not symmetric with zero diagonal")
        return indices.astype(np.int32), matrix, elapsed

    indices = allocate_balanced_silhouette_indices(train_meta)
    sample_x = train_x[indices]
    started = time.perf_counter()
    matrix = cdist_dtw(
        sample_x,
        sample_x,
        global_constraint="sakoe_chiba",
        sakoe_chiba_radius=SAKOE_CHIBA_RADIUS,
        n_jobs=workers,
    )
    elapsed = time.perf_counter() - started
    matrix = (matrix + matrix.T) / 2
    np.fill_diagonal(matrix, 0)
    np.save(indices_path, indices)
    np.savez_compressed(cache_path, distance_matrix=matrix, elapsed_seconds=[elapsed])
    sample_rows = []
    for sample_index, train_index in enumerate(indices):
        row = train_meta[int(train_index)]
        sample_rows.append(
            {
                "sample_index": sample_index,
                "sample_id": row["sample_id"],
                "service_date": row["service_date"],
                "segment_id": row["segment_id"],
                "direction": row["direction"],
            }
        )
    write_dict_csv(manifest_path, sample_rows)
    return indices, matrix, elapsed


def evaluate_candidates(
    train_indices: np.ndarray,
    distance_matrix: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    run_rows: list[dict[str, Any]] = []
    labels_by_run: dict[str, np.ndarray] = {}
    for k in KS:
        for seed in SEEDS:
            identifier = f"k{k:02d}_seed{seed}"
            arrays = np.load(TRAINING_DIR / "runs" / f"{identifier}.npz")
            metrics = read_json(TRAINING_DIR / "runs" / f"{identifier}.json")
            labels = arrays["train_labels"].astype(np.int16)
            centers = arrays["centers"][:, :, np.newaxis]
            labels_by_run[identifier] = labels
            sample_labels = labels[train_indices]
            if len(np.unique(sample_labels)) != k:
                raise RuntimeError(f"Silhouette sample misses a cluster for {identifier}")
            sample_values = silhouette_samples(distance_matrix, sample_labels, metric="precomputed")
            center_distances = cdist_dtw(
                centers,
                centers,
                global_constraint="sakoe_chiba",
                sakoe_chiba_radius=SAKOE_CHIBA_RADIUS,
            )
            upper = center_distances[np.triu_indices(k, 1)]
            run_rows.append(
                {
                    "run_id": identifier,
                    "k": k,
                    "seed": seed,
                    "dtw_silhouette_2000": float(
                        silhouette_score(distance_matrix, sample_labels, metric="precomputed")
                    ),
                    "silhouette_negative_share": float(np.mean(sample_values < 0)),
                    "minimum_center_dtw_separation": float(np.min(upper)),
                    "mean_center_dtw_separation": float(np.mean(upper)),
                    "inertia": float(metrics["inertia"]),
                    "minimum_cluster_share": float(metrics["minimum_cluster_share"]),
                    "holdout_to_train_mean_distance_ratio": float(
                        metrics["holdout_to_train_mean_distance_ratio"]
                    ),
                }
            )

    stability_rows: list[dict[str, Any]] = []
    for k in KS:
        run_ids = [f"k{k:02d}_seed{seed}" for seed in SEEDS]
        for run_a, run_b in combinations(run_ids, 2):
            stability_rows.append(
                {
                    "k": k,
                    "run_id_a": run_a,
                    "run_id_b": run_b,
                    "adjusted_rand_index": float(
                        adjusted_rand_score(labels_by_run[run_a], labels_by_run[run_b])
                    ),
                }
            )

    summary_rows: list[dict[str, Any]] = []
    for k in KS:
        candidates = [row for row in run_rows if row["k"] == k]
        stabilities = [row["adjusted_rand_index"] for row in stability_rows if row["k"] == k]
        summary_rows.append(
            {
                "k": k,
                "silhouette_mean": float(np.mean([row["dtw_silhouette_2000"] for row in candidates])),
                "silhouette_min": float(np.min([row["dtw_silhouette_2000"] for row in candidates])),
                "ari_mean": float(np.mean(stabilities)),
                "ari_min": float(np.min(stabilities)),
                "minimum_cluster_share_across_seeds": float(
                    np.min([row["minimum_cluster_share"] for row in candidates])
                ),
                "holdout_ratio_mean": float(
                    np.mean([row["holdout_to_train_mean_distance_ratio"] for row in candidates])
                ),
                "holdout_ratio_max": float(
                    np.max([row["holdout_to_train_mean_distance_ratio"] for row in candidates])
                ),
                "inertia_mean": float(np.mean([row["inertia"] for row in candidates])),
                "center_separation_min": float(
                    np.min([row["minimum_center_dtw_separation"] for row in candidates])
                ),
                "negative_silhouette_share_mean": float(
                    np.mean([row["silhouette_negative_share"] for row in candidates])
                ),
            }
        )
    write_dict_csv(OUTPUT_DIR / "candidate_model_evaluation.csv", run_rows)
    write_dict_csv(OUTPUT_DIR / "candidate_seed_stability.csv", stability_rows)
    write_dict_csv(OUTPUT_DIR / "candidate_k_summary.csv", summary_rows)
    return run_rows, stability_rows, summary_rows


def select_final_model(
    run_rows: list[dict[str, Any]],
    stability_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
) -> tuple[int, int, str, list[dict[str, Any]], dict[str, float]]:
    ranked_rows: list[dict[str, Any]] = []
    for row in summary_rows:
        ranked_rows.append(
            {
                **row,
                "passes_min_cluster_share": int(
                    row["minimum_cluster_share_across_seeds"] >= MIN_CLUSTER_SHARE
                ),
                "passes_mean_ari": int(row["ari_mean"] >= MIN_MEAN_ARI),
                "passes_screen": int(
                    row["minimum_cluster_share_across_seeds"] >= MIN_CLUSTER_SHARE
                    and row["ari_mean"] >= MIN_MEAN_ARI
                ),
            }
        )
    eligible = [row for row in ranked_rows if row["passes_screen"]]
    if not eligible:
        raise RuntimeError("No k candidate passed the minimum cluster-share and ARI screen")
    eligible.sort(
        key=lambda row: (
            -row["silhouette_mean"],
            row["holdout_ratio_mean"],
            -row["center_separation_min"],
            row["k"],
        )
    )
    selected_k = int(eligible[0]["k"])
    for rank, row in enumerate(eligible, start=1):
        row["eligible_rank"] = rank
    for row in ranked_rows:
        row.setdefault("eligible_rank", "")
        row["selected_k"] = int(row["k"] == selected_k)
    ranked_rows.sort(key=lambda row: row["k"])

    average_ari: dict[str, list[float]] = defaultdict(list)
    for row in stability_rows:
        if row["k"] == selected_k:
            average_ari[row["run_id_a"]].append(float(row["adjusted_rand_index"]))
            average_ari[row["run_id_b"]].append(float(row["adjusted_rand_index"]))
    per_run_ari = {run_id: float(np.mean(values)) for run_id, values in average_ari.items()}
    selected_runs = [row for row in run_rows if row["k"] == selected_k]
    selected_runs.sort(
        key=lambda row: (
            -per_run_ari[row["run_id"]],
            -row["dtw_silhouette_2000"],
            row["holdout_to_train_mean_distance_ratio"],
            -row["minimum_cluster_share"],
        )
    )
    representative = selected_runs[0]
    selected_seed = int(representative["seed"])
    return selected_k, selected_seed, representative["run_id"], ranked_rows, per_run_ari


def time_name(bin_index: int) -> str:
    return f"{bin_index // 4:02d}:{(bin_index % 4) * 15:02d}"


def describe_centers(centers: np.ndarray) -> tuple[dict[int, str], list[dict[str, Any]]]:
    descriptors: dict[int, dict[str, Any]] = {}
    for cluster_id, center in enumerate(centers):
        vector = np.asarray(center).reshape(-1)
        period_mean = lambda start, end: float(np.mean(vector[start * 4 : end * 4]))
        night = period_mean(0, 6)
        am_peak = period_mean(7, 10)
        midday = period_mean(11, 15)
        pm_peak = period_mean(17, 20)
        evening = period_mean(20, 24)
        descriptors[cluster_id] = {
            "cluster_id": cluster_id,
            "center_mean": float(np.mean(vector)),
            "center_min": float(np.min(vector)),
            "center_max": float(np.max(vector)),
            "center_std": float(np.std(vector)),
            "minimum_time": time_name(int(np.argmin(vector))),
            "night_0000_0600_mean": night,
            "am_peak_0700_1000_mean": am_peak,
            "midday_1100_1500_mean": midday,
            "pm_peak_1700_2000_mean": pm_peak,
            "evening_2000_2400_mean": evening,
            "commute_drop_from_night": night - (am_peak + pm_peak) / 2,
            "bins_below_070": int(np.sum(vector < 0.70)),
            "bins_below_050": int(np.sum(vector < 0.50)),
        }

    highest = max(descriptors, key=lambda cluster: descriptors[cluster]["center_mean"])
    lowest = min(descriptors, key=lambda cluster: descriptors[cluster]["center_mean"])
    remaining = [cluster for cluster in descriptors if cluster not in {highest, lowest}]
    commute = max(
        remaining,
        key=lambda cluster: descriptors[cluster]["commute_drop_from_night"],
    )
    mapping = {
        highest: "전일 원활형",
        lowest: "상시 저속형",
        commute: "출퇴근 집중형",
    }
    for cluster in remaining:
        if cluster != commute:
            mapping[cluster] = "시간대 집중 정체형"
    for cluster in descriptors:
        mapping.setdefault(cluster, f"기타 패턴형 {cluster}")

    rules = {
        "전일 원활형": "중심곡선 일평균이 가장 높음",
        "상시 저속형": "중심곡선 일평균이 가장 낮음",
        "출퇴근 집중형": "중간 군집 중 심야 대비 출퇴근 하락폭이 가장 큼",
        "시간대 집중 정체형": "나머지 중간 수준의 특정 시간대 저하 패턴",
    }
    rows = []
    for cluster_id in sorted(descriptors):
        rows.append(
            {
                "cluster_id": cluster_id,
                "cluster_label": mapping[cluster_id],
                "naming_rule": rules.get(mapping[cluster_id], "추가 패턴 검토"),
                **{key: value for key, value in descriptors[cluster_id].items() if key != "cluster_id"},
            }
        )
    return mapping, rows


def load_sampling_manifest() -> dict[str, dict[str, str]]:
    with gzip.open(MANIFEST_PATH, "rt", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != EXPECTED_ALL_ROWS:
        raise RuntimeError("Sampling manifest row count changed")
    result = {row["sample_id"]: row for row in rows}
    if len(result) != len(rows):
        raise RuntimeError("Duplicate sample IDs in sampling manifest")
    return result


def write_final_outputs(
    selected_k: int,
    selected_seed: int,
    selected_run_id: str,
    ranked_rows: list[dict[str, Any]],
    per_run_ari: dict[str, float],
    all_meta: list[dict[str, str]],
    all_x: np.ndarray,
    speed_columns: list[str],
    workers: int,
) -> dict[str, Any]:
    model_source = TRAINING_DIR / "models" / f"dtw_kmeans_{selected_run_id}.joblib"
    arrays_path = TRAINING_DIR / "runs" / f"{selected_run_id}.npz"
    model = joblib.load(model_source)
    model.n_jobs = workers
    centers = np.asarray(model.cluster_centers_)[:, :, 0]
    stored = np.load(arrays_path)
    if not np.allclose(centers, stored["centers"]):
        raise RuntimeError("Selected model centers differ from its run artifact")
    if centers.shape != (selected_k, EXPECTED_BINS):
        raise RuntimeError("Selected model has an unexpected center shape")
    if not np.all(np.isfinite(centers)) or np.min(centers) < 0 or np.max(centers) > 1:
        raise RuntimeError("Selected model centers are outside the valid 0..1 range")
    label_mapping, profile_rows = describe_centers(centers)

    started = time.perf_counter()
    distances = model.transform(all_x)
    labeling_seconds = time.perf_counter() - started
    if distances.shape != (EXPECTED_ALL_ROWS, selected_k) or not np.all(np.isfinite(distances)):
        raise RuntimeError("Final all-sample distance matrix is invalid")
    labels = np.argmin(distances, axis=1).astype(np.int16)
    ordered = np.argsort(distances, axis=1)
    nearest = distances[np.arange(len(all_x)), ordered[:, 0]]
    second = distances[np.arange(len(all_x)), ordered[:, 1]]
    margin = second - nearest
    relative_margin = np.divide(margin, second, out=np.zeros_like(margin), where=second > 0)
    if np.any(nearest < 0) or np.any(margin < -1e-12):
        raise RuntimeError("Final nearest-center distances or margins are invalid")

    manifest = load_sampling_manifest()
    if set(manifest) != {row["sample_id"] for row in all_meta}:
        raise RuntimeError("All-window and sampling-manifest sample IDs differ")
    label_fields = [
        "sample_id", "service_date", "year_month", "day_of_week", "day_name",
        "is_weekend", "season", "segment_id", "direction", "from_station_no",
        "from_station_name", "to_station_no", "to_station_name", "dataset_split",
        "cluster_id", "cluster_label", "distance_to_center", "second_best_distance",
        "distance_margin", "relative_distance_margin",
    ]
    labels_path = OUTPUT_DIR / "dtw_cluster_labels.csv"
    split_cluster_counts: Counter[tuple[str, int]] = Counter()
    cluster_counts: Counter[int] = Counter()
    predicted_by_id: dict[str, tuple[int, float]] = {}
    with labels_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=label_fields)
        writer.writeheader()
        for row, label, nearest_value, second_value, margin_value, relative_value in zip(
            all_meta, labels, nearest, second, margin, relative_margin
        ):
            sample = manifest[row["sample_id"]]
            cluster_id = int(label)
            split = sample["dataset_split"]
            cluster_counts[cluster_id] += 1
            split_cluster_counts[(split, cluster_id)] += 1
            predicted_by_id[row["sample_id"]] = (cluster_id, float(nearest_value))
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "service_date": row["service_date"],
                    "year_month": row["year_month"],
                    "day_of_week": row["day_of_week"],
                    "day_name": row["day_name"],
                    "is_weekend": row["is_weekend"],
                    "season": sample["season"],
                    "segment_id": row["segment_id"],
                    "direction": row["direction"],
                    "from_station_no": row["from_station_no"],
                    "from_station_name": row["from_station_name"],
                    "to_station_no": row["to_station_no"],
                    "to_station_name": row["to_station_name"],
                    "dataset_split": split,
                    "cluster_id": cluster_id,
                    "cluster_label": label_mapping[cluster_id],
                    "distance_to_center": f"{nearest_value:.9f}",
                    "second_best_distance": f"{second_value:.9f}",
                    "distance_margin": f"{margin_value:.9f}",
                    "relative_distance_margin": f"{relative_value:.9f}",
                }
            )

    center_rows: list[dict[str, Any]] = []
    for cluster_id, center in enumerate(centers):
        for bin_index, value in enumerate(center):
            center_rows.append(
                {
                    "selected_run_id": selected_run_id,
                    "cluster_id": cluster_id,
                    "cluster_label": label_mapping[cluster_id],
                    "bin_index": bin_index,
                    "time_bin": speed_columns[bin_index].removeprefix("speed_index_"),
                    "speed_index": f"{float(value):.9f}",
                }
            )
    write_dict_csv(OUTPUT_DIR / "dtw_cluster_centers.csv", center_rows)

    for row in profile_rows:
        cluster_id = int(row["cluster_id"])
        row["all_sample_count"] = cluster_counts[cluster_id]
        row["all_sample_share"] = cluster_counts[cluster_id] / len(all_meta)
        row["train_sample_count"] = split_cluster_counts[("TRAIN_SAMPLE", cluster_id)]
        row["train_pool_unsampled_count"] = split_cluster_counts[
            ("TRAIN_POOL_UNSAMPLED", cluster_id)
        ]
        row["time_holdout_count"] = split_cluster_counts[("TIME_HOLDOUT", cluster_id)]
    write_dict_csv(OUTPUT_DIR / "final_cluster_profiles.csv", profile_rows)

    summary_rows: list[dict[str, Any]] = []
    split_totals = Counter(row["dataset_split"] for row in manifest.values())
    for split in ("ALL", "TRAIN_SAMPLE", "TRAIN_POOL_UNSAMPLED", "TIME_HOLDOUT"):
        total = len(all_meta) if split == "ALL" else split_totals[split]
        for cluster_id in range(selected_k):
            count = cluster_counts[cluster_id] if split == "ALL" else split_cluster_counts[(split, cluster_id)]
            summary_rows.append(
                {
                    "dataset_split": split,
                    "cluster_id": cluster_id,
                    "cluster_label": label_mapping[cluster_id],
                    "sample_count": count,
                    "sample_share": count / total,
                }
            )
    write_dict_csv(OUTPUT_DIR / "final_label_summary.csv", summary_rows)
    write_dict_csv(OUTPUT_DIR / "final_k_selection_metrics.csv", ranked_rows)

    representative_eval = next(
        row for row in csv.DictReader(
            (OUTPUT_DIR / "candidate_model_evaluation.csv").open(
                "r", encoding="utf-8-sig", newline=""
            )
        ) if row["run_id"] == selected_run_id
    )
    selection_row = {
        "selected_k": selected_k,
        "selected_seed": selected_seed,
        "selected_run_id": selected_run_id,
        "selection_screen": f"minimum_cluster_share>={MIN_CLUSTER_SHARE}; mean_ARI>={MIN_MEAN_ARI}",
        "k_ranking_rule": "screen 통과 후 silhouette 평균 내림차순; holdout 비율 오름차순",
        "representative_run_rule": "동일 k 내 평균 쌍별 ARI 내림차순 우선",
        "representative_mean_pairwise_ari": per_run_ari[selected_run_id],
        "representative_dtw_silhouette_2000": representative_eval["dtw_silhouette_2000"],
        "all_labeled_rows": len(all_meta),
        "labeling_seconds": labeling_seconds,
        "source_model": str(model_source.relative_to(HERE)),
        "final_model": "outputs/07_final_k_selection/final_dtw_kmeans_model.joblib",
    }
    write_dict_csv(OUTPUT_DIR / "final_model_selection.csv", [selection_row])
    final_model_path = OUTPUT_DIR / "final_dtw_kmeans_model.joblib"
    shutil.copy2(model_source, final_model_path)
    reloaded = joblib.load(final_model_path)
    if not np.allclose(reloaded.cluster_centers_[:, :, 0], centers):
        raise RuntimeError("Copied final model failed reload verification")

    def validate_split_predictions(
        source_path: Path, expected_labels: np.ndarray, expected_distances: np.ndarray
    ) -> None:
        meta, _, _ = load_windows(source_path, len(expected_labels))
        actual = [predicted_by_id[row["sample_id"]] for row in meta]
        actual_labels = np.asarray([item[0] for item in actual])
        actual_distances = np.asarray([item[1] for item in actual])
        if not np.array_equal(actual_labels, expected_labels):
            raise RuntimeError(f"Full labels differ from stored run labels for {source_path.name}")
        if not np.allclose(actual_distances, expected_distances, atol=1e-8):
            raise RuntimeError(f"Full distances differ from stored run distances for {source_path.name}")

    validate_split_predictions(TRAIN_PATH, stored["train_labels"], stored["train_nearest_distance"])
    validate_split_predictions(
        HOLDOUT_PATH, stored["holdout_labels"], stored["holdout_nearest_distance"]
    )
    if len(cluster_counts) != selected_k or min(cluster_counts.values()) <= 0:
        raise RuntimeError("Final all-sample labels contain an empty cluster")
    if len(label_mapping) != selected_k or len(set(label_mapping.values())) != selected_k:
        raise RuntimeError("Final semantic labels are incomplete or duplicated")
    if len(center_rows) != selected_k * EXPECTED_BINS:
        raise RuntimeError("Final center row count is invalid")

    return {
        "selection": selection_row,
        "profiles": profile_rows,
        "summary": summary_rows,
        "labels_sha256": sha256_file(labels_path),
        "model_sha256": sha256_file(final_model_path),
    }


def write_report(
    evaluation_seconds: float,
    selected_k: int,
    selected_seed: int,
    selected_run_id: str,
    ranked_rows: list[dict[str, Any]],
    result: dict[str, Any],
) -> None:
    lines = [
        "# 최종 k 선택 및 전체 샘플 라벨링 결과",
        "",
        f"- 최종 k: **{selected_k}**",
        f"- 대표 모델: **{selected_run_id}** (seed={selected_seed})",
        f"- 전체 라벨링: **{EXPECTED_ALL_ROWS:,}개** 완전 일별 윈도우",
        f"- 공통 DTW Silhouette 표본: {SILHOUETTE_SAMPLE_SIZE:,}개, 89개 공간 단위 균형 표집",
        f"- DTW 제약: Sakoe-Chiba 반경 {SAKOE_CHIBA_RADIUS}칸(±{SAKOE_CHIBA_RADIUS * 15}분)",
        f"- 공통 거리행렬 계산시간(캐시 기록): {evaluation_seconds:.1f}초",
        f"- 전체 라벨링 계산시간: {float(result['selection']['labeling_seconds']):.1f}초",
        "",
        "## k 후보 종합 평가",
        "",
        "| k | DTW Silhouette 평균 | ARI 평균 | 최소 군집 비중 | 홀드아웃/학습 거리 | 최소 중심분리 | 판정 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in ranked_rows:
        decision = "최종 선택" if row["selected_k"] else ("검토 통과" if row["passes_screen"] else "기준 미달")
        lines.append(
            f"| {row['k']} | {row['silhouette_mean']:.4f} | {row['ari_mean']:.4f} | "
            f"{row['minimum_cluster_share_across_seeds']:.2%} | {row['holdout_ratio_mean']:.3f} | "
            f"{row['center_separation_min']:.4f} | {decision} |"
        )
    lines.extend(
        [
            "",
            "k=6은 한 시드에서 최소 군집 비중 5% 기준을 충족하지 못했고 시드 안정성도 낮았다. "
            "k=4와 k=5는 사전 기준을 통과했으며, k=4가 공통 DTW Silhouette, 최소 군집 비중, "
            "홀드아웃 거리 비율, 최소 중심 분리에서 모두 우수해 최종 선택됐다.",
            "",
            "대표 실행은 동일 k의 다른 시드들과 계산한 평균 쌍별 ARI가 가장 높은 실행을 우선했다. "
            "따라서 시드 간 합의의 중심에 있는 `k04_seed20260715`를 전체 라벨링 모델로 채택했다.",
            "",
            "## 최종 군집 해석과 전체 비중",
            "",
            "| 군집 ID | 군집명 | 전체 표본 | 비중 | 중심 평균 | 최저 시각 | 명명 근거 |",
            "| ---: | --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in result["profiles"]:
        lines.append(
            f"| {row['cluster_id']} | {row['cluster_label']} | {row['all_sample_count']:,} | "
            f"{row['all_sample_share']:.2%} | {row['center_mean']:.3f} | {row['minimum_time']} | "
            f"{row['naming_rule']} |"
        )
    lines.extend(
        [
            "",
            "군집명은 모델 번호의 크기에 의미를 부여하지 않고 중심곡선만으로 결정했다. "
            "`출퇴근 집중형`은 중간 수준 군집 중 심야 대비 07~10시와 17~20시의 평균 하락폭이 가장 큰 군집이다.",
            "",
            "## 검증",
            "",
            f"- 전체 {EXPECTED_ALL_ROWS:,}개 `sample_id`가 누락·중복 없이 한 번씩 라벨링됨",
            f"- 학습 8,000개와 시간 홀드아웃 7,474개의 라벨·중심거리가 선택 실행 저장값과 일치함",
            f"- {selected_k}개 군집 모두 전체 표본을 포함하며 빈 군집 없음",
            f"- 중심곡선 {selected_k * EXPECTED_BINS:,}행({selected_k}개 × 96시각), 값 범위 0~1 검증",
            "- 최종 모델 재로딩 후 중심곡선 일치 확인",
            f"- 라벨 파일 SHA-256: `{result['labels_sha256']}`",
            f"- 최종 모델 SHA-256: `{result['model_sha256']}`",
            "",
            "## 산출물",
            "",
            "- `dtw_cluster_labels.csv`: 전체 일별 구간·방향 군집 라벨, 최근접·차순위 중심거리와 판별 여유",
            "- `dtw_cluster_centers.csv`: 최종 4개 군집의 15분 단위 중심곡선",
            "- `final_cluster_profiles.csv`: 군집별 해석 지표와 데이터 분할별 표본 수",
            "- `final_label_summary.csv`: 전체·학습·미추출 학습풀·시간 홀드아웃의 군집 비중",
            "- `final_k_selection_metrics.csv`: k별 선택 기준과 판정",
            "- `final_model_selection.csv`: 최종 모델 선택 기록",
            "- `final_dtw_kmeans_model.joblib`: 전체 라벨링에 사용한 직렬화 모델",
        ]
    )
    (OUTPUT_DIR / "final_model_selection_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise RuntimeError("workers must be positive")
    os.environ.setdefault("NUMBA_CACHE_DIR", str(HERE / ".numba_cache"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading training windows and common DTW evaluation cache", flush=True)
    train_meta, train_x, train_columns = load_windows(TRAIN_PATH, EXPECTED_TRAIN_ROWS)
    indices, distance_matrix, distance_seconds = load_or_build_distance_cache(
        train_meta, train_x, args.workers, args.rebuild_distance_cache
    )
    print("Evaluating k=4,5,6 with common silhouette sample and full-label ARI", flush=True)
    run_rows, stability_rows, summary_rows = evaluate_candidates(indices, distance_matrix)
    selected_k, selected_seed, selected_run_id, ranked_rows, per_run_ari = select_final_model(
        run_rows, stability_rows, summary_rows
    )
    print(f"Selected {selected_run_id}; loading all complete daily windows", flush=True)
    all_meta, all_x, all_columns = load_windows(
        ALL_WINDOWS_PATH, EXPECTED_ALL_ROWS, retain_all_metadata=True
    )
    if train_columns != all_columns:
        raise RuntimeError("Training and all-window speed columns differ")
    result = write_final_outputs(
        selected_k, selected_seed, selected_run_id, ranked_rows, per_run_ari,
        all_meta, all_x, all_columns, args.workers,
    )
    write_report(
        distance_seconds, selected_k, selected_seed, selected_run_id, ranked_rows, result
    )
    print(
        f"Completed: k={selected_k}, model={selected_run_id}, rows={len(all_meta):,}",
        flush=True,
    )


if __name__ == "__main__":
    main()
