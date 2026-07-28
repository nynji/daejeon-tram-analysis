#!/usr/bin/env python3
"""Train resumable DTW K-means candidates for k=4,5,6 and three seeds."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import sys
import time
import warnings
from collections import Counter
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
    import numba
    import numpy as np
    import scipy
    import sklearn
    import tslearn
    from tslearn.clustering import TimeSeriesKMeans
except ImportError as exc:  # pragma: no cover - environment guidance
    raise RuntimeError(
        "ML dependencies are missing. Install requirements-ml.txt into .deps_ml "
        "with a compatible Python runtime."
    ) from exc


TRAIN_PATH = (
    HERE / "outputs" / "05_stratified_sampling" / "dtw_training_sample.csv.gz"
)
HOLDOUT_PATH = (
    HERE / "outputs" / "05_stratified_sampling" / "dtw_time_holdout.csv.gz"
)
OUTPUT_DIR = HERE / "outputs" / "06_dtw_kmeans_training"
RUN_DIR = OUTPUT_DIR / "runs"
MODEL_DIR = OUTPUT_DIR / "models"

KS = (4, 5, 6)
SEEDS = (20_260_715, 20_260_716, 20_260_717)
EXPECTED_TRAIN_ROWS = 8_000
EXPECTED_HOLDOUT_ROWS = 7_474
EXPECTED_BINS = 96
SAKOE_CHIBA_RADIUS = 4
MAX_ITER = 15
MAX_ITER_BARYCENTER = 10
TOLERANCE = 1e-3
DEFAULT_WORKERS = min(8, os.cpu_count() or 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--ks", nargs="+", type=int, default=list(KS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--single-k", type=int)
    parser.add_argument("--single-seed", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_windows(path: Path, expected_rows: int) -> tuple[list[dict[str, str]], np.ndarray, list[str]]:
    metadata: list[dict[str, str]] = []
    vectors: list[list[float]] = []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        speed_columns = [
            field for field in (reader.fieldnames or []) if field.startswith("speed_index_")
        ]
        if len(speed_columns) != EXPECTED_BINS:
            raise RuntimeError(f"Expected 96 speed-index columns in {path}")
        for row in reader:
            values = [float(row[column]) for column in speed_columns]
            if any(not 0.0 <= value <= 1.0 for value in values):
                raise RuntimeError(f"Speed index outside 0..1: {row['sample_id']}")
            metadata.append(
                {
                    "sample_id": row["sample_id"],
                    "service_date": row["service_date"],
                    "segment_id": row["segment_id"],
                    "direction": row["direction"],
                }
            )
            vectors.append(values)
    if len(metadata) != expected_rows or len({row["sample_id"] for row in metadata}) != expected_rows:
        raise RuntimeError(f"Unexpected row count or duplicate sample IDs in {path}")
    array = np.asarray(vectors, dtype=np.float64)[:, :, np.newaxis]
    return metadata, array, speed_columns


def run_id(k: int, seed: int) -> str:
    return f"k{k:02d}_seed{seed}"


def artifact_paths(identifier: str) -> tuple[Path, Path, Path]:
    return (
        RUN_DIR / f"{identifier}.json",
        RUN_DIR / f"{identifier}.npz",
        MODEL_DIR / f"dtw_kmeans_{identifier}.joblib",
    )


def train_one(
    k: int,
    seed: int,
    workers: int,
    train_x: np.ndarray,
    holdout_x: np.ndarray,
    train_hash: str,
    holdout_hash: str,
    overwrite: bool,
) -> dict[str, Any]:
    identifier = run_id(k, seed)
    metrics_path, arrays_path, model_path = artifact_paths(identifier)
    if not overwrite and metrics_path.exists() and arrays_path.exists() and model_path.exists():
        print(f"Skipping completed {identifier}", flush=True)
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    print(f"Training {identifier}", flush=True)
    started = time.perf_counter()
    model = TimeSeriesKMeans(
        n_clusters=k,
        metric="dtw",
        metric_params={
            "global_constraint": "sakoe_chiba",
            "sakoe_chiba_radius": SAKOE_CHIBA_RADIUS,
        },
        max_iter=MAX_ITER,
        tol=TOLERANCE,
        n_init=1,
        max_iter_barycenter=MAX_ITER_BARYCENTER,
        n_jobs=workers,
        random_state=seed,
        init="k-means++",
        verbose=0,
    )
    model.fit(train_x)
    train_distances = model.transform(train_x)
    train_labels = np.argmin(train_distances, axis=1).astype(np.int16)
    if not np.array_equal(train_labels, model.labels_.astype(np.int16)):
        raise RuntimeError(f"Fit and transform labels differ for {identifier}")
    holdout_distances = model.transform(holdout_x)
    holdout_labels = np.argmin(holdout_distances, axis=1).astype(np.int16)
    train_nearest = train_distances[np.arange(len(train_x)), train_labels]
    holdout_nearest = holdout_distances[np.arange(len(holdout_x)), holdout_labels]
    centers = model.cluster_centers_[:, :, 0]
    cluster_counts = np.bincount(train_labels, minlength=k)
    holdout_cluster_counts = np.bincount(holdout_labels, minlength=k)
    if np.any(cluster_counts == 0) or np.any(holdout_cluster_counts == 0):
        raise RuntimeError(f"Empty training or holdout cluster for {identifier}")
    if not np.all(np.isfinite(centers)) or np.min(centers) < 0 or np.max(centers) > 1:
        raise RuntimeError(f"Invalid center values for {identifier}")

    elapsed = time.perf_counter() - started
    metrics: dict[str, Any] = {
        "run_id": identifier,
        "k": k,
        "seed": seed,
        "train_rows": len(train_x),
        "holdout_rows": len(holdout_x),
        "time_bins": train_x.shape[1],
        "metric": "dtw",
        "global_constraint": "sakoe_chiba",
        "sakoe_chiba_radius_bins": SAKOE_CHIBA_RADIUS,
        "sakoe_chiba_radius_minutes": SAKOE_CHIBA_RADIUS * 15,
        "max_iter": MAX_ITER,
        "max_iter_barycenter": MAX_ITER_BARYCENTER,
        "tolerance": TOLERANCE,
        "n_iter": int(model.n_iter_),
        "converged_before_max_iter": int(model.n_iter_ < MAX_ITER),
        "inertia": float(model.inertia_),
        "train_mean_distance": float(np.mean(train_nearest)),
        "train_median_distance": float(np.median(train_nearest)),
        "train_p95_distance": float(np.quantile(train_nearest, 0.95)),
        "holdout_mean_distance": float(np.mean(holdout_nearest)),
        "holdout_median_distance": float(np.median(holdout_nearest)),
        "holdout_p95_distance": float(np.quantile(holdout_nearest, 0.95)),
        "holdout_to_train_mean_distance_ratio": float(
            np.mean(holdout_nearest) / np.mean(train_nearest)
        ),
        "minimum_cluster_count": int(np.min(cluster_counts)),
        "minimum_cluster_share": float(np.min(cluster_counts) / len(train_x)),
        "maximum_cluster_count": int(np.max(cluster_counts)),
        "maximum_cluster_share": float(np.max(cluster_counts) / len(train_x)),
        "center_min": float(np.min(centers)),
        "center_max": float(np.max(centers)),
        "elapsed_seconds": elapsed,
        "workers": workers,
        "train_input_sha256": train_hash,
        "holdout_input_sha256": holdout_hash,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "scikit_learn_version": sklearn.__version__,
        "tslearn_version": tslearn.__version__,
        "numba_version": numba.__version__,
    }

    np.savez_compressed(
        arrays_path,
        centers=centers,
        train_labels=train_labels,
        train_nearest_distance=train_nearest,
        holdout_labels=holdout_labels,
        holdout_nearest_distance=holdout_nearest,
        train_cluster_counts=cluster_counts,
        holdout_cluster_counts=holdout_cluster_counts,
    )
    joblib.dump(model, model_path, compress=3)
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Completed {identifier}: iter={model.n_iter_} inertia={model.inertia_:.6f} "
        f"min_cluster={np.min(cluster_counts)} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return metrics


def write_combined_outputs(
    runs: list[dict[str, Any]],
    train_meta: list[dict[str, str]],
    holdout_meta: list[dict[str, str]],
    speed_columns: list[str],
) -> None:
    metrics_fields = list(runs[0])
    with (OUTPUT_DIR / "dtw_candidate_run_metrics.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=metrics_fields)
        writer.writeheader()
        writer.writerows(runs)

    label_header = [
        "run_id", "k", "seed", "sample_id", "service_date", "segment_id",
        "direction", "cluster_id", "distance_to_center",
    ]
    center_header = [
        "run_id", "k", "seed", "cluster_id", "bin_index", "time_bin", "speed_index",
    ]
    size_header = [
        "run_id", "k", "seed", "dataset_split", "cluster_id", "sample_count", "sample_share",
    ]
    with gzip.open(
        OUTPUT_DIR / "dtw_training_labels.csv.gz",
        "wt", encoding="utf-8", newline="", compresslevel=1,
    ) as train_handle, gzip.open(
        OUTPUT_DIR / "dtw_holdout_labels.csv.gz",
        "wt", encoding="utf-8", newline="", compresslevel=1,
    ) as holdout_handle, (OUTPUT_DIR / "dtw_candidate_centers.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as center_handle, (OUTPUT_DIR / "dtw_candidate_cluster_sizes.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as size_handle:
        train_writer = csv.writer(train_handle)
        holdout_writer = csv.writer(holdout_handle)
        center_writer = csv.writer(center_handle)
        size_writer = csv.writer(size_handle)
        train_writer.writerow(label_header)
        holdout_writer.writerow(label_header)
        center_writer.writerow(center_header)
        size_writer.writerow(size_header)
        for metrics in sorted(runs, key=lambda row: (row["k"], row["seed"])):
            identifier = metrics["run_id"]
            _, arrays_path, _ = artifact_paths(identifier)
            arrays = np.load(arrays_path)
            train_labels = arrays["train_labels"]
            train_distances = arrays["train_nearest_distance"]
            holdout_labels = arrays["holdout_labels"]
            holdout_distances = arrays["holdout_nearest_distance"]
            centers = arrays["centers"]
            for row, label, distance in zip(train_meta, train_labels, train_distances):
                train_writer.writerow(
                    [identifier, metrics["k"], metrics["seed"], row["sample_id"],
                     row["service_date"], row["segment_id"], row["direction"],
                     int(label), f"{float(distance):.9f}"]
                )
            for row, label, distance in zip(holdout_meta, holdout_labels, holdout_distances):
                holdout_writer.writerow(
                    [identifier, metrics["k"], metrics["seed"], row["sample_id"],
                     row["service_date"], row["segment_id"], row["direction"],
                     int(label), f"{float(distance):.9f}"]
                )
            for cluster_id, center in enumerate(centers):
                for bin_index, value in enumerate(center):
                    center_writer.writerow(
                        [identifier, metrics["k"], metrics["seed"], cluster_id,
                         bin_index, speed_columns[bin_index].removeprefix("speed_index_"),
                         f"{float(value):.9f}"]
                    )
            for split, labels in (("TRAIN_SAMPLE", train_labels), ("TIME_HOLDOUT", holdout_labels)):
                counts = Counter(map(int, labels))
                for cluster_id in range(int(metrics["k"])):
                    count = counts[cluster_id]
                    size_writer.writerow(
                        [identifier, metrics["k"], metrics["seed"], split,
                         cluster_id, count, f"{count / len(labels):.9f}"]
                    )


def write_report(runs: list[dict[str, Any]]) -> None:
    total_seconds = sum(float(row["elapsed_seconds"]) for row in runs)
    report = [
        "# DTW K-means 후보 학습 결과",
        "",
        f"- 학습 표본: {EXPECTED_TRAIN_ROWS:,}개 × {EXPECTED_BINS}개 시각",
        f"- 시간 홀드아웃: {EXPECTED_HOLDOUT_ROWS:,}개",
        f"- 후보 k: {', '.join(map(str, KS))}",
        f"- 시드: {', '.join(map(str, SEEDS))}",
        f"- 완료 후보 모델: {len(runs)}개",
        f"- DTW 제약: Sakoe-Chiba 반경 {SAKOE_CHIBA_RADIUS}칸(±{SAKOE_CHIBA_RADIUS * 15}분)",
        f"- 반복 상한: K-means {MAX_ITER}회, DBA 중심곡선 {MAX_ITER_BARYCENTER}회",
        f"- 전체 후보 실행시간 합계: {total_seconds / 60:.1f}분",
        f"- 최대 반복 전에 종료: {sum(int(row['converged_before_max_iter']) for row in runs)}/{len(runs)}",
        f"- 빈 학습 군집: 0개",
        f"- 중심곡선 값 범위: {min(float(row['center_min']) for row in runs):.4f} ~ {max(float(row['center_max']) for row in runs):.4f}",
        "",
        "## 학습 설정",
        "",
        "- 속도지수 0~1의 절대 혼잡 수준을 보존하고 추가 표준화는 적용하지 않음",
        "- k-means++ 초기화, 시드별 n_init=1로 독립 실행",
        "- 제한 DTW 거리와 DBA(DTW barycenter averaging) 중심곡선 사용",
        "- 각 실행에 학습 라벨·홀드아웃 라벨·중심곡선·모델·환경 버전을 저장",
        "",
        "## 후보 실행 요약",
        "",
    ]
    for row in sorted(runs, key=lambda value: (value["k"], value["seed"])):
        report.append(
            f"- {row['run_id']}: iter={row['n_iter']}, inertia={float(row['inertia']):.6f}, "
            f"min_cluster_share={float(row['minimum_cluster_share']):.1%}, "
            f"holdout/train_distance={float(row['holdout_to_train_mean_distance_ratio']):.3f}"
        )
    report.extend(
        [
            "",
            "## 판정",
            "",
            "k=4·5·6의 다중 시드 후보 학습을 완료했다. 최종 k 선택은 다음 단계에서 Silhouette, 시드 안정성 ARI, 최소 군집 비중, 시간 홀드아웃 거리와 해석 가능성을 종합해 수행한다.",
        ]
    )
    (OUTPUT_DIR / "dtw_training_qc_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    if (args.single_k is None) != (args.single_seed is None):
        raise RuntimeError("--single-k and --single-seed must be supplied together")
    if args.single_k is not None:
        if args.single_k not in KS or args.single_seed not in SEEDS:
            raise RuntimeError("Single-run k or seed is outside the required candidate grid")
        args.ks = [args.single_k]
        args.seeds = [args.single_seed]
        single_run = True
    else:
        single_run = False
    if sorted(args.ks) != list(KS) or sorted(args.seeds) != list(SEEDS):
        if not single_run:
            raise RuntimeError(f"This workflow requires ks={KS} and seeds={SEEDS}")
    if args.workers < 1:
        raise RuntimeError("workers must be positive")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(HERE / ".numba_cache"))

    print("Loading training and holdout windows", flush=True)
    train_meta, train_x, speed_columns = load_windows(TRAIN_PATH, EXPECTED_TRAIN_ROWS)
    holdout_meta, holdout_x, holdout_speed_columns = load_windows(
        HOLDOUT_PATH, EXPECTED_HOLDOUT_ROWS
    )
    if speed_columns != holdout_speed_columns:
        raise RuntimeError("Training and holdout time columns differ")
    if {row["sample_id"] for row in train_meta} & {row["sample_id"] for row in holdout_meta}:
        raise RuntimeError("Training and holdout sample IDs overlap")
    train_hash = sha256_file(TRAIN_PATH)
    holdout_hash = sha256_file(HOLDOUT_PATH)

    runs: list[dict[str, Any]] = []
    for k in sorted(args.ks):
        for seed in sorted(args.seeds):
            runs.append(
                train_one(
                    k, seed, args.workers, train_x, holdout_x,
                    train_hash, holdout_hash, args.overwrite,
                )
            )
    if len(runs) != len(KS) * len(SEEDS):
        if single_run:
            print(f"Completed single run {runs[0]['run_id']}", flush=True)
            return
        raise RuntimeError("Not all required DTW candidates completed")
    write_combined_outputs(runs, train_meta, holdout_meta, speed_columns)
    write_report(runs)
    print(f"Wrote {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
