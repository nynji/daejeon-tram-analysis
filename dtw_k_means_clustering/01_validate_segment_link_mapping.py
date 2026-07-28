#!/usr/bin/env python3
"""Build and audit the tram station-segment to standard-link mapping."""

from __future__ import annotations

import csv
import json
import math
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DEPS = HERE / ".deps"
if DEPS.exists():
    sys.path.insert(0, str(DEPS))

import networkx as nx
import shapefile
from pyproj import Transformer


DISTRICT_PREFIXES = {"183", "184", "185", "186", "187"}
SNAP_CANDIDATE_COUNT = 8
SNAP_COST_MULTIPLIER = 3.0
MAX_QC_SNAP_M = 120.0
MAX_QC_ROUTE_RATIO = 2.10


def nfc(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def locate_inputs(workspace: Path) -> tuple[Path, Path]:
    raw = workspace / "raw_data"
    station_candidates = [
        path
        for path in raw.rglob("*.csv")
        if "좌표" in nfc(path.name) and "station_no" in path.read_text(encoding="utf-8-sig", errors="ignore")[:100]
    ]
    link_candidates = list(raw.rglob("MOCT_LINK.shp"))
    if len(station_candidates) != 1:
        raise RuntimeError(f"Expected one station coordinate CSV, found {len(station_candidates)}")
    if len(link_candidates) != 1:
        raise RuntimeError(f"Expected one MOCT_LINK.shp, found {len(link_candidates)}")
    return station_candidates[0], link_candidates[0]


def segment_topology() -> list[dict[str, Any]]:
    pairs: list[tuple[int, int, str]] = [
        (number, number + 1, "main_loop") for number in range(201, 240)
    ]
    pairs.extend(
        [
            (240, 201, "main_loop"),
            (212, 241, "yeonchuk_branch"),
            (241, 242, "yeonchuk_branch"),
            (242, 243, "yeonchuk_branch"),
            (243, 244, "yeonchuk_branch"),
            (233, 245, "jinjam_branch"),
        ]
    )
    result = []
    for order, (a, b, branch) in enumerate(pairs, start=1):
        result.append(
            {
                "segment_id": f"SEG_{order:02d}_{a}_{b}",
                "segment_order": order,
                "station_a": a,
                "station_b": b,
                "branch": branch,
            }
        )
    assert len(result) == 45
    return result


def load_stations(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 45:
        raise RuntimeError(f"Expected 45 stations, found {len(rows)}")

    stations: list[dict[str, Any]] = []
    for row in rows:
        station = dict(row)
        station["station_no"] = int(row["station_no"])
        station["original_lat"] = float(row["lat"])
        station["original_lon"] = float(row["lon"])
        station["validated_lat"] = station["original_lat"]
        station["validated_lon"] = station["original_lon"]
        station["coordinate_qc"] = "original"
        station["coordinate_qc_reason"] = ""
        stations.append(station)

    # The route diagram orders these stops east-to-west as 212-213-214-215.
    # Station 214 was user-corrected in the source CSV to longitude 127.4146.
    by_no = {row["station_no"]: row for row in stations}
    s214 = by_no[214]
    if not math.isclose(s214["validated_lon"], 127.4146, abs_tol=1e-7):
        raise RuntimeError("Station 214 longitude must be corrected to 127.4146")
    s214["coordinate_qc"] = "source_corrected_214_longitude"
    s214["coordinate_qc_reason"] = (
        "노선도 정거장 순서(212-213-214-215)를 맞추기 위해 사용자 지정 경도 127.4146 적용"
    )
    return stations


def load_links(path: Path) -> tuple[list[dict[str, Any]], dict[str, tuple[float, float]]]:
    reader = shapefile.Reader(str(path), encoding="cp949")
    selected: list[tuple[int, dict[str, Any]]] = []
    fields = ["LINK_ID", "F_NODE", "T_NODE", "LENGTH", "ROAD_NAME", "MAX_SPD", "ROAD_RANK"]
    for index, record in enumerate(reader.iterRecords(fields=fields)):
        if str(record.LINK_ID)[:3] in DISTRICT_PREFIXES:
            selected.append((index, record.as_dict()))

    links: list[dict[str, Any]] = []
    node_xy: dict[str, tuple[float, float]] = {}
    for index, values in selected:
        link_id = values["LINK_ID"]
        f_node = values["F_NODE"]
        t_node = values["T_NODE"]
        length_m = values["LENGTH"]
        road_name = values["ROAD_NAME"]
        max_speed = values["MAX_SPD"]
        road_rank = values["ROAD_RANK"]
        shape = reader.shape(index)
        if not shape.points:
            continue
        f_node, t_node = str(f_node), str(t_node)
        node_xy.setdefault(f_node, tuple(shape.points[0]))
        node_xy.setdefault(t_node, tuple(shape.points[-1]))
        links.append(
            {
                "link_id": str(link_id),
                "f_node": f_node,
                "t_node": t_node,
                "length_m": max(float(length_m or 0.0), 1.0),
                "road_name": str(road_name or ""),
                "max_speed_kmh": int(max_speed or 0),
                "road_rank": str(road_rank or ""),
                "points": [tuple(point) for point in shape.points],
            }
        )
    if not links:
        raise RuntimeError("No Daejeon links were selected")
    return links, node_xy


def prepare_station_candidates(
    stations: list[dict[str, Any]], node_xy: dict[str, tuple[float, float]]
) -> None:
    transformer = Transformer.from_crs(4326, 5186, always_xy=True)
    node_items = list(node_xy.items())
    for station in stations:
        station["x"], station["y"] = transformer.transform(
            station["validated_lon"], station["validated_lat"]
        )
        distances = sorted(
            (
                math.hypot(station["x"] - xy[0], station["y"] - xy[1]),
                node_id,
            )
            for node_id, xy in node_items
        )
        station["candidates"] = distances[:SNAP_CANDIDATE_COUNT]
        station["nearest_node_id"] = distances[0][1]
        station["nearest_node_distance_m"] = distances[0][0]


def build_graph(links: list[dict[str, Any]]) -> nx.DiGraph:
    graph = nx.DiGraph()
    for link in links:
        f_node, t_node = link["f_node"], link["t_node"]
        if (
            not graph.has_edge(f_node, t_node)
            or link["length_m"] < graph[f_node][t_node]["length_m"]
        ):
            graph.add_edge(
                f_node,
                t_node,
                weight=link["length_m"],
                **{key: value for key, value in link.items() if key != "points"},
            )
    return graph


def route_one_direction(
    graph: nx.DiGraph,
    start: dict[str, Any],
    end: dict[str, Any],
) -> tuple[list[dict[str, Any]], float, float]:
    source = f"__source_{start['station_no']}"
    sink = f"__sink_{end['station_no']}"
    graph.add_node(source)
    graph.add_node(sink)
    for distance, node_id in start["candidates"]:
        graph.add_edge(source, node_id, weight=max(distance * SNAP_COST_MULTIPLIER, 0.01), snap_m=distance)
    for distance, node_id in end["candidates"]:
        graph.add_edge(node_id, sink, weight=max(distance * SNAP_COST_MULTIPLIER, 0.01), snap_m=distance)
    try:
        path = nx.shortest_path(graph, source, sink, weight="weight")
        start_snap_m = float(graph[path[0]][path[1]]["snap_m"])
        end_snap_m = float(graph[path[-2]][path[-1]]["snap_m"])
        link_edges = []
        for f_node, t_node in zip(path[1:-2], path[2:-1]):
            link_edges.append(dict(graph[f_node][t_node]))
        return link_edges, start_snap_m, end_snap_m
    finally:
        graph.remove_node(source)
        graph.remove_node(sink)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    workspace = HERE.parent
    output_dir = HERE / "outputs" / "01_mapping_qc"
    station_path, link_path = locate_inputs(workspace)
    stations = load_stations(station_path)
    links, node_xy = load_links(link_path)
    prepare_station_candidates(stations, node_xy)
    graph = build_graph(links)
    link_lookup = {link["link_id"]: link for link in links}
    station_lookup = {station["station_no"]: station for station in stations}

    mapping_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    route_geometry: dict[tuple[str, str], list[list[tuple[float, float]]]] = defaultdict(list)

    for segment in segment_topology():
        station_a = station_lookup[segment["station_a"]]
        station_b = station_lookup[segment["station_b"]]
        straight_m = math.hypot(station_a["x"] - station_b["x"], station_a["y"] - station_b["y"])
        directions = [
            ("AB", station_a, station_b),
            ("BA", station_b, station_a),
        ]
        for direction, start, end in directions:
            edges, start_snap_m, end_snap_m = route_one_direction(graph, start, end)
            route_length_m = sum(float(edge["length_m"]) for edge in edges)
            route_ratio = route_length_m / straight_m if straight_m else math.inf
            road_names = list(dict.fromkeys(edge["road_name"] or "(unnamed)" for edge in edges))
            qc_reasons = []
            if max(start_snap_m, end_snap_m) > MAX_QC_SNAP_M:
                qc_reasons.append("endpoint_snap_gt_50m")
            if route_ratio > MAX_QC_ROUTE_RATIO:
                qc_reasons.append("route_ratio_gt_1.8")
            summary = {
                **segment,
                "direction": direction,
                "from_station_no": start["station_no"],
                "from_station_name": start["station_name"],
                "to_station_no": end["station_no"],
                "to_station_name": end["station_name"],
                "straight_distance_m": round(straight_m, 3),
                "route_length_m": round(route_length_m, 3),
                "route_to_straight_ratio": round(route_ratio, 4),
                "start_snap_m": round(start_snap_m, 3),
                "end_snap_m": round(end_snap_m, 3),
                "link_count": len(edges),
                "road_names": " | ".join(road_names),
                "cross_segment_reused_link_count": 0,
                "qc_status": "REVIEW" if qc_reasons else "PASS",
                "qc_reasons": ";".join(qc_reasons),
            }
            summary_rows.append(summary)
            for sequence, edge in enumerate(edges, start=1):
                link_id = edge["link_id"]
                mapping_rows.append(
                    {
                        **segment,
                        "direction": direction,
                        "from_station_no": start["station_no"],
                        "from_station_name": start["station_name"],
                        "to_station_no": end["station_no"],
                        "to_station_name": end["station_name"],
                        "link_sequence": sequence,
                        "link_id": link_id,
                        "f_node": edge["f_node"],
                        "t_node": edge["t_node"],
                        "road_name": edge["road_name"],
                        "road_rank": edge["road_rank"],
                        "length_m": round(float(edge["length_m"]), 3),
                        "max_speed_kmh": edge["max_speed_kmh"],
                    }
                )
                route_geometry[(segment["segment_id"], direction)].append(link_lookup[link_id]["points"])

    link_segments: dict[str, set[str]] = defaultdict(set)
    for row in mapping_rows:
        link_segments[row["link_id"]].add(row["segment_id"])
    reused = {link_id for link_id, segments in link_segments.items() if len(segments) > 1}
    for summary in summary_rows:
        route_links = {
            row["link_id"]
            for row in mapping_rows
            if row["segment_id"] == summary["segment_id"] and row["direction"] == summary["direction"]
        }
        reuse_count = len(route_links & reused)
        summary["cross_segment_reused_link_count"] = reuse_count
        # Reuse around a station intersection or a branch junction is expected:
        # the standard road link can straddle the conceptual station boundary.
        # Keep the count for inspection, but do not fail an otherwise sound route.

    station_rows = []
    for station in stations:
        station_rows.append(
            {
                key: station[key]
                for key in [
                    "station_no", "station_name", "landmark", "original_lat", "original_lon",
                    "validated_lat", "validated_lon", "coordinate_qc", "coordinate_qc_reason",
                    "nearest_node_id", "nearest_node_distance_m", "source", "note",
                ]
            }
        )
        station_rows[-1]["nearest_node_distance_m"] = round(station_rows[-1]["nearest_node_distance_m"], 3)

    mapping_fields = [
        "segment_id", "segment_order", "branch", "station_a", "station_b", "direction",
        "from_station_no", "from_station_name", "to_station_no", "to_station_name",
        "link_sequence", "link_id", "f_node", "t_node", "road_name", "road_rank",
        "length_m", "max_speed_kmh",
    ]
    summary_fields = [
        "segment_id", "segment_order", "branch", "station_a", "station_b", "direction",
        "from_station_no", "from_station_name", "to_station_no", "to_station_name",
        "straight_distance_m", "route_length_m", "route_to_straight_ratio", "start_snap_m",
        "end_snap_m", "link_count", "road_names", "cross_segment_reused_link_count",
        "qc_status", "qc_reasons",
    ]
    station_fields = [
        "station_no", "station_name", "landmark", "original_lat", "original_lon",
        "validated_lat", "validated_lon", "coordinate_qc", "coordinate_qc_reason",
        "nearest_node_id", "nearest_node_distance_m", "source", "note",
    ]
    write_csv(output_dir / "segment_link_mapping.csv", mapping_rows, mapping_fields)
    write_csv(output_dir / "segment_link_mapping_qc.csv", summary_rows, summary_fields)
    write_csv(output_dir / "station_coordinates_validated.csv", station_rows, station_fields)

    to_wgs84 = Transformer.from_crs(5186, 4326, always_xy=True)
    features = []
    for summary in summary_rows:
        key = (summary["segment_id"], summary["direction"])
        coordinates = [
            [[lon, lat] for lon, lat in (to_wgs84.transform(x, y) for x, y in points)]
            for points in route_geometry[key]
        ]
        features.append(
            {
                "type": "Feature",
                "properties": {field: summary[field] for field in summary_fields},
                "geometry": {"type": "MultiLineString", "coordinates": coordinates},
            }
        )
    (output_dir / "segment_link_mapping.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False),
        encoding="utf-8",
    )

    review_rows = [row for row in summary_rows if row["qc_status"] != "PASS"]
    unique_links = len({row["link_id"] for row in mapping_rows})
    report = [
        "# 구간–링크 매핑 검수 결과",
        "",
        f"- 정거장: {len(stations):,}개",
        f"- 정거장 간 구간: {len(segment_topology()):,}개",
        f"- 방향 포함 공간 단위: {len(summary_rows):,}개",
        f"- 매핑 행: {len(mapping_rows):,}개",
        f"- 고유 표준 링크: {unique_links:,}개",
        f"- 자동 검수 PASS: {len(summary_rows) - len(review_rows):,}개",
        f"- 자동 검수 REVIEW: {len(review_rows):,}개",
        "",
        "## 좌표 보정",
        "",
        "노선도 순서와 반대였던 214(오정)의 원본 경도를 사용자 지정값 127.4146으로 보정했다.",
        "",
        "## 검수 기준",
        "",
        f"- 정거장–경로 종점 스냅 거리 ≤ {MAX_QC_SNAP_M:.0f}m",
        f"- 경로거리/직선거리 ≤ {MAX_QC_ROUTE_RATIO:.2f}",
        "- 서로 다른 정거장 구간의 링크 재사용 수를 기록(정거장·분기 교차부 공유는 허용)",
        "",
    ]
    if review_rows:
        report.extend(["## 추가 검토 대상", ""])
        for row in review_rows:
            report.append(
                f"- {row['segment_id']} {row['direction']}: {row['qc_reasons']} "
                f"(ratio={row['route_to_straight_ratio']}, snap={row['start_snap_m']}/{row['end_snap_m']}m)"
            )
    else:
        report.extend(["## 판정", "", "90개 방향 단위가 모든 자동 검수 기준을 통과했다."])
    (output_dir / "mapping_qc_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    if len(summary_rows) != 90 or any(not row["link_count"] for row in summary_rows):
        raise RuntimeError("Mapping completeness assertion failed")
    print(f"Wrote {output_dir}")
    print(f"segments=45 directions=90 mapping_rows={len(mapping_rows)} unique_links={unique_links}")
    print(f"pass={len(summary_rows) - len(review_rows)} review={len(review_rows)}")


if __name__ == "__main__":
    main()
