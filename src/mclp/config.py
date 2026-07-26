"""
파라미터 설정 관리 모듈
======================
YAML 설정 파일 로딩 + 유효성 검증
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
import yaml
import logging

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG = BASE_DIR / "configs" / "mclp_config.yaml"


@dataclass
class SolverConfig:
    P: int = 10
    coverage_radius_km: float = 1.0
    alpha: float = 0.8
    D_min_m: float = 500.0
    time_limit_sec: int = 300
    amr_speed_kmh: float = 10.0


@dataclass
class CapacityConfig:
    parking_to_robot_ratio: float = 0.5
    min_capacity: int = 1


@dataclass
class SoftConstraintConfig:
    parking_incentive: float = 1.3
    parking_time_start: str = "10:00"
    parking_time_end: str = "17:00"
    slope_5pct_penalty: float = 0.7
    slope_10pct_penalty: float = 0.4
    enforcement_grace_incentive: float = 1.2
    grace_time_start: str = "11:30"
    grace_time_end: str = "13:30"
    school_zone_penalty: float = 0.6
    silver_zone_penalty: float = 0.7
    crosswalk_penalty_per_crossing: float = 0.95
    crosswalk_heavy_penalty_per_crossing: float = 0.90
    crosswalk_heavy_threshold: int = 5


@dataclass
class HardConstraintConfig:
    min_road_width_m: float = 4.0
    min_lanes: int = 2
    heavy_rain_threshold_mm: float = 30.0
    excavation_buffer_m: float = 100.0
    snow_threshold_cm: float = 3.0
    freezing_temp_c: float = 0.0
    fire_hydrant_buffer_m: float = 5.0
    min_sidewalk_width_m: float = 1.0
    max_delivery_time_min: float = 6.0
    capacity_enabled: bool = True


@dataclass
class WeightsConfig:
    high_risk_incentive: float = 1.5
    use_integrated_score: bool = True


@dataclass
class ScenarioConfig:
    name: str = "기본"
    P: int = 10
    coverage_radius_km: float = 1.0


@dataclass
class MCLPConfig:
    solver: SolverConfig = field(default_factory=SolverConfig)
    capacity: CapacityConfig = field(default_factory=CapacityConfig)
    soft: SoftConstraintConfig = field(default_factory=SoftConstraintConfig)
    hard: HardConstraintConfig = field(default_factory=HardConstraintConfig)
    weights: WeightsConfig = field(default_factory=WeightsConfig)
    scenarios: List[ScenarioConfig] = field(default_factory=list)


def validate_range(value: float, name: str, low: float = 0.0, high: float = 2.0):
    """패널티/인센티브 계수 유효 범위 검증."""
    if not (low <= value <= high):
        raise ValueError(
            f"파라미터 '{name}' 값 {value}이 허용 범위 [{low}, {high}]를 벗어남"
        )


def load_config(config_path: Optional[Path] = None) -> MCLPConfig:
    """설정 파일 로딩 및 검증.

    Args:
        config_path: YAML 설정 파일 경로. None이면 기본 경로 사용.

    Returns:
        MCLPConfig 인스턴스

    Raises:
        ValueError: 파라미터 유효 범위 위반 시
        FileNotFoundError: 설정 파일 미존재 시 (기본값 사용 안내)
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG

    config = MCLPConfig()

    if not config_path.exists():
        logger.warning(f"설정 파일 없음: {config_path}. 기본값 사용.")
        return config

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        logger.warning("설정 파일이 비어있음. 기본값 사용.")
        return config

    # Solver
    if "solver" in raw:
        s = raw["solver"]
        config.solver = SolverConfig(
            P=s.get("P", 10),
            coverage_radius_km=s.get("coverage_radius_km", 1.0),
            alpha=s.get("alpha", 0.8),
            D_min_m=s.get("D_min_m", 500.0),
            time_limit_sec=s.get("time_limit_sec", 300),
            amr_speed_kmh=s.get("amr_speed_kmh", 10.0),
        )

    # Capacity
    if "capacity" in raw:
        c = raw["capacity"]
        config.capacity = CapacityConfig(
            parking_to_robot_ratio=c.get("parking_to_robot_ratio", 0.5),
            min_capacity=c.get("min_capacity", 1),
        )

    # Soft constraints
    if "soft_constraints" in raw:
        sc = raw["soft_constraints"]
        config.soft = SoftConstraintConfig(
            parking_incentive=sc.get("parking_incentive", 1.3),
            parking_time_start=sc.get("parking_time_start", "10:00"),
            parking_time_end=sc.get("parking_time_end", "17:00"),
            slope_5pct_penalty=sc.get("slope_5pct_penalty", 0.7),
            slope_10pct_penalty=sc.get("slope_10pct_penalty", 0.4),
            enforcement_grace_incentive=sc.get("enforcement_grace_incentive", 1.2),
            grace_time_start=sc.get("grace_time_start", "11:30"),
            grace_time_end=sc.get("grace_time_end", "13:30"),
            school_zone_penalty=sc.get("school_zone_penalty", 0.6),
            silver_zone_penalty=sc.get("silver_zone_penalty", 0.7),
            crosswalk_penalty_per_crossing=sc.get("crosswalk_penalty_per_crossing", 0.95),
            crosswalk_heavy_penalty_per_crossing=sc.get("crosswalk_heavy_penalty_per_crossing", 0.90),
            crosswalk_heavy_threshold=sc.get("crosswalk_heavy_threshold", 5),
        )

    # Hard constraints
    if "hard_constraints" in raw:
        hc = raw["hard_constraints"]
        config.hard = HardConstraintConfig(
            min_road_width_m=hc.get("min_road_width_m", 4.0),
            min_lanes=hc.get("min_lanes", 2),
            heavy_rain_threshold_mm=hc.get("heavy_rain_threshold_mm", 30.0),
            excavation_buffer_m=hc.get("excavation_buffer_m", 100.0),
            snow_threshold_cm=hc.get("snow_threshold_cm", 3.0),
            freezing_temp_c=hc.get("freezing_temp_c", 0.0),
            fire_hydrant_buffer_m=hc.get("fire_hydrant_buffer_m", 5.0),
            min_sidewalk_width_m=hc.get("min_sidewalk_width_m", 1.0),
            max_delivery_time_min=hc.get("max_delivery_time_min", 6.0),
            capacity_enabled=hc.get("enabled", True),
        )

    # Weights
    if "weights" in raw:
        w = raw["weights"]
        config.weights = WeightsConfig(
            high_risk_incentive=w.get("high_risk_incentive", 1.5),
            use_integrated_score=w.get("use_integrated_score", True),
        )

    # Scenarios
    if "scenarios" in raw:
        config.scenarios = [
            ScenarioConfig(
                name=sc.get("name", f"scenario_{i}"),
                P=sc.get("P", config.solver.P),
                coverage_radius_km=sc.get("coverage_radius_km", config.solver.coverage_radius_km),
            )
            for i, sc in enumerate(raw["scenarios"])
        ]

    # 유효성 검증
    _validate(config)

    logger.info(f"설정 로딩 완료: P={config.solver.P}, R={config.solver.coverage_radius_km}km")
    return config


def _validate(config: MCLPConfig):
    """패널티/인센티브 계수 유효 범위 검증."""
    validate_range(config.soft.parking_incentive, "parking_incentive")
    validate_range(config.soft.slope_5pct_penalty, "slope_5pct_penalty")
    validate_range(config.soft.slope_10pct_penalty, "slope_10pct_penalty")
    validate_range(config.soft.enforcement_grace_incentive, "enforcement_grace_incentive")
    validate_range(config.soft.school_zone_penalty, "school_zone_penalty")
    validate_range(config.soft.silver_zone_penalty, "silver_zone_penalty")
    validate_range(config.soft.crosswalk_penalty_per_crossing, "crosswalk_penalty_per_crossing")
    validate_range(config.solver.alpha, "alpha", 0.0, 1.0)

    if config.solver.P < 1:
        raise ValueError(f"P는 1 이상이어야 함: {config.solver.P}")
    if config.solver.coverage_radius_km <= 0:
        raise ValueError(f"coverage_radius_km은 양수여야 함: {config.solver.coverage_radius_km}")
