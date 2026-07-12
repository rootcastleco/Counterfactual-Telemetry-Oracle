#!/usr/bin/env python3
"""
counterfactual_telemetry_oracle.py

A standard-library-only prototype that:
1. Learns robust normal ranges from multivariate telemetry.
2. Detects anomalous rows.
3. Builds a lagged dependency sketch between sensors.
4. Produces counterfactual interventions:
   "What is the smallest sensor change that could have kept the system normal?"

This is not a causal proof engine. It is an engineering hypothesis generator.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


EPS = 1e-12


@dataclass
class RobustStats:
    median: float
    mad: float
    scale: float


@dataclass
class Edge:
    source: str
    target: str
    lag: int
    correlation: float
    slope: float
    intercept: float
    samples: int


@dataclass
class Intervention:
    kind: str
    variable: str
    current_value: float
    suggested_value: float
    delta: float
    normalized_cost: float
    rationale: str
    confidence: float


@dataclass
class Anomaly:
    row_index: int
    timestamp: Optional[str]
    score: float
    primary_sensor: str
    sensor_zscores: Dict[str, float]
    interventions: List[Intervention]


def median_absolute_deviation(values: Sequence[float], med: float) -> float:
    deviations = [abs(v - med) for v in values]
    return statistics.median(deviations) if deviations else 0.0


def robust_stats(values: Sequence[float]) -> RobustStats:
    med = statistics.median(values)
    mad = median_absolute_deviation(values, med)
    scale = max(1.4826 * mad, EPS)
    return RobustStats(median=med, mad=mad, scale=scale)


def robust_z(value: float, stats: RobustStats) -> float:
    return (value - stats.median) / stats.scale


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 3:
        return 0.0

    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]

    numerator = sum(a * b for a, b in zip(dx, dy))
    denom_x = math.sqrt(sum(a * a for a in dx))
    denom_y = math.sqrt(sum(b * b for b in dy))

    if denom_x < EPS or denom_y < EPS:
        return 0.0
    return numerator / (denom_x * denom_y)


def linear_fit(xs: Sequence[float], ys: Sequence[float]) -> Tuple[float, float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0, statistics.fmean(ys) if ys else 0.0

    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    var_x = sum((x - mx) ** 2 for x in xs)

    if var_x < EPS:
        return 0.0, my

    covariance = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = covariance / var_x
    intercept = my - slope * mx
    return slope, intercept


def parse_float(value: str) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def load_csv(
    path: Path,
    timestamp_column: Optional[str],
) -> Tuple[List[Dict[str, float]], List[Optional[str]], List[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV has no header.")

        rows: List[Dict[str, float]] = []
        timestamps: List[Optional[str]] = []
        numeric_columns: Optional[List[str]] = None

        for raw in reader:
            if numeric_columns is None:
                numeric_columns = []
                for key in reader.fieldnames:
                    if key == timestamp_column:
                        continue
                    if parse_float(raw.get(key, "")) is not None:
                        numeric_columns.append(key)

                if not numeric_columns:
                    raise ValueError("No numeric sensor columns were found.")

            parsed: Dict[str, float] = {}
            valid = True
            for column in numeric_columns:
                number = parse_float(raw.get(column, ""))
                if number is None:
                    valid = False
                    break
                parsed[column] = number

            if not valid:
                continue

            rows.append(parsed)
            timestamps.append(raw.get(timestamp_column) if timestamp_column else None)

    if len(rows) < 12:
        raise ValueError("At least 12 valid numeric rows are required.")

    assert numeric_columns is not None
    return rows, timestamps, numeric_columns


def generate_demo_csv(path: Path, samples: int = 320, seed: int = 7) -> None:
    random.seed(seed)
    pressure = 4.0
    temperature = 42.0
    vibration = 1.2

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "timestamp",
                "pressure_bar",
                "temperature_c",
                "vibration_mm_s",
                "current_a",
            ]
        )

        for i in range(samples):
            pressure += random.gauss(0, 0.025)
            temperature += 0.08 * (pressure - 4.0) + random.gauss(0, 0.18)
            vibration = (
                1.2
                + 0.55 * max(0.0, pressure - 4.1)
                + 0.015 * max(0.0, temperature - 42.0)
                + random.gauss(0, 0.07)
            )
            current = (
                8.0
                + 0.9 * pressure
                + 0.65 * vibration
                + random.gauss(0, 0.12)
            )

            if i in (170, 171, 172):
                pressure += 0.55
            if i == 230:
                vibration += 2.8
            if i == 270:
                temperature += 8.0

            writer.writerow(
                [
                    f"2026-07-12T{i // 60:02d}:{i % 60:02d}:00",
                    f"{pressure:.5f}",
                    f"{temperature:.5f}",
                    f"{vibration:.5f}",
                    f"{current:.5f}",
                ]
            )


class CounterfactualTelemetryOracle:
    def __init__(
        self,
        rows: List[Dict[str, float]],
        timestamps: List[Optional[str]],
        sensors: List[str],
        anomaly_threshold: float = 4.0,
        max_lag: int = 4,
        min_correlation: float = 0.45,
    ) -> None:
        self.rows = rows
        self.timestamps = timestamps
        self.sensors = sensors
        self.anomaly_threshold = anomaly_threshold
        self.max_lag = max_lag
        self.min_correlation = min_correlation

        self.stats: Dict[str, RobustStats] = {}
        self.normal_mask: List[bool] = []
        self.edges: List[Edge] = []

    def fit(self) -> None:
        self.stats = {
            sensor: robust_stats([row[sensor] for row in self.rows])
            for sensor in self.sensors
        }

        self.normal_mask = []
        for row in self.rows:
            score = max(
                abs(robust_z(row[sensor], self.stats[sensor]))
                for sensor in self.sensors
            )
            self.normal_mask.append(score < self.anomaly_threshold)

        for sensor in self.sensors:
            clean_values = [
                row[sensor]
                for row, is_normal in zip(self.rows, self.normal_mask)
                if is_normal
            ]
            if len(clean_values) >= 8:
                self.stats[sensor] = robust_stats(clean_values)

        self.normal_mask = []
        for row in self.rows:
            score = max(
                abs(robust_z(row[sensor], self.stats[sensor]))
                for sensor in self.sensors
            )
            self.normal_mask.append(score < self.anomaly_threshold)

        self.edges = self._learn_dependency_edges()

    def _learn_dependency_edges(self) -> List[Edge]:
        edges: List[Edge] = []

        for target in self.sensors:
            for source in self.sensors:
                if source == target:
                    continue

                best: Optional[Edge] = None

                for lag in range(self.max_lag + 1):
                    xs: List[float] = []
                    ys: List[float] = []

                    for t in range(lag, len(self.rows)):
                        source_index = t - lag
                        if not self.normal_mask[t] or not self.normal_mask[source_index]:
                            continue
                        xs.append(self.rows[source_index][source])
                        ys.append(self.rows[t][target])

                    if len(xs) < 12:
                        continue

                    corr = pearson(xs, ys)
                    if abs(corr) < self.min_correlation:
                        continue

                    slope, intercept = linear_fit(xs, ys)
                    candidate = Edge(
                        source=source,
                        target=target,
                        lag=lag,
                        correlation=corr,
                        slope=slope,
                        intercept=intercept,
                        samples=len(xs),
                    )

                    if best is None or abs(candidate.correlation) > abs(
                        best.correlation
                    ):
                        best = candidate

                if best is not None:
                    edges.append(best)

        pruned: List[Edge] = []
        for target in self.sensors:
            candidates = [edge for edge in edges if edge.target == target]
            candidates.sort(
                key=lambda edge: abs(edge.correlation),
                reverse=True,
            )
            pruned.extend(candidates[:3])

        return pruned

    def _normal_boundary(self, sensor: str, value: float) -> float:
        stats = self.stats[sensor]
        sign = 1.0 if value >= stats.median else -1.0
        return stats.median + sign * self.anomaly_threshold * stats.scale

    def _direct_intervention(self, sensor: str, current: float) -> Intervention:
        suggested = self._normal_boundary(sensor, current)
        delta = suggested - current
        cost = abs(delta) / self.stats[sensor].scale

        return Intervention(
            kind="direct",
            variable=sensor,
            current_value=current,
            suggested_value=suggested,
            delta=delta,
            normalized_cost=cost,
            rationale=(
                f"Move {sensor} directly to the nearest robust normal boundary."
            ),
            confidence=0.98,
        )

    def _parent_interventions(
        self,
        row_index: int,
        target: str,
        target_value: float,
    ) -> List[Intervention]:
        interventions: List[Intervention] = []
        desired_target = self._normal_boundary(target, target_value)

        for edge in self.edges:
            if edge.target != target or abs(edge.slope) < EPS:
                continue

            source_index = row_index - edge.lag
            if source_index < 0:
                continue

            current_source = self.rows[source_index][edge.source]
            suggested_source = (desired_target - edge.intercept) / edge.slope
            delta = suggested_source - current_source
            cost = abs(delta) / self.stats[edge.source].scale

            confidence = min(
                0.99,
                max(
                    0.05,
                    abs(edge.correlation) * min(1.0, edge.samples / 100.0),
                ),
            )

            interventions.append(
                Intervention(
                    kind="upstream",
                    variable=edge.source,
                    current_value=current_source,
                    suggested_value=suggested_source,
                    delta=delta,
                    normalized_cost=cost,
                    rationale=(
                        f"Via learned relation {edge.source}(t-{edge.lag}) "
                        f"-> {target}; r={edge.correlation:.3f}, "
                        f"slope={edge.slope:.5g}."
                    ),
                    confidence=confidence,
                )
            )

        return interventions

    def analyze(self, top_n: int = 10) -> List[Anomaly]:
        if not self.stats:
            raise RuntimeError("Call fit() before analyze().")

        anomalies: List[Anomaly] = []

        for index, row in enumerate(self.rows):
            zscores = {
                sensor: robust_z(row[sensor], self.stats[sensor])
                for sensor in self.sensors
            }
            primary = max(zscores, key=lambda name: abs(zscores[name]))
            score = abs(zscores[primary])

            if score < self.anomaly_threshold:
                continue

            candidates = [self._direct_intervention(primary, row[primary])]
            candidates.extend(
                self._parent_interventions(index, primary, row[primary])
            )

            candidates.sort(
                key=lambda item: (
                    item.normalized_cost / max(item.confidence, 0.05),
                    item.normalized_cost,
                )
            )

            anomalies.append(
                Anomaly(
                    row_index=index,
                    timestamp=self.timestamps[index],
                    score=score,
                    primary_sensor=primary,
                    sensor_zscores=zscores,
                    interventions=candidates[:4],
                )
            )

        anomalies.sort(key=lambda anomaly: anomaly.score, reverse=True)
        return anomalies[:top_n]

    def report(self, anomalies: List[Anomaly]) -> Dict[str, object]:
        return {
            "engine": "Counterfactual Telemetry Oracle",
            "warning": (
                "Dependency edges are statistical hypotheses, not causal proof. "
                "Validate interventions experimentally or against domain physics."
            ),
            "configuration": {
                "rows": len(self.rows),
                "sensors": self.sensors,
                "anomaly_threshold": self.anomaly_threshold,
                "max_lag": self.max_lag,
                "min_correlation": self.min_correlation,
            },
            "robust_stats": {
                name: asdict(stats) for name, stats in self.stats.items()
            },
            "dependency_edges": [asdict(edge) for edge in self.edges],
            "anomalies": [
                {
                    **{
                        key: value
                        for key, value in asdict(anomaly).items()
                        if key != "interventions"
                    },
                    "interventions": [
                        asdict(intervention)
                        for intervention in anomaly.interventions
                    ],
                }
                for anomaly in anomalies
            ],
        }


def print_summary(anomalies: Sequence[Anomaly]) -> None:
    if not anomalies:
        print("No anomaly crossed the configured threshold.")
        return

    print(f"\nDetected {len(anomalies)} high-priority anomalies:\n")

    for anomaly in anomalies:
        label = anomaly.timestamp or f"row {anomaly.row_index}"
        print(
            f"- {label}: {anomaly.primary_sensor} "
            f"(robust |z|={anomaly.score:.2f})"
        )

        for rank, intervention in enumerate(anomaly.interventions[:3], start=1):
            print(
                f"  {rank}. {intervention.kind}: {intervention.variable} "
                f"{intervention.current_value:.5g} -> "
                f"{intervention.suggested_value:.5g} "
                f"(delta={intervention.delta:+.5g}, "
                f"cost={intervention.normalized_cost:.2f}, "
                f"confidence={intervention.confidence:.2f})"
            )
        print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Detect telemetry anomalies and generate counterfactual interventions."
        )
    )
    parser.add_argument("csv_path", nargs="?", help="Input CSV path.")
    parser.add_argument(
        "--timestamp",
        default="timestamp",
        help="Timestamp column name. Default: timestamp",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=4.0,
        help="Robust z-score anomaly threshold. Default: 4.0",
    )
    parser.add_argument(
        "--max-lag",
        type=int,
        default=4,
        help="Maximum lag searched between sensors. Default: 4",
    )
    parser.add_argument(
        "--min-correlation",
        type=float,
        default=0.45,
        help="Minimum absolute correlation for a dependency edge.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Maximum number of anomalies in the report.",
    )
    parser.add_argument(
        "--output",
        default="counterfactual_report.json",
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Generate synthetic telemetry and analyze it.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.demo:
        input_path = Path("demo_telemetry.csv")
        generate_demo_csv(input_path)
        print(f"Generated demo data: {input_path}")
    elif args.csv_path:
        input_path = Path(args.csv_path)
    else:
        parser.error("Provide a CSV path or use --demo.")

    if args.threshold <= 0:
        parser.error("--threshold must be positive.")
    if args.max_lag < 0:
        parser.error("--max-lag cannot be negative.")
    if not 0.0 <= args.min_correlation <= 1.0:
        parser.error("--min-correlation must be between 0 and 1.")

    rows, timestamps, sensors = load_csv(input_path, args.timestamp)

    oracle = CounterfactualTelemetryOracle(
        rows=rows,
        timestamps=timestamps,
        sensors=sensors,
        anomaly_threshold=args.threshold,
        max_lag=args.max_lag,
        min_correlation=args.min_correlation,
    )
    oracle.fit()
    anomalies = oracle.analyze(top_n=args.top)

    output_path = Path(args.output)
    output_path.write_text(
        json.dumps(oracle.report(anomalies), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print_summary(anomalies)
    print(f"Full report written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
