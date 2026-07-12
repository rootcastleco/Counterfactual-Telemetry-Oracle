# Counterfactual Telemetry Oracle

A research-oriented Python prototype for robust multivariate telemetry anomaly detection and counterfactual intervention hypothesis generation.

Developed by [Rootcastle Engineering & Innovation](https://www.rootcastle.com/) and Batuhan Ayrıbaş. Research and engineering profile: [batuhanayribas.com](https://batuhanayribas.com/).

## Abstract

Conventional telemetry monitors generally answer one question: **Is the current observation anomalous?** Counterfactual Telemetry Oracle extends that workflow with a second, operationally useful question:

> What is the smallest measured-variable change that could have returned the observation to the learned normal region?

The prototype estimates robust normal ranges from multivariate sensor data, identifies high-deviation observations, constructs a sparse sketch of lagged statistical dependencies, and proposes ranked direct or upstream interventions. The implementation uses only the Python standard library and runs locally without network access.

This project is an engineering hypothesis generator. It does not establish causal relationships and must not be used as an autonomous safety controller.

## Research Question

Given a multivariate telemetry sequence

\[
X_t = \{x_{t,1}, x_{t,2}, \ldots, x_{t,p}\},
\]

can a lightweight system produce interpretable counterfactual hypotheses of the form:

> If sensor \(u\) at time \(t-k\) had changed by \(\Delta u\), sensor \(y\) at time \(t\) might have remained inside its robust normal boundary?

The current implementation approaches this question through robust descriptive statistics, lagged correlation screening, ordinary least-squares approximations, and normalized intervention costs.

## Method

### 1. Robust normal-region estimation

For each sensor \(j\), the system calculates the median

\[
m_j = \operatorname{median}(x_j)
\]

and median absolute deviation

\[
MAD_j = \operatorname{median}(|x_j - m_j|).
\]

The robust scale estimate is

\[
s_j = \max(1.4826 \cdot MAD_j, \epsilon),
\]

where the factor 1.4826 makes MAD comparable to the standard deviation under an approximately Gaussian distribution.

The robust standardized deviation is

\[
z_{t,j} = \frac{x_{t,j} - m_j}{s_j}.
\]

An observation is considered anomalous when

\[
\max_j |z_{t,j}| \geq \tau,
\]

where \(\tau\) is the configurable anomaly threshold.

The model performs an initial fit, excludes likely-anomalous rows, and refits the robust statistics to reduce contamination.

### 2. Lagged dependency sketch

For every ordered sensor pair \(u \rightarrow y\), the system evaluates lags from zero to `max_lag`. Candidate edges are retained when the absolute Pearson correlation exceeds `min_correlation` on likely-normal observations.

For each retained edge, the model estimates

\[
y_t \approx a u_{t-k} + b
\]

using a univariate least-squares fit. At most three strongest incoming edges are retained for each target sensor to limit explanation noise.

These edges represent statistical dependency hypotheses, not verified causal mechanisms.

### 3. Counterfactual intervention generation

For an anomalous sensor value \(y_t\), the nearest robust boundary is

\[
y^* = m_y + \operatorname{sign}(y_t-m_y)\tau s_y.
\]

A direct intervention proposes changing \(y_t\) to \(y^*\).

For an upstream edge \(u_{t-k} \rightarrow y_t\), the estimated source value required to reach that boundary is

\[
u^* = \frac{y^* - b}{a}.
\]

The normalized intervention cost is

\[
C = \frac{|u^*-u|}{s_u}.
\]

Candidates are ranked using normalized cost adjusted by an empirical confidence score derived from correlation magnitude and sample count.

## Processing Pipeline

```text
CSV telemetry
    |
    v
Numeric-column discovery and validation
    |
    v
Robust median/MAD fit
    |
    v
Likely-normal subset and robust refit
    |
    +--------------------------+
    |                          |
    v                          v
Anomaly scoring        Lagged dependency sketch
    |                          |
    +-------------+------------+
                  |
                  v
      Counterfactual candidates
                  |
                  v
       Cost/confidence ranking
                  |
                  v
          JSON engineering report
```

## Properties

- Standard-library-only Python implementation
- Local execution with no telemetry upload
- Robust median and MAD-based anomaly scoring
- Two-pass fitting to reduce anomaly contamination
- Configurable lag search and dependency threshold
- Direct and upstream counterfactual hypotheses
- Human-readable terminal summary
- Machine-readable JSON report
- Deterministic synthetic demo data through a fixed random seed

## Requirements

- Python 3.9 or newer
- No external packages

## Quick Start

Clone the repository and run the deterministic demo:

```bash
git clone https://github.com/rootcastleco/Counterfactual-Telemetry-Oracle.git
cd Counterfactual-Telemetry-Oracle
python3 counterfactual_telemetry_oracle.py --demo
```

The command generates:

```text
demo_telemetry.csv
counterfactual_report.json
```

Analyze an existing CSV file:

```bash
python3 counterfactual_telemetry_oracle.py telemetry.csv \
  --timestamp timestamp \
  --threshold 4.0 \
  --max-lag 8 \
  --min-correlation 0.40 \
  --top 20 \
  --output counterfactual_report.json
```

## Input Format

The first row must contain column names. The timestamp column is optional; all detected sensor columns must contain finite numeric values.

```csv
timestamp,pressure_bar,temperature_c,vibration_mm_s,current_a
2026-07-12T10:00:00,4.01,42.2,1.18,12.4
2026-07-12T10:00:01,4.04,42.3,1.21,12.6
```

Current parser behavior:

- Requires at least 12 valid rows
- Detects numeric sensor columns from the first data row
- Skips rows containing invalid or non-finite values in detected sensor columns
- Uses `timestamp` as the default timestamp-column name

## Output Structure

The JSON report contains:

- Runtime configuration
- Robust statistics for every sensor
- Learned lagged dependency edges
- Ranked anomalies
- Per-sensor robust z-scores
- Direct and upstream intervention candidates
- Suggested values, deltas, normalized costs, rationales, and confidence estimates

Example intervention object:

```json
{
  "kind": "upstream",
  "variable": "pressure_bar",
  "current_value": 4.61,
  "suggested_value": 4.18,
  "delta": -0.43,
  "normalized_cost": 2.71,
  "rationale": "Via learned relation pressure_bar(t-3) -> vibration_mm_s",
  "confidence": 0.74
}
```

## Reproducibility Check

```bash
python3 -m py_compile counterfactual_telemetry_oracle.py
python3 counterfactual_telemetry_oracle.py --demo --top 5 --output report.json
python3 -c "import json; d=json.load(open('report.json')); assert d['anomalies']; print('reproducibility check passed')"
```

The included demonstration dataset is synthetic. It is intended to verify execution and report generation, not to establish real-world diagnostic accuracy.

## Scientific and Engineering Limitations

1. **Correlation is not causation.** A retained dependency edge may be caused by confounding, common inputs, feedback loops, or temporal drift.
2. **The regressions are univariate and linear.** Nonlinear, multivariate, hysteretic, and regime-dependent processes are not represented.
3. **Stationarity is assumed implicitly.** Sensor distributions and relationships may change over time.
4. **Autocorrelation can inflate apparent evidence.** Confidence values are heuristic and are not calibrated probabilities.
5. **No actuator constraints are modeled.** A statistically small intervention may be physically impossible or unsafe.
6. **No uncertainty interval is produced.** Suggested values are point estimates rather than confidence or credible intervals.
7. **Missing-data handling is conservative.** Invalid rows are discarded instead of imputed.

## Recommended Validation Protocol

For real deployment, evaluate the system with a domain-specific protocol:

1. Define normal and fault regimes using independently verified labels.
2. Split data by time or machine identity to prevent leakage.
3. Report precision, recall, false alarms per operating hour, and detection delay.
4. Compare against fixed thresholds, isolation forest, change-point detection, and domain rules.
5. Validate proposed interventions against physical models or controlled experiments.
6. Perform sensitivity analysis over `threshold`, `max_lag`, and `min_correlation`.
7. Reject any intervention that violates equipment, process, or safety constraints.

## Safety and Threat Model

The current program is an offline analytical tool. It does not control equipment, open network connections, or execute content from the CSV file. Primary risks are:

- Misinterpreting statistical associations as causal instructions
- Resource exhaustion from very large files or high-dimensional telemetry
- Data-quality failures, unit mismatches, and sensor calibration drift
- Disclosure of sensitive industrial telemetry through copied reports

Recommended controls include file-size limits, schema validation, unit metadata, access control for reports, domain-rule filters, and mandatory human approval before operational use.

## Project Status

Experimental research prototype. The current release is suitable for reproducible demonstrations, algorithm inspection, and development of more rigorous causal and temporal models. It is not validated for safety-critical, medical, financial, aviation, automotive control, or industrial protection functions.

## Roadmap

- Rolling-window and regime-aware robust statistics
- Multivariate and nonlinear dependency models
- Bootstrap uncertainty intervals
- Constraint-aware intervention optimization
- Explicit causal graph import and domain-rule enforcement
- Streaming telemetry adapters
- Unit, property, and benchmark test suites
- Machine-specific calibration and drift monitoring

## Branding and Attribution

**Rootcastle Engineering & Innovation** develops software, telemetry, IoT, data infrastructure, and engineering prototypes with an emphasis on testability, observability, and disciplined system design.

Project author and research direction: **Batuhan Ayrıbaş**  
Website: [batuhanayribas.com](https://batuhanayribas.com/)  
Company: [rootcastle.com](https://www.rootcastle.com/)

## Citation

When referencing this prototype, use:

```text
Ayrıbaş, B. (2026). Counterfactual Telemetry Oracle: A Robust Telemetry
Anomaly Detection and Counterfactual Intervention Prototype.
Rootcastle Engineering & Innovation.
https://github.com/rootcastleco/Counterfactual-Telemetry-Oracle
```

## License

Released under the MIT License. See [LICENSE](LICENSE).
