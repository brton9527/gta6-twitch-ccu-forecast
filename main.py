"""Forecast GTA VI's launch-day Twitch viewers, hour by hour (make.fun
gta6-twitch-ccu-forecast-01).

Model, in plain words
--------------------
Launch days on Twitch have a shape, not a level. Hour 1 of the UTC release
day (00:00-01:00) only holds the regions that already unlocked at midnight
local (New Zealand, Australia, Asia, Europe, the UK); the United States
unlocks at 05:00Z (East) and 08:00Z (West), and the evening prime time then
lifts the count far above hour 1. So the forecast is built from the
launch-day *shapes* of past big launches, not from hour 1 alone:

1. Shape curves. For every comparable launch in the case file, take its
   release-day hourly viewers and form log ratios to its own hour 1,
   lr[h] = ln(viewers_h / viewers_1), hours 2..24.
2. Size regression. Per hour, fit a quadratic of lr[h] on log10(hour-1
   viewers) across the comparables. Big launches flatten out: a game whose
   hour 1 is already huge grows less over the day than one whose hour 1 is
   small. The case's hour 1 is clamped to the comparables' observed range,
   so a record-breaking hour 1 never extrapolates the curve.
3. Hype correction. A second-stage regression of the residuals on
   ln(hour1 / pre-launch 48-hour median viewers): a game whose hour 1 jumps
   far above its own pre-launch baseline behaves differently from one that
   was already heavily watched before release. The pre-launch median is
   taken over the 48 hours before release, the same window the final case's
   history covers.
4. Big-launch shape bias. The comparable pool's average shape is flatter
   than real big launches: on the four practice launches the model above
   over-predicts hours 3-4 and under-predicts the US-unlock jump (hours
   5-8) and the evening peak (hours 17-24). BIAS below is half the mean
   log-error per hour over those four practice launches (leave-one-case-out
   validated: it cut the practice loss from 0.158 to 0.142).
5. Spread. The log spread per hour is the residual standard deviation of
   comparables whose hour 1 is within 1 dex of the case's, times 0.7,
   floored at 0.15. Quantiles are the median times exp(z_tau * sigma).

Local practice scores (SullyGnome hourly averages, pinball loss on the log
scale; house baseline 0.387): leave-one-case-out 0.142 overall
(p00 0.140, p01 0.153, p02 0.110, p03 0.164), typical miss 15-31%,
90% coverage 0.78-1.00.

Python 3.13, standard library only, no network, deterministic: no
randomness, files read in name order, no dict-order dependence.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Half the mean log-error per hour of this model (without the bias) over the
# four practice launches (Wolverine, RDR2, God of War Ragnarok,
# Spider-Man 2), fitted on SullyGnome hourly averages. Positive means the
# model under-predicted that hour on big launches.
BIAS = {
    2: -0.065, 3: -0.134, 4: -0.167, 5: 0.115, 6: 0.166, 7: 0.105,
    8: 0.040, 9: 0.002, 10: -0.042, 11: -0.057, 12: -0.150, 13: -0.090,
    14: -0.022, 15: 0.039, 16: 0.068, 17: 0.132, 18: 0.199, 19: 0.156,
    20: 0.136, 21: 0.132, 22: 0.122, 23: 0.097, 24: 0.074,
}

SIGMA_SCALE = 0.7
SIGMA_FLOOR = 0.15


def round2(x):
    """Round to 2 decimals, half up, as JavaScript's Math.round does."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value == int(value) else value


def parse_time(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def median(values):
    s = sorted(values)
    n = len(s)
    if n == 0:
        return None
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def solve_linear(matrix, rhs):
    """Gaussian elimination for a small dense system; None if singular."""
    n = len(rhs)
    aug = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
        aug[col], aug[piv] = aug[piv], aug[col]
        if abs(aug[col][col]) < 1e-12:
            return None
        for r in range(n):
            if r != col:
                f = aug[r][col] / aug[col][col]
                for k in range(col, n + 1):
                    aug[r][k] -= f * aug[col][k]
    return [aug[i][n] / aug[i][i] for i in range(n)]


def fit_poly(xs, ys, degree):
    """Least-squares polynomial fit; falls back to a lower degree."""
    for deg in range(degree, 0, -1):
        m = deg + 1
        matrix = [[sum(x ** (i + j) for x in xs) for j in range(m)] for i in range(m)]
        rhs = [sum(y * (x ** i) for x, y in zip(xs, ys)) for i in range(m)]
        coef = solve_linear(matrix, rhs)
        if coef is not None:
            return coef
    return [sum(ys) / len(ys)]


def eval_poly(coef, x):
    return sum(c * (x ** i) for i, c in enumerate(coef))


def ols1(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx if sxx > 0 else 0.0
    return my - b * mx, b


def launch_day_curve(comp):
    """Hour (1..24) -> viewers on the comparable's release UTC day."""
    day_start = parse_time(comp["day_start"])
    out = {}
    for point in comp["hourly"]:
        viewers = point.get("viewers")
        if viewers is None:
            continue
        hour = int((parse_time(point["from"]) - day_start).total_seconds() // 3600) + 1
        if 1 <= hour <= 24:
            out[hour] = viewers
    return out if len(out) >= 20 else None


def prelaunch_median(points, day_start, hours=48):
    """Median viewers in the `hours` before release day, from (time, viewers)."""
    start = day_start - timedelta(hours=hours)
    vals = [v for t, v in points if start <= t < day_start and v and v > 0]
    return median(vals)


def smooth(values):
    n = len(values)
    return [
        (values[max(0, i - 1)] + values[i] + values[min(n - 1, i + 1)]) / 3
        for i in range(n)
    ]


def baseline_quantiles(p1, hour):
    """House persistence fallback: flat median, widening log spread."""
    return [
        round2(max(1.0, p1 * math.exp(z * 0.15 * math.sqrt(hour - 1))))
        for z in Z
    ]


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    day_start = parse_time(case["day_start"])

    # Comparable launch-day curves, skipping the GTA VI category's own
    # pre-release history (it is not a launch day). Each comparable's
    # hype feature (hour 1 over its pre-launch 48h median) is computed
    # once here, not once per hour: the per-hour loop below is the hot
    # path (the grader allows 8 seconds per case on 1 CPU).
    curves = []
    for comp in case.get("comparables") or []:
        if "Grand Theft Auto VI" in comp.get("name", ""):
            continue
        curve = launch_day_curve(comp)
        if curve is None or curve.get(1, 0) <= 0:
            continue
        points = [(parse_time(p["from"]), p["viewers"]) for p in comp["hourly"]]
        pre = prelaunch_median(points, parse_time(comp["day_start"]))
        feat = math.log(curve[1] / pre) if pre else None
        curves.append((curve, math.log10(curve[1]), feat))

    hours = [row["hour"] for row in case["forecast"]]
    if len(curves) < 5:
        return {"hours": [{"hour": h, "q": baseline_quantiles(p1, h)} for h in hours]}

    x_all = [x for _, x, _ in curves]
    x0 = min(max(math.log10(p1), min(x_all)), max(x_all))

    # The case's own hype feature: hour 1 over its pre-launch 48h median.
    hist_points = [
        (parse_time(p["from"]), p["viewers"])
        for p in case.get("history_hourly") or []
    ]
    pre_case = prelaunch_median(hist_points, day_start)
    feat_case = math.log(p1 / pre_case) if pre_case else None

    medians = []
    sigmas = []
    for hour in hours:
        rows = []
        for curve, x1, feat in curves:
            if hour not in curve:
                continue
            rows.append((x1, math.log(curve[hour] / curve[1]), feat))
        if len(rows) < 5:
            medians.append(0.0)
            sigmas.append(0.15 * math.sqrt(hour - 1))
            continue
        xs = [r[0] for r in rows]
        ys = [r[1] for r in rows]
        coef = fit_poly(xs, ys, 2)
        pred = eval_poly(coef, x0)
        if feat_case is not None:
            pairs = [
                (r[2], y - eval_poly(coef, r[0]))
                for r, y in zip(rows, ys)
                if r[2] is not None
            ]
            if len(pairs) > 5:
                a2, b2 = ols1([p[0] for p in pairs], [p[1] for p in pairs])
                pred += a2 + b2 * feat_case
        pred += BIAS.get(hour, 0.0)
        near = [
            y - eval_poly(coef, x) for x, y in zip(xs, ys) if abs(x - x0) <= 1.0
        ]
        if len(near) < 5:
            near = [y - eval_poly(coef, x) for x, y in zip(xs, ys)]
        var = sum(r * r for r in near) / max(len(near) - len(coef), 1)
        sigma = max(math.sqrt(var) * SIGMA_SCALE, SIGMA_FLOOR)
        medians.append(pred)
        sigmas.append(sigma)

    medians = smooth(medians)
    out_hours = []
    for hour, med, sigma in zip(hours, medians, sigmas):
        qs = []
        for z in Z:
            q = round2(max(1.0, p1 * math.exp(med + z * sigma)))
            q = min(q, 1_000_000_000)
            qs.append(q)
        # Enforce non-decreasing quantiles after rounding.
        for i in range(1, 5):
            if qs[i] < qs[i - 1]:
                qs[i] = qs[i - 1]
        out_hours.append({"hour": hour, "q": qs})
    return {"hours": out_hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
