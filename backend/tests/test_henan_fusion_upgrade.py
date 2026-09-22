"""Controlled fusion of the two Henan price-forecasting implementations.

The upgrade keeps the strongest operational ideas from both versions:

* D-1 labels for pre-day-ahead real-time forecasting when the latest
  complete real-time labels are available by the forecasting deadline;
* XGBoost absolute-error (L1) blending for robust tails;
* a D-2 day-ahead forecast as a segment-specific pre-DA anchor;
* post-day-ahead spread prediction with a conservative day-ahead anchor;
* all weights are frozen on a selection interval before the test interval.

This is a research backtest. Weather and power forecast issue timestamps are
not present in the supplied snapshot, so it must not be treated as an as-of
production evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


XGB_PARAMS = dict(
    n_estimators=180,
    max_depth=5,
    learning_rate=0.035,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=8,
    reg_lambda=8.0,
    tree_method="hist",
    n_jobs=1,
    random_state=17,
    verbosity=0,
)
L1_PARAMS = {**XGB_PARAMS, "objective": "reg:absoluteerror"}
SQR_PARAMS = {**XGB_PARAMS, "objective": "reg:squarederror"}
CLIP = (-100.0, 2000.0)
ANCHOR_WEIGHTS = (0.0, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.0)
POST_WEIGHTS = (0.0, 0.10, 0.25, 0.50, 0.75, 1.0)
SEGMENTS = {
    "night": (range(1, 8), (0.0, 0.10, 0.20, 0.30, 0.40, 0.50)),
    "morning": (range(8, 12), (0.0, 0.10, 0.20, 0.30, 0.40)),
    "solar_core": (range(12, 18), (0.0, 0.10, 0.20)),
    "evening": (range(18, 25), (0.0, 0.10, 0.20, 0.25, 0.30, 0.40)),
}


def read_data(data_dir: Path) -> pd.DataFrame:
    def read(name: str) -> pd.DataFrame:
        rows = json.loads((data_dir / name).read_text(encoding="utf-8"))
        out = pd.DataFrame(rows)
        out["date"] = pd.to_datetime(out.pop("marketDate"))
        out["period"] = out["period"].astype(int)
        return out

    prices = read("prices_hourly.json").rename(
        columns={"dayAheadPriceYuanMwh": "da", "realTimePriceYuanMwh": "rt"}
    )
    power = read("power_forecast_hourly.json")
    renewable = read("renewable_forecast_hourly.json")
    weather = read("weather_hourly_province.json")
    for item in (power, renewable, weather):
        item.drop(columns=["cityCount"], errors="ignore", inplace=True)
    frame = prices.merge(power, on=["date", "period"], how="left")
    frame = frame.merge(renewable, on=["date", "period"], how="left", suffixes=("", "_renew"))
    frame = frame.merge(weather, on=["date", "period"], how="left", suffixes=("", "_weather"))
    frame["spread"] = frame["rt"] - frame["da"]
    frame["netLoadForecastMw"] = frame["loadForecastMw"] - frame["renewableMw"]
    frame = frame.sort_values(["date", "period"]).reset_index(drop=True)
    if frame.duplicated(["date", "period"]).any():
        raise ValueError("duplicate date/period keys in input")
    return frame


def build_features(frame: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, list[str]]:
    if mode == "day_ahead":
        target, cutoff, lags = "da", 1, (1, 2, 3, 7, 14)
    elif mode == "pre_rt":
        target, cutoff, lags = "rt", 1, (1, 2, 3, 7, 14)
    elif mode == "anchor_da":
        target, cutoff, lags = "da", 2, (2, 3, 4, 7, 14)
    elif mode == "post_rt":
        target, cutoff, lags = "spread", 1, (2, 3, 7, 14)
    else:
        raise ValueError(mode)
    out = frame[["date", "period"]].copy()
    out["target"] = pd.to_numeric(frame[target], errors="coerce")
    hour = frame["period"].astype(int) - 1
    date = frame["date"]
    out["hour"] = hour
    out["dow"] = date.dt.dayofweek
    out["month"] = date.dt.month
    out["dayofyear"] = date.dt.dayofyear
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * date.dt.dayofweek / 7)
    out["dow_cos"] = np.cos(2 * np.pi * date.dt.dayofweek / 7)
    out["is_weekend"] = (date.dt.dayofweek >= 5).astype(float)
    indexed = frame.set_index(["date", "period"])

    def lag(column: str, days: int) -> np.ndarray:
        keys = pd.MultiIndex.from_arrays([date - pd.Timedelta(days=days), frame["period"]])
        return indexed[column].reindex(keys).to_numpy(float)

    for days in lags:
        out[f"{target}_lag_{days}d"] = lag(target, days)
    if mode in {"pre_rt", "post_rt"}:
        for days in lags:
            out[f"da_lag_{days}d"] = lag("da", days)
    if mode == "post_rt":
        out["same_day_da"] = frame["da"].to_numpy(float)
        for days in lags:
            out[f"spread_lag_{days}d"] = lag("spread", days)
        out["same_day_da_ramp_1h"] = out["same_day_da"].groupby(out["date"]).diff()
        out["same_day_da_ramp_3h"] = out["same_day_da"].groupby(out["date"]).diff(3)
        out["same_day_da_daily_mean_gap"] = out["same_day_da"] - out.groupby("date")["same_day_da"].transform("mean")
    else:
        out["same_day_da"] = 0.0
    exog = [
        "loadForecastMw", "interconnectorMw", "totalOutputMw", "nonSpotOutputMw",
        "renewableMw", "hydroMw", "pumpedStorageMw", "windForecastMw", "solarForecastMw",
        "temperature2mC", "apparentTemperatureC", "precipitationMm", "windSpeed10mMs",
        "relativeHumidityPct", "netLoadForecastMw",
    ]
    for name in exog:
        out[name] = pd.to_numeric(frame.get(name), errors="coerce")
    out["renewable_share"] = out["renewableMw"] / out["loadForecastMw"].abs().where(out["loadForecastMw"].abs() >= 1)
    out["is_morning_peak"] = hour.between(7, 10).astype(float)
    out["is_evening_peak"] = hour.between(17, 22).astype(float)
    out["solar_window"] = hour.between(9, 16).astype(float)
    out["net_load_ramp_1h"] = out["netLoadForecastMw"].groupby(out["date"]).diff()
    out["net_load_ramp_3h"] = out["netLoadForecastMw"].groupby(out["date"]).diff(3)
    out = out.replace([np.inf, -np.inf], np.nan)
    columns = [c for c in out.columns if c not in {"date", "period", "target"}]
    return out, columns


def metric(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    error = np.asarray(prediction, float) - np.asarray(actual, float)
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(error.mean()),
        "n": int(len(error)),
    }


def train_rows(features: pd.DataFrame, columns: list[str], date: pd.Timestamp, cutoff_days: int) -> pd.DataFrame:
    cutoff = date - pd.Timedelta(days=cutoff_days)
    good = features["target"].notna() & features[columns].notna().sum(axis=1).ge(len(columns) - 2)
    rows = features.loc[good & features["date"].le(cutoff)].copy()
    if len(rows) < 24 * 21:
        raise ValueError("at least 21 complete historical days are required")
    return rows


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, columns: list[str], kind: str) -> np.ndarray:
    medians = train[columns].median().fillna(0.0)
    x_train = train[columns].fillna(medians).to_numpy(float)
    x_test = test[columns].fillna(medians).to_numpy(float)
    if kind == "ridge":
        model = make_pipeline(StandardScaler(), Ridge(alpha=20.0))
    else:
        model = XGBRegressor(**(L1_PARAMS if kind == "l1" else SQR_PARAMS))
    model.fit(x_train, train["target"].to_numpy(float))
    return np.clip(np.asarray(model.predict(x_test), float), *CLIP)


def daily_predictions(features: pd.DataFrame, columns: list[str], mode: str, start: str, end: str) -> pd.DataFrame:
    cutoff_days = {"day_ahead": 1, "pre_rt": 1, "anchor_da": 2, "post_rt": 1}[mode]
    dates = [d for d in pd.date_range(start, end) if (features["date"] == d).sum() == 24]
    outputs = []
    for i, date in enumerate(dates, 1):
        test = features.loc[features["date"].eq(date)].sort_values("period").copy()
        train = train_rows(features, columns, date, cutoff_days)
        validation_start = date - pd.Timedelta(days=cutoff_days + 20)
        validation = train.loc[train["date"] >= validation_start]
        fit_train = train.loc[train["date"] < validation_start]
        candidates = {name: fit_predict(fit_train, validation, columns, name) for name in ("xgb", "l1", "ridge")}
        scores = {name: metric(validation["target"].to_numpy(float), pred)["mae"] for name, pred in candidates.items()}
        best = min(scores, key=scores.get)
        # Robust L1 gets a chance to blend only when validation supports it.
        blend = 0.6 * candidates["l1"] + 0.4 * candidates[best]
        if metric(validation["target"].to_numpy(float), blend)["mae"] < scores[best]:
            base_name, base_val = "l1_blend", blend
        else:
            base_name, base_val = best, candidates[best]
        final_base = fit_predict(train, test, columns, "l1") if base_name == "l1_blend" else fit_predict(train, test, columns, base_name)
        if base_name == "l1_blend":
            final_base = 0.6 * final_base + 0.4 * fit_predict(train, test, columns, best)
        row = test[["date", "period", "target"]].copy()
        row["prediction"] = final_base
        row["base_model"] = base_name
        row["validation_mae"] = metric(validation["target"].to_numpy(float), base_val)["mae"]
        if mode == "post_rt":
            row["published_da"] = test["same_day_da"].to_numpy(float)
        outputs.append(row)
        if i == 1 or i % 7 == 0 or i == len(dates):
            print(f"{mode}: {i}/{len(dates)} days", flush=True)
    return pd.concat(outputs, ignore_index=True)


def choose_pre_anchor(selection: pd.DataFrame) -> dict[str, float]:
    chosen = {}
    for name, (periods, grid) in SEGMENTS.items():
        part = selection[selection["period"].isin(periods)]
        scores = []
        for weight in grid:
            pred = (1 - weight) * part["pre_prediction"] + weight * part["anchor_prediction"]
            scores.append((metric(part["actual"], pred)["mae"], weight))
        chosen[name] = min(scores)[1]
    return chosen


def apply_pre_anchor(rows: pd.DataFrame, weights: dict[str, float]) -> np.ndarray:
    out = rows["pre_prediction"].to_numpy(float).copy()
    for name, (periods, _) in SEGMENTS.items():
        mask = rows["period"].isin(periods).to_numpy()
        weight = weights[name]
        out[mask] = (1 - weight) * rows.loc[mask, "pre_prediction"] + weight * rows.loc[mask, "anchor_prediction"]
    return out


def choose_post_strategy(selection: pd.DataFrame) -> dict[str, Any]:
    reports = []
    for weight in POST_WEIGHTS:
        pred = (1 - weight) * selection["published_da"] + weight * (selection["published_da"] + selection["post_prediction"])
        reports.append((metric(selection["actual"], pred)["mae"], weight))
    _, weight = min(reports)
    # Per-period safety: only switch away from the global strategy with >=2% validation gain.
    rules = {}
    for period, part in selection.groupby("period"):
        base = (1 - weight) * part["published_da"] + weight * (part["published_da"] + part["post_prediction"])
        da = part["published_da"]
        gain = (metric(part["actual"], base)["mae"] - metric(part["actual"], da)["mae"]) / max(metric(part["actual"], base)["mae"], 1e-9)
        rules[str(int(period))] = "da" if gain >= 0.02 else "global"
    return {"spread_weight": float(weight), "period_rules": rules}


def apply_post_strategy(rows: pd.DataFrame, strategy: dict[str, Any]) -> np.ndarray:
    weight = float(strategy["spread_weight"])
    pred = (1 - weight) * rows["published_da"].to_numpy(float) + weight * (rows["published_da"] + rows["post_prediction"])
    for period, rule in strategy["period_rules"].items():
        if rule == "da":
            pred[rows["period"].to_numpy(int) == int(period)] = rows.loc[rows["period"].eq(int(period)), "published_da"]
    return np.clip(pred, *CLIP)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-start", default="2026-06-01")
    parser.add_argument("--selection-end", default="2026-08-07")
    parser.add_argument("--test-start", default="2026-08-08")
    parser.add_argument("--test-end", default="2026-09-06")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = read_data(args.data_dir)
    tables = {mode: build_features(frame, mode) for mode in ("pre_rt", "anchor_da", "post_rt")}
    sel_pre = daily_predictions(*tables["pre_rt"], "pre_rt", args.selection_start, args.selection_end)
    sel_anchor = daily_predictions(*tables["anchor_da"], "anchor_da", args.selection_start, args.selection_end)
    sel_post = daily_predictions(*tables["post_rt"], "post_rt", args.selection_start, args.selection_end)
    selection = sel_pre.rename(columns={"target": "actual", "prediction": "pre_prediction"})[["date", "period", "actual", "pre_prediction"]]
    selection = selection.merge(sel_anchor[["date", "period", "prediction"]].rename(columns={"prediction": "anchor_prediction"}), on=["date", "period"])
    selection = selection.merge(sel_post[["date", "period", "target", "prediction", "published_da"]].rename(columns={"target": "post_target", "prediction": "post_prediction"}), on=["date", "period"])
    pre_weights = choose_pre_anchor(selection)
    post_strategy = choose_post_strategy(selection)
    test_pre = daily_predictions(*tables["pre_rt"], "pre_rt", args.test_start, args.test_end)
    test_anchor = daily_predictions(*tables["anchor_da"], "anchor_da", args.test_start, args.test_end)
    test_post = daily_predictions(*tables["post_rt"], "post_rt", args.test_start, args.test_end)
    test = test_pre.rename(columns={"target": "actual", "prediction": "pre_prediction"})[["date", "period", "actual", "pre_prediction"]]
    test = test.merge(test_anchor[["date", "period", "prediction"]].rename(columns={"prediction": "anchor_prediction"}), on=["date", "period"])
    test = test.merge(test_post[["date", "period", "target", "prediction", "published_da"]].rename(columns={"target": "actual_post", "prediction": "post_prediction"}), on=["date", "period"])
    test["pre_upgraded"] = apply_pre_anchor(test, pre_weights)
    test["post_upgraded"] = apply_post_strategy(test, post_strategy)
    # post_rt is trained on spread, but is scored after adding the published DA
    # price against the actual real-time price from the same target date.
    test["actual_post"] = test["actual"].astype(float)
    metrics = {
        "protocol": {
            "pre_rt_label_cutoff": "D-1",
            "day_ahead_anchor_label_cutoff": "D-2",
            "post_rt_label_cutoff": "D-1",
            "selection": [args.selection_start, args.selection_end],
            "test": [args.test_start, args.test_end],
            "weather_and_power_issue_time_available": False,
        },
        "pre_rt": {
            "upgraded": metric(test["actual"], test["pre_upgraded"]),
            "direct_d1_base": metric(test["actual"], test["pre_prediction"]),
            "anchor_weights": pre_weights,
        },
        "post_rt": {
            "upgraded": metric(test["actual_post"], test["post_upgraded"]),
            "published_day_ahead": metric(test["actual_post"], test["published_da"]),
            "strategy": post_strategy,
        },
    }
    test.to_csv(args.output_dir / "test-predictions.csv", index=False)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "README.md").write_text(
        "# 河南融合升级版\n\n"
        "D-1盘前实时 + L1稳健融合 + 分时D-2日前锚定；事后实时采用D-1价差模型并以已公布日前价安全锚定。\n"
        "权重只在选择期冻结，测试期独立评估。天气和电力预测缺少issue_time，结果仍是研究回测。\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
