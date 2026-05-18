"""
Прогноз выработки ВЭС на Q1 2026 с использованием ансамбля CatBoost (20 мешков).
Запуск: python predict_q1.py
"""

import pickle
import warnings
import numpy as np
import pandas as pd
import catboost as cb
import yaml

from features import (add_features, add_lag_features, add_icing_accumulation,
                      add_power_curve_prior, FEATURE_COLS)

warnings.filterwarnings("ignore")
TARGET = "Выработка. Результирующий расчет"
WEIGHTS_DIR = "weights_run19"


def predict(valid_path, output_path):
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    dt_col = cfg["datetime_col"]
    turbine_mw = cfg["turbine_capacity_mw"]
    cf_clip = cfg["cf_clip_max"]

    with open(f"{WEIGHTS_DIR}/meta.pkl", "rb") as f:
        meta = pickle.load(f)
    n_bags = meta["n_bags"]
    fcols = meta["fcols"]

    with open(f"{WEIGHTS_DIR}/isotonic.pkl", "rb") as f:
        iso_data = pickle.load(f)
    iso_80m = iso_data["iso_80m"]
    iso_120m = iso_data["iso_120m"]

    df = pd.read_csv(valid_path, parse_dates=[dt_col])
    df_sorted = df.sort_values(dt_col).reset_index(drop=True)
    df_sorted = add_features(df_sorted)
    df_sorted = add_lag_features(df_sorted, dt_col)
    df_sorted = add_icing_accumulation(df_sorted, dt_col)
    df_sorted["n_working"] = 26 - df_sorted["Кол-во_ВЭУ_в_ремонте"]
    df_sorted["capacity"] = df_sorted["n_working"] * turbine_mw
    df_sorted = add_power_curve_prior(df_sorted, iso_80m, iso_120m)

    actual_fcols = [c for c in fcols if c in df_sorted.columns]
    cap = df_sorted["capacity"].values
    cf_preds = np.zeros(len(df_sorted))

    for i in range(n_bags):
        m = cb.CatBoostRegressor()
        m.load_model(f"{WEIGHTS_DIR}/cb_{i}.cbm")
        cf_preds += m.predict(df_sorted[actual_fcols]).clip(0, cf_clip) / n_bags
        del m

    pred_mw = (cf_preds * cap).clip(0, None)
    df_sorted["pred_mw"] = pred_mw
    result = df[[dt_col]].merge(
        df_sorted[[dt_col, "pred_mw"]], on=dt_col, how="left"
    )["pred_mw"].values

    pd.DataFrame({TARGET: result}).to_csv(output_path, index=False)
    print(f"Сохранено: {output_path}  mean={result.mean():.2f}, std={result.std():.2f}")
    return result


if __name__ == "__main__":
    import os
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    # Q1 2026 прогноз
    valid_path = cfg.get("valid_path", "../data_extracted/dataset/valid_features.csv")
    predict(valid_path, "submission_q1_2026.csv")

    # Прогноз на май 2026
    extra_path = cfg.get("extra_path", "../data_extracted/dataset/test_dataset.csv")
    if os.path.exists(extra_path):
        df_extra = pd.read_csv(extra_path, parse_dates=["METEOFORECASTHOUR_OPENM_Datetime"])
        may = df_extra[df_extra["METEOFORECASTHOUR_OPENM_Datetime"].dt.month == 5].copy()
        may_features = may.drop(columns=[TARGET], errors="ignore")
        may_path = "/tmp/may_features.csv"
        may_features.sort_values("METEOFORECASTHOUR_OPENM_Datetime").to_csv(may_path, index=False)
        may_pred = predict(may_path, "submission_may18.csv")
        print(f"Прогноз май 2026: mean={may_pred.mean():.2f} МВт·ч")
