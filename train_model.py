"""
Run 19: Same as R18 but with wind veer features (dir_veer_* family).
Finds new best_iter via Q1 2025 OOF, then trains 20 bags on full data.
"""

import os
import warnings
import numpy as np
import pandas as pd
import yaml
import catboost as cb
from sklearn.isotonic import IsotonicRegression
import pickle

from features import (add_features, add_lag_features, add_icing_accumulation,
                      add_power_curve_prior, FEATURE_COLS)

warnings.filterwarnings("ignore")

N_BAGS = 20
BAG_SEEDS = list(range(42, 62))
N_UST = 90.09
TARGET = "Выработка. Результирующий расчет"
WEIGHTS_DIR = "weights_run19"

CB_REF_PARAMS = {
    "iterations": 3000,
    "learning_rate": 0.020,
    "depth": 9,
    "loss_function": "MAE",
    "eval_metric": "MAE",
    "early_stopping_rounds": 200,
    "verbose": 0,
    "thread_count": -1,
}


def compute_mae_pct(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)) / N_UST * 100)


def get_feature_cols(df):
    return [c for c in FEATURE_COLS if c in df.columns]


def make_sample_weights(df):
    month = df["METEOFORECASTHOUR_OPENM_Datetime"].dt.month.values
    n_repair = df["Кол-во_ВЭУ_в_ремонте"].values
    ws80 = df["wind_speed_80m"].values
    temp80 = df["temperature_80m"].values
    season_bonus = np.where(np.isin(month, [1, 2, 3]), 2.5,
                   np.where(np.isin(month, [4]), 1.8,
                   np.where(month == 12, 1.2, 1.0)))
    repair_bonus = np.where(n_repair == 2, 3.0,
                   np.where(n_repair == 3, 2.5,
                   np.where(n_repair == 4, 1.3,
                   np.where(n_repair == 5, 0.5, 0.7))))
    is_q1 = np.isin(month, [1, 2, 3])
    zone_bonus = np.where(is_q1 & (ws80 >= 6) & (ws80 < 12), 1.4,
                 np.where(is_q1 & (ws80 < 3), 0.7, 1.0))
    cold_bonus = np.where(is_q1 & (temp80 < 0), 1.5,
                 np.where(is_q1 & (temp80 < 4), 1.2, 1.0))
    icing_arr = df["icing_risk"].values if "icing_risk" in df.columns else np.zeros(len(df))
    icing_bonus = np.where(is_q1 & (icing_arr > 0), 1.5, 1.0)
    return df["capacity"].values * season_bonus * repair_bonus * zone_bonus * cold_bonus * icing_bonus


def run(train_path="../data_extracted/dataset/train_dataset.csv",
        extra_path="/home/beganovr/Work/Hackaton/3888f9f2-9bda-4b2c-94af-5562668bce86_test_dataset.csv",
        valid_path="../data_extracted/dataset/valid_features.csv"):

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    dt_col = cfg["datetime_col"]
    target_col = cfg["target_col"]
    turbine_mw = cfg["turbine_capacity_mw"]
    cf_clip = cfg["cf_clip_max"]

    df_train = pd.read_csv(train_path, parse_dates=[dt_col])
    df_train = df_train.sort_values(dt_col).reset_index(drop=True)

    df_extra = pd.read_csv(extra_path, parse_dates=[dt_col])
    df_extra_april = df_extra[
        df_extra[dt_col].dt.month == 4
    ].dropna(subset=[target_col]).copy()
    print(f"Train: {len(df_train)} rows, April 2026: {len(df_extra_april)} rows")

    df = pd.concat([df_train, df_extra_april], ignore_index=True)
    df = df.sort_values(dt_col).reset_index(drop=True)

    df = add_features(df)
    df = add_lag_features(df, dt_col)
    df = add_icing_accumulation(df, dt_col)
    df["n_working"] = 26 - df["Кол-во_ВЭУ_в_ремонте"]
    df["capacity"] = df["n_working"] * turbine_mw
    df["cf"] = (df[target_col] / df["capacity"]).clip(0.0, cf_clip)

    # Q1 2025 fold for best_iter
    fold_mask = (df[dt_col].dt.year == 2025) & (df[dt_col].dt.month.isin([1, 2, 3]))
    train_mask = df[dt_col] < df.loc[fold_mask, dt_col].min()

    df_tr = df[train_mask].copy()
    df_va = df[fold_mask].copy()

    iso_80m_ref = IsotonicRegression(increasing=True, out_of_bounds="clip")
    iso_80m_ref.fit(df_tr["vp_80m"].values, df_tr["cf"].values)
    iso_120m_ref = IsotonicRegression(increasing=True, out_of_bounds="clip")
    iso_120m_ref.fit(df_tr["vp_120m"].values, df_tr["cf"].values)
    df_tr = add_power_curve_prior(df_tr, iso_80m_ref, iso_120m_ref)
    df_va = add_power_curve_prior(df_va, iso_80m_ref, iso_120m_ref)

    fcols = get_feature_cols(df_tr)
    print(f"Признаков: {len(fcols)}")

    X_tr = df_tr[fcols].values
    y_tr = df_tr["cf"].values
    sw_tr = make_sample_weights(df_tr)
    X_va = df_va[fcols].values
    y_va = df_va["cf"].values
    cap_va = df_va["capacity"].values
    y_va_mw = df_va[target_col].values

    print("Поиск best_iter (early stopping на Q1 2025)...")
    ref_m = cb.CatBoostRegressor(**CB_REF_PARAMS, random_seed=42)
    ref_m.fit(X_tr, y_tr, sample_weight=sw_tr,
              eval_set=(X_va, y_va), verbose=False)
    best_iter = max(ref_m.best_iteration_, 300)
    print(f"  best_iter: {best_iter}")

    val_cf = ref_m.predict(X_va).clip(0, cf_clip)
    mae_ref = compute_mae_pct(y_va_mw, val_cf * cap_va)
    print(f"  Q1 2025 MAE (1 bag, ref): {mae_ref:.4f}%")
    del ref_m

    # Full training
    iso_80m_full = IsotonicRegression(increasing=True, out_of_bounds="clip")
    iso_80m_full.fit(df["vp_80m"].values, df["cf"].values)
    iso_120m_full = IsotonicRegression(increasing=True, out_of_bounds="clip")
    iso_120m_full.fit(df["vp_120m"].values, df["cf"].values)
    df = add_power_curve_prior(df, iso_80m_full, iso_120m_full)

    fcols_full = get_feature_cols(df)
    X_full = df[fcols_full].values
    y_full = df["cf"].values
    sw_full = make_sample_weights(df)

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    with open(f"{WEIGHTS_DIR}/isotonic.pkl", "wb") as f:
        pickle.dump({"iso_80m": iso_80m_full, "iso_120m": iso_120m_full}, f)
    with open(f"{WEIGHTS_DIR}/meta.pkl", "wb") as f:
        pickle.dump({"n_bags": N_BAGS, "best_iter": best_iter,
                     "fcols": fcols_full, "mae_q1_2025": mae_ref}, f)

    print(f"\nОбучение {N_BAGS} сумок (iter={best_iter})...")
    CB_FULL_PARAMS = {
        "iterations": best_iter,
        "learning_rate": 0.020,
        "depth": 9,
        "loss_function": "MAE",
        "eval_metric": "MAE",
        "verbose": 0,
        "thread_count": -1,
    }
    for i, seed in enumerate(BAG_SEEDS):
        model_path = f"{WEIGHTS_DIR}/cb_{i}.cbm"
        if os.path.exists(model_path):
            print(f"  Сумка {i} (seed={seed}) уже есть, пропуск")
            continue
        params = dict(CB_FULL_PARAMS, random_seed=seed)
        print(f"  Сумка {i} (seed={seed})...", end="", flush=True)
        m = cb.CatBoostRegressor(**params)
        m.fit(X_full, y_full, sample_weight=sw_full, verbose=False)
        m.save_model(model_path)
        del m
        print(" готово")

    print(f"\nR19 сохранён в {WEIGHTS_DIR}/ ({N_BAGS} сумок, best_iter={best_iter})")
    return best_iter


if __name__ == "__main__":
    run()
