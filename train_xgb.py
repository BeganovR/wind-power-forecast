"""XGBoost v2: same as v1 but with wind veer features (135 features)."""
import os, pickle, warnings
import numpy as np, pandas as pd, yaml
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

from features import (add_features, add_lag_features, add_icing_accumulation,
                      add_power_curve_prior, FEATURE_COLS)

warnings.filterwarnings("ignore")
N_BAGS = 6
BAG_SEEDS = list(range(200, 206))
N_UST = 90.09
TARGET = "Выработка. Результирующий расчет"
WEIGHTS_DIR = "weights_xgb_v2"

XGB_PARAMS = {
    "n_estimators": 2200, "learning_rate": 0.015, "max_depth": 8,
    "min_child_weight": 3, "subsample": 0.8, "colsample_bytree": 0.8,
    "colsample_bylevel": 0.9, "reg_alpha": 0.1, "reg_lambda": 1.0,
    "objective": "reg:absoluteerror", "eval_metric": "mae",
    "tree_method": "hist", "n_jobs": -1, "verbosity": 0,
}


def compute_mae_pct(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)) / N_UST * 100)


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


with open("config.yaml") as f:
    cfg = yaml.safe_load(f)
dt_col = cfg["datetime_col"]
target_col = cfg["target_col"]
turbine_mw = cfg["turbine_capacity_mw"]
cf_clip = cfg["cf_clip_max"]

df_train = pd.read_csv("../data_extracted/dataset/train_dataset.csv", parse_dates=[dt_col])
df_train = df_train.sort_values(dt_col).reset_index(drop=True)

extra_path = "/home/beganovr/Work/Hackaton/3888f9f2-9bda-4b2c-94af-5562668bce86_test_dataset.csv"
df_extra = pd.read_csv(extra_path, parse_dates=[dt_col])
df_extra_april = df_extra[df_extra[dt_col].dt.month == 4].dropna(subset=[target_col]).copy()

df = pd.concat([df_train, df_extra_april], ignore_index=True)
df = df.sort_values(dt_col).reset_index(drop=True)

df = add_features(df)
df = add_lag_features(df, dt_col)
df = add_icing_accumulation(df, dt_col)
df["n_working"] = 26 - df["Кол-во_ВЭУ_в_ремонте"]
df["capacity"] = df["n_working"] * turbine_mw
df["cf"] = (df[target_col] / df["capacity"]).clip(0.0, cf_clip)

print(f"Данные: {len(df)} строк")

# Isotonic на полных данных
iso_80m = IsotonicRegression(increasing=True, out_of_bounds="clip")
iso_80m.fit(df["vp_80m"].values, df["cf"].values)
iso_120m = IsotonicRegression(increasing=True, out_of_bounds="clip")
iso_120m.fit(df["vp_120m"].values, df["cf"].values)
df = add_power_curve_prior(df, iso_80m, iso_120m)

fcols = [c for c in FEATURE_COLS if c in df.columns]
print(f"Признаков: {len(fcols)}")

X = df[fcols].values
y = df["cf"].values
sw = make_sample_weights(df)

os.makedirs(WEIGHTS_DIR, exist_ok=True)
with open(f"{WEIGHTS_DIR}/isotonic.pkl", "wb") as f:
    pickle.dump({"iso_80m": iso_80m, "iso_120m": iso_120m}, f)
with open(f"{WEIGHTS_DIR}/meta.pkl", "wb") as f:
    pickle.dump({"n_bags": N_BAGS, "best_iter": 2200, "fcols": fcols}, f)

print(f"Обучение {N_BAGS} XGB v2 мешков (iter=2200)...")
for i, seed in enumerate(BAG_SEEDS):
    mp = f"{WEIGHTS_DIR}/xgb_{i}.json"
    if os.path.exists(mp):
        print(f"  Мешок {i} уже есть, пропуск")
        continue
    params = dict(XGB_PARAMS, random_state=seed)
    print(f"  Мешок {i} (seed={seed})...", end="", flush=True)
    m = xgb.XGBRegressor(**params)
    m.fit(X, y, sample_weight=sw, verbose=False)
    m.save_model(mp)
    del m
    print(" готово")

print(f"XGB v2 сохранён в {WEIGHTS_DIR}/ ({N_BAGS} мешков)")
