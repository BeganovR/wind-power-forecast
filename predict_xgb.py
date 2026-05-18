"""Predict using XGBoost v2 ensemble (weights_xgb_v2/, with veer features)."""
import pickle, warnings
import numpy as np, pandas as pd
import xgboost as xgb
import yaml

from features import (add_features, add_lag_features, add_icing_accumulation,
                      add_power_curve_prior)

warnings.filterwarnings("ignore")
TARGET = "Выработка. Результирующий расчет"
WEIGHTS_DIR = "weights_xgb_v2"


def predict(valid_path, output_path):
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    dt_col = cfg["datetime_col"]
    turbine_mw = cfg["turbine_capacity_mw"]
    cf_clip = cfg["cf_clip_max"]

    with open(f"{WEIGHTS_DIR}/meta.pkl", "rb") as f:
        meta = pickle.load(f)
    with open(f"{WEIGHTS_DIR}/isotonic.pkl", "rb") as f:
        iso_data = pickle.load(f)

    df = pd.read_csv(valid_path, parse_dates=[dt_col])
    df_sorted = df.sort_values(dt_col).reset_index(drop=True)
    df_sorted = add_features(df_sorted)
    df_sorted = add_lag_features(df_sorted, dt_col)
    df_sorted = add_icing_accumulation(df_sorted, dt_col)
    df_sorted["n_working"] = 26 - df_sorted["Кол-во_ВЭУ_в_ремонте"]
    df_sorted["capacity"] = df_sorted["n_working"] * turbine_mw
    df_sorted = add_power_curve_prior(df_sorted, iso_data["iso_80m"], iso_data["iso_120m"])

    fcols = [c for c in meta["fcols"] if c in df_sorted.columns]
    cap = df_sorted["capacity"].values
    xgb_cf = np.zeros(len(df_sorted))

    for i in range(meta["n_bags"]):
        m = xgb.XGBRegressor()
        m.load_model(f"{WEIGHTS_DIR}/xgb_{i}.json")
        xgb_cf += m.predict(df_sorted[fcols].values).clip(0, cf_clip) / meta["n_bags"]
        del m

    pred_mw = (xgb_cf * cap).clip(0, None)
    df_sorted["pred_mw"] = pred_mw
    result = df[[dt_col]].merge(
        df_sorted[[dt_col, "pred_mw"]], on=dt_col, how="left"
    )["pred_mw"].values
    pd.DataFrame({TARGET: result}).to_csv(output_path, index=False)
    print(f"Сохранено: {output_path}  mean={result.mean():.2f}, std={result.std():.2f}")
    return result


if __name__ == "__main__":
    predict("../data_extracted/dataset/valid_features.csv", "submission_xgb_v2.csv")
