"""Финальный бленд R19 + XGB v2 (оба с veer-признаками)."""
import numpy as np, pandas as pd, pickle, yaml

TARGET = "Выработка. Результирующий расчет"
N_UST = 90.09

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)
dt_col = cfg["datetime_col"]
turbine_mw = cfg["turbine_capacity_mw"]
cf_clip = cfg.get("cf_clip_max", 1.0)

valid = pd.read_csv("../data_extracted/dataset/valid_features.csv", parse_dates=[dt_col])
n_repair = valid["Кол-во_ВЭУ_в_ремонте"].values
n_working = 26 - n_repair
cap = n_working * turbine_mw

r19 = pd.read_csv("submission_run19.csv")[TARGET].values
xgb_v2 = pd.read_csv("submission_xgb_v2.csv")[TARGET].values
xgb_old = pd.read_csv("submission_xgb.csv")[TARGET].values
r7 = pd.read_csv("submission_run7_8.20pct.csv")[TARGET].values
knn = pd.read_csv("submission_knn.csv")[TARGET].values
et = pd.read_csv("submission_et.csv")[TARGET].values

print(f"R19:     mean={r19.mean():.2f}")
print(f"XGB v2:  mean={xgb_v2.mean():.2f}")
print(f"XGB old: mean={xgb_old.mean():.2f}")

# Основной бленд: R19(67%) + XGB_v2(33%)
b_main = 0.67 * r19 + 0.33 * xgb_v2
print(f"\nR19(67)+XGBv2(33): mean={b_main.mean():.2f}")

# 5-model: R19(44%) + R7(18%) + XGBv2(28%) + KNN(7%) + ET(3%)
b_5m = 0.44*r19 + 0.18*r7 + 0.28*xgb_v2 + 0.07*knn + 0.03*et
print(f"5-model с XGBv2: mean={b_5m.mean():.2f}")

# С избирательной апрельской калибровкой
try:
    with open("weights_april_calib/calibrators.pkl", "rb") as f:
        d = pickle.load(f)
    cal = d["global"]
    mask3 = n_repair == 3
    r19_sc = r19.copy()
    if mask3.sum() > 0:
        r19_sc[mask3] = (cal.predict((r19[mask3]/cap[mask3]).clip(0, cf_clip)) * cap[mask3]).clip(0)
    b_main_sc = 0.67 * r19_sc + 0.33 * xgb_v2
    b_5m_sc = 0.44*r19_sc + 0.18*r7 + 0.28*xgb_v2 + 0.07*knn + 0.03*et
    print(f"\nR19selcalib(67)+XGBv2(33): mean={b_main_sc.mean():.2f}")
    print(f"5-model selcalib с XGBv2: mean={b_5m_sc.mean():.2f}")
    pd.DataFrame({TARGET: b_main_sc}).to_csv("SUBMIT_FINAL_R19_XGBv2_selcalib.csv", index=False)
    pd.DataFrame({TARGET: b_5m_sc}).to_csv("SUBMIT_FINAL_5M_XGBv2_selcalib.csv", index=False)
except Exception as e:
    print(f"Калибровка недоступна: {e}")

pd.DataFrame({TARGET: b_main}).to_csv("SUBMIT_FINAL_R19_XGBv2.csv", index=False)
pd.DataFrame({TARGET: b_5m}).to_csv("SUBMIT_FINAL_5M_XGBv2.csv", index=False)

# Обновляем основной SUBMIT_Q1_2026
best_name = "SUBMIT_FINAL_5M_XGBv2.csv" if True else "SUBMIT_FINAL_R19_XGBv2.csv"
import shutil
shutil.copy("SUBMIT_FINAL_5M_XGBv2.csv", "SUBMIT_Q1_2026.csv")
print(f"\n✓ SUBMIT_Q1_2026.csv = {best_name}")
print(f"  mean={b_5m.mean():.2f}, rows={len(b_5m)}")
