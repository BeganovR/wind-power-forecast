"""
Feature engineering for wind power prediction.
All transformations are applied identically to train and validation data.
"""

import numpy as np
import pandas as pd


WIND_DIR_SCALE = 1000.0  # raw direction * 1000 = degrees
N_TURBINES = 26
TURBINE_MW = 3.465
HUB_HEIGHT = 80  # meters

SPEED_COLS = ["wind_speed_10m", "wind_speed_80m", "wind_speed_120m", "wind_speed_180m"]
DIR_COLS = ["wind_direction_10m", "wind_direction_80m", "wind_direction_120m", "wind_direction_180m"]
HEIGHTS = [10, 80, 120, 180]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── impute missing 180m data using power law from 80m/120m ───────────────
    if "wind_speed_180m" in df.columns:
        nan_mask = df["wind_speed_180m"].isna()
        if nan_mask.any():
            eps = 1e-3
            alpha_80_120 = (np.log(df["wind_speed_120m"].clip(eps) / df["wind_speed_80m"].clip(eps))
                            / np.log(120 / 80))
            alpha_80_120 = alpha_80_120.clip(0, 0.6)
            v180_imputed = df["wind_speed_120m"] * (180 / 120) ** alpha_80_120
            df.loc[nan_mask, "wind_speed_180m"] = v180_imputed[nan_mask]

    if "wind_direction_180m" in df.columns:
        nan_mask = df["wind_direction_180m"].isna()
        if nan_mask.any():
            df.loc[nan_mask, "wind_direction_180m"] = df.loc[nan_mask, "wind_direction_120m"]

    # ── working turbines ────────────────────────────────────────────────────
    df["n_working"] = N_TURBINES - df["Кол-во_ВЭУ_в_ремонте"]
    df["availability"] = df["n_working"] / N_TURBINES

    # ── air density (kg/m³) via ideal gas ───────────────────────────────────
    df["T_K_80m"] = df["temperature_80m"] + 273.15
    df["T_K_120m"] = df["temperature_120m"] + 273.15
    df["rho"] = df["pressure_msl"] * 100.0 / (287.0 * df["T_K_80m"])
    df["rho_norm"] = df["rho"] / 1.225

    # ── wind power proxy (physics: P ∝ rho * v³) ───────────────────────────
    df["vp_80m"] = df["rho_norm"] * df["wind_speed_80m"] ** 3
    df["vp_120m"] = df["rho_norm"] * df["wind_speed_120m"] ** 3

    # ── wind speed squared and cubed ─────────────────────────────────────────
    for col, h in zip(SPEED_COLS, HEIGHTS):
        df[f"v2_{h}m"] = df[col] ** 2
        df[f"v3_{h}m"] = df[col] ** 3

    # ── vertical wind shear (Hellmann exponent) ─────────────────────────────
    eps = 1e-3
    df["shear_10_80"] = np.log(df["wind_speed_80m"].clip(eps) / df["wind_speed_10m"].clip(eps)) / np.log(80 / 10)
    df["shear_10_180"] = np.log(df["wind_speed_180m"].clip(eps) / df["wind_speed_10m"].clip(eps)) / np.log(180 / 10)
    df["shear_80_120"] = np.log(df["wind_speed_120m"].clip(eps) / df["wind_speed_80m"].clip(eps)) / np.log(120 / 80)
    df["shear_80_180"] = np.log(df["wind_speed_180m"].clip(eps) / df["wind_speed_80m"].clip(eps)) / np.log(180 / 80)
    df["speed_spread"] = df["wind_speed_180m"] - df["wind_speed_10m"]

    # ── estimated hub-height speed via power law ─────────────────────────────
    df["v_hub_estimated"] = df["wind_speed_10m"] * (HUB_HEIGHT / 10) ** df["shear_10_80"].clip(0.0, 0.5)

    # ── interpolated wind at 100m ────────────────────────────────────────────
    alpha_80_120 = df["shear_80_120"].clip(0.0, 0.6)
    df["wind_speed_100m"] = df["wind_speed_80m"] * (100.0 / 80.0) ** alpha_80_120
    df["vp_100m"] = df["rho_norm"] * df["wind_speed_100m"] ** 3

    # ── wind direction → cyclic sin/cos ──────────────────────────────────────
    for raw_col, h in zip(DIR_COLS, HEIGHTS):
        deg = df[raw_col] * WIND_DIR_SCALE
        rad = np.deg2rad(deg)
        df[f"dir_sin_{h}m"] = np.sin(rad)
        df[f"dir_cos_{h}m"] = np.cos(rad)

    # ── direction × speed interaction ─────────────────────────────────────────
    df["dir_speed_80m"] = df["wind_speed_80m"] * df["dir_sin_80m"]
    df["dir_speed_80m_cos"] = df["wind_speed_80m"] * df["dir_cos_80m"]

    # ── wind veer (directional shear between heights) ────────────────────────
    # High veer → atmospheric instability/turbulence → power below speed-only estimate
    def angular_diff_deg(dir_col_a, dir_col_b):
        deg_a = df[dir_col_a] * WIND_DIR_SCALE
        deg_b = df[dir_col_b] * WIND_DIR_SCALE
        d = deg_a - deg_b
        return ((d + 180) % 360) - 180  # normalize to [-180, 180]

    df["dir_veer_10_80"]  = angular_diff_deg("wind_direction_10m", "wind_direction_80m")
    df["dir_veer_80_120"] = angular_diff_deg("wind_direction_80m", "wind_direction_120m")
    df["dir_veer_80_180"] = angular_diff_deg("wind_direction_80m", "wind_direction_180m")
    df["dir_veer_abs_10_80"]  = df["dir_veer_10_80"].abs()
    df["dir_veer_abs_80_180"] = df["dir_veer_80_180"].abs()
    df["dir_veer_total"] = df["dir_veer_abs_10_80"] + df["dir_veer_abs_80_180"]

    # ── cyclic time features ─────────────────────────────────────────────────
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)
    dt_col_name = "METEOFORECASTHOUR_OPENM_Datetime"
    if dt_col_name in df.columns:
        doy = pd.to_datetime(df[dt_col_name]).dt.dayofyear
        df["doy_sin"] = np.sin(2 * np.pi * doy / 365)
        df["doy_cos"] = np.cos(2 * np.pi * doy / 365)

    # ── temperature delta (inversion proxy) ─────────────────────────────────
    df["temp_delta"] = df["temperature_80m"] - df["temperature_120m"]

    # ── gusts ratio ──────────────────────────────────────────────────────────
    df["gust_ratio"] = df["wind_gusts_10m"] / df["wind_speed_10m"].clip(eps)
    df["gust_excess"] = df["wind_gusts_10m"] - df["wind_speed_80m"]

    # ── wind speed regime (physics zones of power curve) ─────────────────────
    df["in_ramp_zone"] = ((df["wind_speed_80m"] >= 3) & (df["wind_speed_80m"] < 12)).astype(float)
    df["in_rated_zone"] = (df["wind_speed_80m"] >= 12).astype(float)
    df["in_steep_zone"] = ((df["wind_speed_80m"] >= 5) & (df["wind_speed_80m"] < 10)).astype(float)

    # ── night × high-shear interaction (LLJ proxy) ───────────────────────────
    df["is_night"] = ((df["hour_of_day"] >= 20) | (df["hour_of_day"] < 6)).astype(float)
    if "shear_80_120" in df.columns:
        df["night_shear"] = df["is_night"] * df["shear_80_120"]
        df["steep_shear"] = df["in_steep_zone"] * df["shear_80_120"]

    # ── ramp zone × turbine availability (curtailment in key power zone) ──────
    # MAE in 6-9 m/s zone is 11.10 MW = 50% of total error; this interaction helps
    df["ramp_n_working"] = df["in_ramp_zone"] * df["n_working"]
    df["night_ramp"] = df["is_night"] * df["in_steep_zone"]
    # Veer × ramp zone: directional instability in the critical power zone
    df["veer_ramp"] = df["dir_veer_abs_10_80"] * df["in_ramp_zone"]

    # ── cross-height speed consistency (turbulence proxy) ────────────────────
    speeds = df[["wind_speed_10m", "wind_speed_80m", "wind_speed_120m", "wind_speed_180m"]]
    df["speed_mean_4h"] = speeds.mean(axis=1)
    df["speed_std_4h"] = speeds.std(axis=1)
    df["speed_cv_4h"] = df["speed_std_4h"] / df["speed_mean_4h"].clip(eps)

    # ── power-law extrapolation using all heights ─────────────────────────────
    log_heights = np.log(np.array([10, 80, 120, 180]))
    log_speeds = np.stack([
        np.log(df["wind_speed_10m"].clip(eps)),
        np.log(df["wind_speed_80m"].clip(eps)),
        np.log(df["wind_speed_120m"].clip(eps)),
        np.log(df["wind_speed_180m"].clip(eps)),
    ], axis=1)
    log_h_centered = log_heights - log_heights.mean()
    alpha_est = (log_speeds @ log_h_centered) / (log_h_centered @ log_h_centered)
    df["alpha_est"] = alpha_est.clip(0, 0.6)
    df["v80_from_regression"] = np.exp(log_speeds.mean(axis=1)) * np.exp(alpha_est * (np.log(80) - log_heights.mean()))

    # ── repair status transition (curtailment signal) ────────────────────────
    # Sudden increase in n_repair → turbines going into maintenance/curtailment
    n_rep = df["Кол-во_ВЭУ_в_ремонте"].values.astype(float)
    n_rep_change = np.zeros(len(n_rep))
    n_rep_change[1:] = n_rep[1:] - n_rep[:-1]
    df["n_repair_change"] = n_rep_change
    df["repair_increased"] = (n_rep_change > 0).astype(float)
    df["repair_decreased"] = (n_rep_change < 0).astype(float)

    # ── turbine count nonlinearity ───────────────────────────────────────────
    df["n_working_sq"] = df["n_working"] ** 2

    # ── pressure tendency (dP/dt) — storm/wind intensification signal ────────
    # Falling pressure → wind increasing; captures ramp-up in 6-9 m/s zone
    p_vals = df["pressure_msl"].values.astype(float)
    dp_dt = np.zeros(len(p_vals))
    dp_dt[1:] = p_vals[1:] - p_vals[:-1]
    df["pressure_change"] = dp_dt

    # ── temperature tendency (dT/dt) — icing onset/end signal ───────────────
    t_vals = df["temperature_80m"].values.astype(float)
    dt_vals = np.zeros(len(t_vals))
    dt_vals[1:] = t_vals[1:] - t_vals[:-1]
    df["temp_change"] = dt_vals

    # ── precipitation lag (icing memory: last hour's snowfall) ───────────────
    snow_vals = df["snowfall"].values.astype(float)
    df["snowfall_lagm1"] = np.concatenate([[snow_vals[0]], snow_vals[:-1]])

    # ── precipitation / clouds ───────────────────────────────────────────────
    df["precip_total"] = df["rain"] + df["showers"] + df["snowfall"]

    # ── wind speed sector bins ────────────────────────────────────────────────
    deg_80m = df["wind_direction_80m"] * WIND_DIR_SCALE
    df["sector_8"] = ((deg_80m / 45).astype(int) % 8).astype(float)

    # ── blade icing risk (Q1 specific: cold + precipitation → power loss) ────
    # T < 2°C + snowfall/showers → ice accretes on blades, reduces lift
    df["icing_risk"] = (
        (df["temperature_80m"] < 2.0) &
        ((df["snowfall"] > 0.01) | (df["showers"] > 0.01))
    ).astype(float)
    # Severity: colder and more precipitation = worse icing
    cold_intensity = (2.0 - df["temperature_80m"].clip(upper=2.0)).clip(lower=0.0)
    precip_intensity = (df["snowfall"] + df["showers"]).clip(upper=2.0)
    df["icing_severity"] = cold_intensity * precip_intensity * df["icing_risk"]

    return df


def add_icing_accumulation(df: pd.DataFrame, dt_col: str = "METEOFORECASTHOUR_OPENM_Datetime") -> pd.DataFrame:
    """
    Add temporal icing accumulation features.
    Ice builds up over consecutive icing hours → increasing power loss.
    Only meaningful when icing_risk is already computed (call after add_features).
    """
    df = df.copy()
    if dt_col not in df.columns or "icing_risk" not in df.columns:
        return df

    dt = pd.to_datetime(df[dt_col])
    df_indexed = df.set_index(dt)
    icing_series = df_indexed["icing_risk"]

    # Rolling count of icing hours (backward-looking, no leakage)
    df["icing_hours_4h"] = icing_series.rolling("4h", min_periods=1).sum().fillna(0).values
    df["icing_hours_8h"] = icing_series.rolling("8h", min_periods=1).sum().fillna(0).values
    df["icing_hours_12h"] = icing_series.rolling("12h", min_periods=1).sum().fillna(0).values
    df["icing_hours_24h"] = icing_series.rolling("24h", min_periods=1).sum().fillna(0).values

    # Intensity-weighted: cold + precipitation accumulation (heavier icing → more power loss)
    severity_series = df_indexed["icing_severity"]
    df["icing_severity_8h"] = severity_series.rolling("8h", min_periods=1).sum().fillna(0).values

    # Consecutive icing streak (unbounded) — captures sustained 38h events in Q1 2026
    icing_arr = icing_series.values.astype(float)
    streak = np.zeros(len(icing_arr))
    current = 0.0
    for i in range(len(icing_arr)):
        if icing_arr[i] > 0:
            current += 1.0
        else:
            current = 0.0
        streak[i] = current
    df["ice_streak"] = streak

    # Hours since last icing (blade thaw time: degradation persists 1-3h after icing stops)
    since_icing = np.zeros(len(icing_arr))
    hours_since = 24.0
    for i in range(len(icing_arr)):
        if icing_arr[i] > 0:
            hours_since = 0.0
        else:
            hours_since = min(hours_since + 1.0, 24.0)
        since_icing[i] = hours_since
    df["hours_since_icing"] = since_icing

    return df


def add_lag_features(df: pd.DataFrame, dt_col: str = "METEOFORECASTHOUR_OPENM_Datetime") -> pd.DataFrame:
    """
    Add temporal lag/lead features for wind speed.
    df must be sorted ascending by dt_col. Gaps are filled with the current value.
    """
    df = df.copy()
    if dt_col not in df.columns:
        return df

    dt = pd.to_datetime(df[dt_col])
    df_indexed = df.set_index(dt)

    lag_cols = {
        "wind_speed_80m": [-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6],
        "wind_speed_120m": [-3, -2, -1, 1, 2, 3],
        "wind_direction_80m": [-1, 1],
        "vp_80m": [-3, -2, -1, 1, 2, 3],
        "vp_120m": [-2, -1, 1, 2],
        "n_working": [-1, 1],
    }
    for col, shifts in lag_cols.items():
        if col not in df.columns:
            continue
        series = df_indexed[col]
        for s in shifts:
            sign = "m" if s < 0 else "p"
            lag_name = f"{col}_lag{sign}{abs(s)}"
            shifted = series.shift(-s, freq="h")
            aligned = shifted.reindex(df_indexed.index)
            df[lag_name] = aligned.fillna(series).values

    # ── derived lag features ──────────────────────────────────────────────────
    if "wind_speed_80m_lagm1" in df.columns and "wind_speed_80m_lagp1" in df.columns:
        df["ws80_trend"] = df["wind_speed_80m_lagp1"] - df["wind_speed_80m_lagm1"]
        df["ws80_accel"] = (df["wind_speed_80m_lagp1"] + df["wind_speed_80m_lagm1"]
                            - 2 * df["wind_speed_80m"])
    if "wind_speed_80m_lagm3" in df.columns and "wind_speed_80m_lagp3" in df.columns:
        df["ws80_ramp_6h"] = df["wind_speed_80m_lagp3"] - df["wind_speed_80m_lagm3"]
    if "vp_80m_lagm2" in df.columns and "vp_80m_lagp2" in df.columns:
        df["vp80_trend_4h"] = df["vp_80m_lagp2"] - df["vp_80m_lagm2"]

    # ── rolling wind statistics (persistence / stability) ────────────────────
    ws80 = df_indexed["wind_speed_80m"]
    df["ws80_mean_3h"] = ws80.rolling("3h", min_periods=1).mean().reindex(df_indexed.index).fillna(ws80).values
    df["ws80_mean_6h"] = ws80.rolling("6h", min_periods=1).mean().reindex(df_indexed.index).fillna(ws80).values
    df["ws80_mean_12h"] = ws80.rolling("12h", min_periods=1).mean().reindex(df_indexed.index).fillna(ws80).values
    df["ws80_max_6h"]  = ws80.rolling("6h", min_periods=1).max().reindex(df_indexed.index).fillna(ws80).values
    df["ws80_std_3h"]  = ws80.rolling("3h", min_periods=1).std().reindex(df_indexed.index).fillna(0).values

    # ── pressure rolling change (3h tendency) — front passage signal ─────────
    if "pressure_change" in df.columns:
        pc_ser = df_indexed["pressure_change"]
        df["pressure_change_3h"] = pc_ser.rolling("3h", min_periods=1).sum().reindex(df_indexed.index).fillna(0).values

    return df


def add_power_curve_prior(df: pd.DataFrame, iso_80m, iso_120m=None) -> pd.DataFrame:
    """
    Add empirical power curve prior as a feature.
    iso_80m: IsotonicRegression fitted on (vp_80m, cf) — hub-height physics.
    iso_120m: IsotonicRegression fitted on (vp_120m, cf) — optional second prior.
    """
    df = df.copy()
    df["pc_prior_cf"] = iso_80m.predict(df["vp_80m"].values)
    if iso_120m is not None:
        df["pc_prior_cf_120m"] = iso_120m.predict(df["vp_120m"].values)
    return df


FEATURE_COLS = [
    # raw wind speeds
    "wind_speed_10m", "wind_speed_80m", "wind_speed_120m", "wind_speed_180m",
    "wind_speed_100m",
    # regression-based hub height estimate
    "v80_from_regression", "alpha_est",
    # wind cubes
    "v3_10m", "v3_80m", "v3_120m", "v3_180m",
    # wind squares
    "v2_80m", "v2_120m",
    # physics power proxy
    "vp_80m", "vp_100m", "vp_120m",
    # shear / profile
    "shear_10_80", "shear_10_180", "shear_80_120", "shear_80_180", "speed_spread",
    "v_hub_estimated",
    # direction cyclic
    "dir_sin_10m", "dir_cos_10m",
    "dir_sin_80m", "dir_cos_80m",
    "dir_sin_120m", "dir_cos_120m",
    "dir_sin_180m", "dir_cos_180m",
    # direction × speed
    "dir_speed_80m", "dir_speed_80m_cos",
    # wind veer (directional shear — atmospheric instability / turbulence proxy)
    "dir_veer_10_80", "dir_veer_80_120", "dir_veer_80_180",
    "dir_veer_abs_10_80", "dir_veer_abs_80_180", "dir_veer_total",
    "veer_ramp",  # veer × ramp zone (critical 6-9 m/s power zone)
    # air density
    "rho", "rho_norm",
    # turbine availability
    "n_working", "availability",
    # time
    "hour_of_day", "hour_sin", "hour_cos",
    "month", "month_sin", "month_cos",
    "doy_sin", "doy_cos",
    # regime flags and interactions
    "in_ramp_zone", "in_rated_zone", "in_steep_zone",
    "is_night", "night_shear", "steep_shear",
    # turbulence proxy
    "speed_mean_4h", "speed_std_4h", "speed_cv_4h",
    # sector
    "sector_8",
    # weather
    "wind_gusts_10m", "gust_ratio", "gust_excess",
    "temperature_80m", "temperature_120m", "temp_delta",
    "pressure_msl",
    "rain", "showers", "snowfall", "precip_total",
    "cloud_cover_low",
    # power curve priors (added dynamically)
    "pc_prior_cf",
    "pc_prior_cf_120m",
    # lag features (short-range: ±1, ±2, ±3h)
    "wind_speed_80m_lagm2", "wind_speed_80m_lagm1",
    "wind_speed_80m_lagp1", "wind_speed_80m_lagp2",
    "wind_speed_120m_lagm3", "wind_speed_120m_lagm2", "wind_speed_120m_lagm1",
    "wind_speed_120m_lagp1", "wind_speed_120m_lagp2", "wind_speed_120m_lagp3",
    "wind_direction_80m_lagm1", "wind_direction_80m_lagp1",
    "vp_80m_lagm3", "vp_80m_lagm2", "vp_80m_lagm1", "vp_80m_lagp1", "vp_80m_lagp2", "vp_80m_lagp3",
    "vp_120m_lagm2", "vp_120m_lagm1", "vp_120m_lagp1", "vp_120m_lagp2",
    "n_working_lagm1", "n_working_lagp1",
    "wind_speed_80m_lagm3", "wind_speed_80m_lagp3",
    # extended look-ahead/back lags (±4, ±5, ±6h) — NWP forecast horizon signal
    "wind_speed_80m_lagm4", "wind_speed_80m_lagm5", "wind_speed_80m_lagm6",
    "wind_speed_80m_lagp4", "wind_speed_80m_lagp5", "wind_speed_80m_lagp6",
    # 12h rolling mean
    "ws80_mean_12h",
    # derived trend features
    "ws80_trend", "ws80_accel",
    "ws80_ramp_6h", "vp80_trend_4h",
    # icing physics (Q1 specific) — accumulation key: ice builds over consecutive hours
    "icing_risk", "icing_severity",
    "icing_hours_4h", "icing_hours_8h", "icing_hours_12h",
    "icing_severity_8h",
    "ice_streak",          # unbounded consecutive icing hours (captures 38h Q1 2026 event)
    "hours_since_icing",   # thaw detection: power degraded 1-3h after icing stops
    "icing_hours_24h",     # 24h rolling icing count — captures long multi-day icing events
    # repair transition features (curtailment signal)
    "n_repair_change",     # Δn_repair vs prev hour — sudden increase = curtailment/emergency
    "repair_increased",    # binary: turbines went into repair last hour
    "repair_decreased",    # binary: turbines came out of repair last hour
    # ramp zone interactions (6-9 m/s = 50% of Q1 2025 MAE)
    "ramp_n_working",      # in_ramp_zone × n_working: curtailment impact in key power zone
    "night_ramp",          # is_night × in_steep_zone: LLJ-prone scenario (underprediction)
    # turbine availability nonlinearity
    "n_working_sq",
    # pressure tendency (storm/wind ramp signal)
    "pressure_change",
    "pressure_change_3h",
    # temperature tendency (icing onset/end)
    "temp_change",
    # precipitation lag (icing memory)
    "snowfall_lagm1",
    # rolling wind statistics (wind persistence / stability in 6-9 m/s zone)
    "ws80_mean_3h",
    "ws80_mean_6h",
    "ws80_max_6h",
    "ws80_std_3h",
]
