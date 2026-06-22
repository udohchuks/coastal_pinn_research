"""Sediment recovery R source.

v2: approximates R using the sediment-budget equation:

    dS/dt = -dQ/dx / h_active + R

where Q is the CERC longshore transport rate (inferred from wave data)
and dS/dt is the observed shoreline change rate per transect.

This is a physics-based approximation, not a direct measurement. The PINN
still learns a refined R_θ closure; this estimate provides a useful
initial value and comparison baseline.

Algorithm per transect i:
  1. Fit dS/dt_i via linear regression of observed S(t) over all dates
  2. Compute mean W_longshore_i and mean W_i across all timestamps
  3. Compute CERC transport proxy:
       Q_i = W_i^(3/2) * mean(W_longshore_i)
     (simplification of Q ∝ H^(5/2) * sin(2θ))
  4. Compute transport divergence via centred finite difference:
       dQ/dx_i = (Q_{i+1} - Q_{i-1}) / (2 * delta_x)
  5. R_i = max(0, dS/dt_i + dQ/dx_i / h_active)
     where h_active ≈ 10 m (closure depth for the Gulf of Guinea shelf)

Returns a DataFrame with columns (transect_id, R_sediment_m_yr).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from coastal_pinn.config import PipelineConfig


def compute_sediment_recovery(
    cfg: PipelineConfig,
    merged: pd.DataFrame,
    *,
    h_active: float = 10.0,
) -> pd.DataFrame:
    """Compute per-transect sediment recovery R (m/yr).

    Parameters
    ----------
    cfg : PipelineConfig
        Pipeline config (used for transect spacing).
    merged : pd.DataFrame
        The wide-format reconciled DataFrame MUST contain:
        - transect_id, timestamp, cross_shore_S_m, W_longshore, W_m
    h_active : float
        Active beach profile height / closure depth in meters.
        Default 10 m is reasonable for the Gulf of Guinea shelf.

    Returns
    -------
    pd.DataFrame with columns [transect_id, R_sediment_m_yr].
    """
    if merged.empty:
        return pd.DataFrame(columns=["transect_id", "R_sediment_m_yr"])

    required = {"transect_id", "timestamp", "cross_shore_S_m", "W_longshore", "W_m"}
    missing = required - set(merged.columns)
    if missing:
        raise ValueError(
            f"merged DataFrame missing columns for R computation: {sorted(missing)}"
        )

    delta_x = cfg.region.transect_spacing_m  # 100 m for Keta

    transect_ids = np.sort(merged["transect_id"].unique())
    n_transects = len(transect_ids)

    r_values = np.full(n_transects, np.nan)

    for idx, tid in enumerate(transect_ids):
        rows = merged[merged["transect_id"] == tid]
        if len(rows) < 2:
            continue

        # 1. Fit dS/dt via linear regression of cross_shore_S_m vs timestamp
        t_num = (rows["timestamp"].astype("int64") / 1e9 / 86400 / 365.25).values
        t_mean = t_num.mean()
        S = rows["cross_shore_S_m"].values.astype(float)
        S_mean = S.mean()

        # Simple slope: cov(t, S) / var(t)
        t_demean = t_num - t_mean
        var_t = (t_demean ** 2).sum()
        if var_t < 1e-12:
            continue
        ds_dt = (t_demean * (S - S_mean)).sum() / var_t  # m/yr

        # 2. Mean forcing at this transect
        W_longshore_avg = rows["W_longshore"].mean(skipna=True)
        W_avg = rows["W_m"].mean(skipna=True)

        # 3. CERC transport proxy (dimensionless relative measure)
        # Q ∝ H^(5/2) * sin(2θ) = H^(3/2) * H * sin(2θ) = H^(3/2) * W_longshore
        if W_avg <= 0 or np.isnan(W_avg):
            Q_i = 0.0
        else:
            Q_i = (W_avg ** 1.5) * W_longshore_avg

        # 4. Transport divergence via centred finite difference
        if 0 < idx < n_transects - 1:
            prev_tid = transect_ids[idx - 1]
            next_tid = transect_ids[idx + 1]

            rows_prev = merged[merged["transect_id"] == prev_tid]
            rows_next = merged[merged["transect_id"] == next_tid]

            W_avg_prev = rows_prev["W_m"].mean(skipna=True)
            WL_avg_prev = rows_prev["W_longshore"].mean(skipna=True)
            W_avg_next = rows_next["W_m"].mean(skipna=True)
            WL_avg_next = rows_next["W_longshore"].mean(skipna=True)

            Q_prev = (W_avg_prev ** 1.5) * WL_avg_prev if W_avg_prev > 0 else 0.0
            Q_next = (W_avg_next ** 1.5) * WL_avg_next if W_avg_next > 0 else 0.0

            dQ_dx = (Q_next - Q_prev) / (2.0 * delta_x)
        elif idx == 0:
            # Forward difference at left boundary
            next_tid = transect_ids[idx + 1]
            rows_next = merged[merged["transect_id"] == next_tid]
            W_avg_next = rows_next["W_m"].mean(skipna=True)
            WL_avg_next = rows_next["W_longshore"].mean(skipna=True)
            Q_next = (W_avg_next ** 1.5) * WL_avg_next if W_avg_next > 0 else 0.0
            dQ_dx = (Q_next - Q_i) / delta_x
        else:
            # Backward difference at right boundary
            prev_tid = transect_ids[idx - 1]
            rows_prev = merged[merged["transect_id"] == prev_tid]
            W_avg_prev = rows_prev["W_m"].mean(skipna=True)
            WL_avg_prev = rows_prev["W_longshore"].mean(skipna=True)
            Q_prev = (W_avg_prev ** 1.5) * WL_avg_prev if W_avg_prev > 0 else 0.0
            dQ_dx = (Q_i - Q_prev) / delta_x

        # 5. Sediment budget: R = dS/dt + dQ/dx / h_active
        # dQ/dx has arbitrary units (m^(5/2) / m); dividing by h_active
        # gives it dimensions of m/yr after scaling
        r_candidate = ds_dt + dQ_dx / h_active
        r_values[idx] = max(0.0, r_candidate)

    return pd.DataFrame({
        "transect_id": transect_ids,
        "R_sediment_m_yr": r_values,
    })


# ---------------------------------------------------------------------------
# Legacy stub for backward compatibility (Series-based API)
# ---------------------------------------------------------------------------

def _compute_r_series(cfg: PipelineConfig, dates: pd.DatetimeIndex) -> pd.Series:
    """DEPRECATED: Series-based API. Use compute_sediment_recovery instead."""
    raise NotImplementedError(
        "sediment_recovery v1 is not implemented. "
        "Use compute_sediment_recovery(cfg, merged_df) for v2 API."
    )
