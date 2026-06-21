"""Sediment recovery R source.

The paper's Eq. 2 is dS/dt = αW − βR, where R is sediment recovery
(m/yr). The PINN learns α and β as parameters; R itself is a model
concern and not a dataset.

This module is a deliberate placeholder. It raises NotImplementedError
when called, so a missing real-data source for R is loud rather than
silent. To enable a real R source:

    1. Implement the function below (e.g. via river-discharge x
       suspended-sediment-concentration rating curves for the Volta
       River, the dominant sediment source for the Keta segment).
    2. Wire it into coastal_pinn.pipeline.reconcile() so the resulting
       column is added to the wide table.
    3. Add a CLI subcommand and config flag.

The wide table reserves the column 'R_sediment_m_yr' so the schema is
stable across versions. Until this module is implemented, that column
is left as NaN in the output.
"""

from __future__ import annotations

import pandas as pd

from coastal_pinn.config import PipelineConfig


def compute_sediment_recovery(
    cfg: PipelineConfig,
    dates: pd.DatetimeIndex,
) -> pd.Series:
    """Compute sediment recovery R (m/yr) for each date in `dates`.

    v1: raises NotImplementedError. R is a model concern (the PINN
    learns β implicitly); no real sediment source is wired into this
    pipeline yet.

    Future implementation should return a pd.Series of length len(dates),
    indexed by dates (UTC tz-aware), with R in m/yr.
    """
    raise NotImplementedError(
        "sediment_recovery is not implemented in v1. "
        "R is a model concern (PINN learns β). "
        "To enable: implement this function with a real sediment source "
        "(e.g. Volta River discharge x SSC rating curve), then wire into "
        "coastal_pinn.pipeline.reconcile()."
    )