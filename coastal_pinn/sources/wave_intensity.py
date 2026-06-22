"""Wave source re-export. The implementation is in coastal_pinn.sources.wam.

This module is kept as a thin re-export so that legacy imports
(`from coastal_pinn.sources.wave_intensity import fetch_wave_intensity`)
continue to work, while the actual implementation lives in `wam.py`
(Copernicus Marine WAM model, replacing the prior NOAA WAVEWATCH III).
"""

from coastal_pinn.sources.wam import (  # noqa: F401
    fetch_wave_intensity,
    _to_dataframe,
    _download_wam,
)

__all__ = ["fetch_wave_intensity", "_to_dataframe", "_download_wam"]
