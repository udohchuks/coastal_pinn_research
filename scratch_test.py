import xarray as xr
import pandas as pd
import numpy as np

# Create a dummy Dataset
time = pd.date_range("2018-01-01", "2018-01-02", freq="3h")
ds = xr.Dataset({
    "VHM0": (["time", "points"], np.random.rand(len(time), 2)),
    "VMDR": (["time", "points"], np.random.rand(len(time), 2))
}, coords={"time": time})

# Resample
d_rad = np.deg2rad(ds["VMDR"])
sin_d = np.sin(d_rad)
cos_d = np.cos(d_rad)

h_daily = ds["VHM0"].resample(time="1D").mean()
sin_d_daily = sin_d.resample(time="1D").mean()
cos_d_daily = cos_d.resample(time="1D").mean()
d_daily = (np.rad2deg(np.arctan2(sin_d_daily, cos_d_daily)) % 360.0)

print(h_daily.shape)
print(d_daily.shape)
