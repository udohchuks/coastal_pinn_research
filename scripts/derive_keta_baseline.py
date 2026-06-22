"""Derive the Keta onshore baseline from the OpenStreetMap Atlantic coastline.

The Keta study geometry must follow the *ocean* shoreline (the NE-trending
Atlantic coast), not the inland Keta Lagoon. OSM's ``natural=coastline`` tag
marks only the sea/land boundary (lagoons are tagged separately), so it is the
authoritative source for the ocean coast.

This script:
  1. Fetches ``natural=coastline`` ways for the Keta bbox via the Overpass API
     (trying several mirrors).
  2. Smooths them into an ordered W->E coast polyline (median northing per
     longitude bin).
  3. Offsets each point ~150 m onshore (toward land, i.e. north) to form the
     DSAS-style baseline.
  4. Writes data/keta_baseline.json (consumed by config.py REGIONS and
     config/keta.yaml).

Run:  python scripts/derive_keta_baseline.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import requests

from coastal_pinn.core.coords import lonlat_to_utm, utm_to_lonlat

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "keta_baseline.json"
RAW = ROOT / "data" / "keta_coastline_osm.json"

# Overpass bbox (south, west, north, east) and mirrors.
BBOX = (5.78, 0.88, 6.14, 1.22)
MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]
N_BINS = 60          # baseline guide points
ONSHORE_OFFSET_M = 150.0


def fetch_coastline() -> list[list[float]]:
    s, w, n, e = BBOX
    q = (f"[out:json][timeout:120];"
         f'way["natural"="coastline"]({s},{w},{n},{e});out geom;')
    for mirror in MIRRORS:
        for _ in range(3):
            try:
                r = requests.post(mirror, data={"data": q},
                                  headers={"User-Agent": "coastal-pinn/1.0"},
                                  timeout=120)
                if r.status_code == 200 and r.text.strip().startswith("{"):
                    j = r.json()
                    pts = [[p["lon"], p["lat"]]
                           for el in j["elements"]
                           for p in el.get("geometry", [])]
                    if pts:
                        print(f"fetched {len(pts)} coastline nodes via {mirror}")
                        return pts
            except Exception:
                pass
            time.sleep(2)
    raise RuntimeError("all Overpass mirrors failed")


def build_baseline(coast_lonlat: list[list[float]]) -> list[list[float]]:
    lon = np.array([p[0] for p in coast_lonlat])
    lat = np.array([p[1] for p in coast_lonlat])
    x, y = lonlat_to_utm(lon, lat, "31N")
    order = np.argsort(x)
    x, y = x[order], y[order]

    # Smooth: median northing per longitude bin -> ordered coast polyline.
    xb = np.linspace(x.min(), x.max(), N_BINS + 1)
    cx, cy = [], []
    for i in range(N_BINS):
        m = (x >= xb[i]) & (x < xb[i + 1])
        if m.sum() > 3:
            cx.append(x[m].mean())
            cy.append(np.median(y[m]))
    coast = np.column_stack([cx, cy])

    # Offset each point ~150 m onshore (north = toward land here).
    base = np.zeros_like(coast)
    for i in range(len(coast)):
        j0, j1 = max(0, i - 1), min(len(coast) - 1, i + 1)
        t = coast[j1] - coast[j0]
        t = t / np.linalg.norm(t)
        perp = np.array([-t[1], t[0]])
        if perp[1] < 0:          # choose the northward (inland) perpendicular
            perp = -perp
        base[i] = coast[i] + ONSHORE_OFFSET_M * perp

    blon, blat = utm_to_lonlat(base[:, 0], base[:, 1], "31N")
    return [[round(float(a), 4), round(float(b), 4)] for a, b in zip(blon, blat)]


if __name__ == "__main__":
    coast = fetch_coastline()
    RAW.parent.mkdir(parents=True, exist_ok=True)
    RAW.write_text(json.dumps(coast), encoding="utf-8")
    baseline = build_baseline(coast)
    OUT.write_text(json.dumps(baseline, indent=0), encoding="utf-8")
    print(f"wrote {len(baseline)} baseline points -> {OUT}")
    print("NOTE: also regenerate config/keta.yaml's baseline block to match.")
