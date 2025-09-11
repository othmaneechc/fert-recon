# build_big_dataframe_with_labels.py
# ----------------------------------
#  • Reads yearly_2002 … yearly_2015 stacks (244 bands each)
#  • Column names come from the GeoTIFF band descriptions
#  • pixel_id = row * 192 + col   (0-based)
#  • Result → Parquet table: 192×198×14 = 532 224 rows  ×  (246 columns)

import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from tqdm.auto import tqdm     # nice progress bar

# ----------------------------------------------------
STACK_DIR   = Path("output/processed/stack_yearly")
YEARS       = range(2002, 2016)          # 14 years
WIDTH, HEIGHT = 192, 198
N_PIXELS      = WIDTH * HEIGHT           # 38 016
# ----------------------------------------------------

def band_labels(fp: Path) -> list[str]:
    """Return a list of 244 band descriptions from the GeoTIFF."""
    with rasterio.open(fp) as src:
        labels = [src.descriptions[i] or f"band_{i+1:03d}"
                  for i in range(src.count)]
    # sanitise (spaces → _, lowercase, keep unique)
    clean = []
    seen  = set()
    for lbl in labels:
        lbl = lbl.lower().replace(" ", "_")
        if lbl in seen:                  # make unique if duplicated
            base = lbl; k = 1
            while f"{base}_{k}" in seen: k += 1
            lbl = f"{base}_{k}"
        clean.append(lbl); seen.add(lbl)
    return clean

# get the canonical label list from the first stack (2002)
LABELS = band_labels(STACK_DIR / "yearly_2002.tif")
assert len(LABELS) == 244, "Expected 244 bands!"

dfs = []

for yr in tqdm(YEARS, desc="Building DataFrame"):
    fp = STACK_DIR / f"yearly_{yr}.tif"
    with rasterio.open(fp) as src:
        arr = src.read(out_shape=(src.count, HEIGHT, WIDTH))        # (244,198,192)
        arr2d = arr.reshape(src.count, N_PIXELS).T                 # (38k, 244)

    df = pd.DataFrame(arr2d, columns=LABELS, copy=False)
    df.insert(0, "pixel_id", np.arange(N_PIXELS, dtype=np.int32))
    df.insert(1, "year", yr)
    dfs.append(df)

big_df = pd.concat(dfs, ignore_index=True)
print("Final shape:", big_df.shape)        # (532224, 246)

# ---- save ----
out_file = STACK_DIR / "pixels_2002_2015.csv"
big_df.to_csv(STACK_DIR / "pixels_2002_2015.csv", index=False)
print("Saved to", out_file)
