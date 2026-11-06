"""Phase 0 / P2 probe: measure the real cost of reading one ARCO-ERA5 3D chunk.

ARCO-ERA5 3D arrays are chunked (time=1, level=37, lat=721, lon=1440) i.e. ~154 MB
(float32) per chunk. This probe measures: time-coordinate read, and one
single-timestep 3-level regional read of `temperature` (one chunk), so we can
decide whether a 7-variable slice is feasible on this link.

Run: /opt/miniconda3/envs/atmosgen/bin/python -u research/env/probe_arco_read.py
"""
import datetime
import fsspec  # noqa: F401
import gcsfs
import numpy as np
import time
import xarray as xr

STORE = "gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
TARGET = np.datetime64("2020-01-01T00")
BASE = np.datetime64("1900-01-01T00")


def log(*a):
    print(*a, flush=True)


fs = gcsfs.GCSFileSystem(token="anon")
mapper = fs.get_mapper(STORE, check=False)

t0 = time.time()
ds = xr.open_zarr(mapper, consolidated=True, chunks=None)
log(f"[OPEN] xr.open_zarr in {time.time()-t0:.1f}s dims={dict(ds.sizes)} nvars={len(ds.data_vars)}")

t0 = time.time()
tv = ds["time"].values
log(f"[TIME] coordinate read {len(tv)} values in {time.time()-t0:.1f}s; first={tv[0]} last={tv[-1]}")

hours = tv.astype("datetime64[h]").astype("int64") - BASE.astype("datetime64[h]").astype("int64")
want = int((TARGET - BASE) / np.timedelta64(1, "h"))
idx = int(np.searchsorted(hours, want))
log(f"[INDEX] target={TARGET} -> idx={idx} actual={tv[idx]}")

chunk_obj = f"{STORE}/temperature/{idx}.0.0"
t0 = time.time()
info = fs.info(chunk_obj)
log(f"[CHUNK] {chunk_obj} size={info['size']} bytes ({info['size']/1e6:.1f} MB compressed) in {time.time()-t0:.1f}s")

# One chunk read through xarray (all 37 levels pulled, then subset)
t0 = time.time()
sub = ds["temperature"].isel(time=idx).sel(level=[500, 700, 850]).sel(
    latitude=slice(45, 20), longitude=slice(100, 125)).load()
load_s = time.time() - t0
log(f"[LOAD] temperature 1 chunk read+decode+subset in {load_s:.1f}s -> {dict(sub.sizes)}")
log(f"[LOAD] local subset bytes={sub.nbytes} B ({sub.nbytes/1e6:.2f} MB)")
log(f"[LOAD] effective GCS throughput ~{info['size']/load_s/1e6:.2f} MB/s (compressed chunk / load time)")
log(f"[VALUE] t min={float(sub.min()):.4f} max={float(sub.max()):.4f} mean={float(sub.mean()):.4f}")
log("[PROBE_DONE]")
