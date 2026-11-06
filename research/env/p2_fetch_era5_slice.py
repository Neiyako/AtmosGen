"""Phase 0 / P2: land a REAL ERA5 slice reachable on this link (degraded source, see log)."""
import os, time, importlib
import numpy as np
import xarray as xr

OUT_NC = "data/raw/era5_slice.nc"
OUT_MD = "data/raw/era5_slice_SUMMARY.md"
SOURCE = "gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr"
TARGET_MAP = {
    "t": ["temperature", "t"], "q": ["specific_humidity", "q"],
    "u": ["u_component_of_wind", "u"], "v": ["v_component_of_wind", "v"],
    "clwc": ["specific_cloud_liquid_water_content", "clwc"],
    "ciwc": ["specific_cloud_ice_water_content", "ciwc"],
    "crwc": ["specific_rain_water_content", "crwc"],
    "cswc": ["specific_snow_water_content", "cswc"],
    "tcwv": ["total_column_water_vapour", "tcwv"],
}
TIME_SLICE = ("2020-01-01T00", "2020-01-02T18")
LEVELS = [500, 700, 850]
LAT_RANGE = (20, 45)
LON_RANGE = (100, 125)

def log(m=""):
    print(m, flush=True)

def versions():
    log("=== package versions ===")
    for p in ["numpy","pandas","xarray","zarr","dask","gcsfs","netCDF4","fsspec"]:
        try:
            m = importlib.import_module(p); log(f"{p}=={getattr(m,'__version__','?')}")
        except Exception as e:
            log(f"{p}: IMPORT_FAILED {type(e).__name__}: {e}")

def open_ds(url):
    import gcsfs
    fs = gcsfs.GCSFileSystem(token="anon")
    mapper = fs.get_mapper(url.replace("gs://", "", 1), check=False)
    errs = []
    for label, kw in [("consolidated=True", dict(consolidated=True, chunks=None)),
                      ("consolidated=False", dict(consolidated=False, chunks=None))]:
        try:
            t0 = time.time(); ds = xr.open_zarr(mapper, **kw)
            log(f"[XARRAY] open_zarr({label}) in {time.time()-t0:.1f}s dims={dict(ds.sizes)} nvars={len(ds.data_vars)}")
            return ds
        except Exception as e:
            errs.append(f"{label}: {type(e).__name__}: {e}"); log(f"[XARRAY] open_zarr({label}) FAILED: {type(e).__name__}: {e}")
    raise RuntimeError("open_zarr failed: " + " | ".join(errs))

def main():
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    os.makedirs("data/raw", exist_ok=True)
    versions(); log(f"\n=== SOURCE {SOURCE}")
    t_all = time.time(); ds = open_ds(SOURCE)
    log(f"[VARS] {list(ds.data_vars)}")
    resolved, present, missing = {}, [], []
    for short, cands in TARGET_MAP.items():
        hit = next((c for c in cands if c in ds.data_vars), None)
        if hit: resolved[short] = hit; present.append(short)
        else: missing.append(short)
    log(f"[COVERAGE] present_short={present}")
    log(f"[COVERAGE] missing_short={missing}")
    log(f"[COVERAGE] mapping={resolved}")
    lat_vals = ds["latitude"].values; desc = bool(lat_vals[0] > lat_vals[-1])
    lat_sel = slice(LAT_RANGE[1], LAT_RANGE[0]) if desc else slice(LAT_RANGE[0], LAT_RANGE[1])
    lon_sel = slice(LON_RANGE[0], LON_RANGE[1])
    log(f"[COORD] latitude descending={desc} {lat_vals.min()}..{lat_vals.max()} n={len(lat_vals)}")
    log(f"[COORD] longitude {float(ds['longitude'].values.min())}..{float(ds['longitude'].values.max())} n={len(ds['longitude'])}")
    log(f"[COORD] levels={list(ds.level.values)}")
    log(f"[COORD] time {ds.time.values[0]}..{ds.time.values[-1]} n={ds.sizes['time']}")
    names = list(resolved.values())
    v3d = [v for v in names if "level" in ds[v].dims]; v2d = [v for v in names if "level" not in ds[v].dims]
    levels = [lv for lv in LEVELS if lv in ds.level.values]
    log(f"[SELECT] 3D={v3d} levels={levels} 2D={v2d}")
    pieces = []
    for v in v3d:
        t0 = time.time()
        arr = ds[v].sel(level=levels).sel(time=slice(*TIME_SLICE), latitude=lat_sel, longitude=lon_sel).load()
        log(f"[LOAD] {v}: {dict(arr.sizes)} in {time.time()-t0:.1f}s"); pieces.append(arr)
    for v in v2d:
        t0 = time.time()
        arr = ds[v].sel(time=slice(*TIME_SLICE), latitude=lat_sel, longitude=lon_sel).load()
        log(f"[LOAD] {v}: {dict(arr.sizes)} in {time.time()-t0:.1f}s"); pieces.append(arr)
    out = xr.merge(pieces)
    log(f"[SLICE] {dict(out.sizes)} total_read_wall={time.time()-t_all:.1f}s")
    checks = []
    for v in out.data_vars:
        a = out[v].values; n_nan = int(np.isnan(a).sum()); sz = int(out[v].size)
        checks.append((v, float(np.nanmin(a)), float(np.nanmax(a)), n_nan, sz, n_nan/sz))
        log(f"[RANGE] {v}: min={checks[-1][1]:.6g} max={checks[-1][2]:.6g} nan={n_nan}/{sz} ({checks[-1][5]*100:.2f}%)")
    t_write = time.time(); out.to_netcdf(OUT_NC, engine="netcdf4"); write_s = time.time()-t_write
    size_b = os.path.getsize(OUT_NC)
    log(f"[WRITE] {OUT_NC} size={size_b} B ({size_b/1e6:.3f} MB) in {write_s:.1f}s")
    lines = ["# ERA5 slice summary (Phase 0 / P2)", "",
        f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')} Asia/Shanghai (access date 2026-11-06)",
        f"- Source (actual): `{SOURCE}` (anonymous GCS, gcsfs token='anon')",
        "- **Degradation note**: 0.25deg full-37 pressure stores (WB2 and ARCO) use chunk "
        "(time=1, level=37, lat=721, lon=1440) = 85.2 MB compressed per chunk; measured link throughput "
        "38.3 KB/s => ~37 min/chunk, so the requested 0.25deg 7-var slice was NOT pullable tonight. "
        "This slice uses the WB2 coarse regridded real-ERA5 store to prove the Zarr->xarray->NetCDF path. "
        "Cloud-water vars (clwc/ciwc/crwc/cswc) are absent from all WB2 stores.",
        f"- Requested short -> store long name mapping: `{resolved}`",
        f"- Present: `{present}`; **absent from source**: `{missing}`",
        "- ARCO 0.25deg streaming recipe (with clwc/ciwc) for Phase 1: `research/env/logs/p2_arco_infeasible_evidence.md`",
        f"- Total read wall time: {time.time()-t_all:.1f}s; NetCDF write: {write_s:.1f}s", "",
        "## dims", "", "```", str(dict(out.sizes)), "```", "", "## coords", "", "```",
        str({c: (out[c].dims, int(out[c].size), float(out[c].min()), float(out[c].max())) for c in out.coords}),
        "```", "", "## variables (units / long_name)", "", "| var | dims | units | long_name |", "|---|---|---|---|"]
    for v in out.data_vars:
        a = out[v].attrs
        lines.append(f"| {v} | {out[v].dims} | {a.get('units','?')} | {a.get('long_name','?')} |")
    lines += ["", "## time range", "", f"- start: `{out['time'].values[0]}`", f"- end: `{out['time'].values[-1]}`",
        f"- steps: {int(out.sizes.get('time',0))} (step {out['time'].values[1]-out['time'].values[0]})"]
    if "latitude" in out.coords:
        lines += [f"- latitude: {float(out.latitude.min())} .. {float(out.latitude.max())} (descending={bool(out.latitude[0]>out.latitude[-1])}, n={int(out.sizes['latitude'])})"]
    if "longitude" in out.coords:
        lines += [f"- longitude: {float(out.longitude.min())} .. {float(out.longitude.max())} (n={int(out.sizes['longitude'])})"]
    if "level" in out.coords:
        lines += [f"- level (hPa): {[int(x) for x in out.level.values]}"]
    lines += ["", "## value ranges / missing values", "", "| var | min | max | nan count | size | nan ratio |", "|---|---|---|---|---|---|"]
    for v, mn, mx, nn, sz, ratio in checks:
        lines.append(f"| {v} | {mn:.6g} | {mx:.6g} | {nn} | {sz} | {ratio*100:.2f}% |")
    lines += ["", "## file", "", f"- path: `{OUT_NC}`", f"- size: {size_b} bytes ({size_b/1e6:.3f} MB)",
              f"- <50MB: {'YES' if size_b < 50e6 else 'NO'}", ""]
    open(OUT_MD, "w").write("\n".join(lines))
    log(f"[SUMMARY] wrote {OUT_MD}"); log("[DONE] ok")

if __name__ == "__main__":
    main()
