"""Phase 0 / P2b: inspect the local ERA5 0.25deg files, extract a small slice, copy cloud sample.

Read-only on /Volumes/Untitled and on the DBAS ER data; outputs go to data/raw/.
Run: /opt/miniconda3/envs/atmosgen/bin/python -u research/env/p2b_local_era5.py > research/env/logs/p2b_local_era5.log 2>&1
"""
import os, shutil, time
import numpy as np
import xarray as xr

SRC = "/Volumes/Untitled/2010"
DATE = "20100501"
FILES = {short: f"{SRC}/ERA5_025dg_{DATE}_{long}.nc" for short, long in
         {"t": "temperature", "q": "specific_humidity",
          "u": "u_component_of_wind", "v": "v_component_of_wind"}.items()}
CLOUD = "/Users/apple/Desktop/era5数据dbas/data/downloads/reanalysis-era5-pressure-levels/2024/20240101_multi_fdb02e.nc"
OUT_NC = "data/raw/era5_local_slice_20100501.nc"
OUT_MD = "data/raw/era5_local_slice_SUMMARY.md"
CLOUD_NC = "data/raw/era5_cloud_sample_20240101.nc"
CLOUD_MD = "data/raw/era5_cloud_sample_SUMMARY.md"

def log(m=""):
    print(m, flush=True)

def find(ds, cands):
    for c in cands:
        if c in ds.coords or c in ds.dims:
            return c
    return None

def stats(da):
    a = da.values
    n_nan = int(np.isnan(a).sum())
    return dict(min=float(np.nanmin(a)), max=float(np.nanmax(a)),
                mean=float(np.nanmean(a)), nan=n_nan, size=int(a.size), nan_ratio=n_nan / a.size)

log("=== A. inspect local ERA5 files (read-only) ===")
t0 = time.time()
dsets = {}
for short, path in FILES.items():
    size = os.path.getsize(path)
    ds = xr.open_dataset(path)
    dsets[short] = ds
    log(f"\n--- {short}: {path}\n    file_size={size} B ({size/1e6:.1f} MB) open={time.time()-t0:.1f}s")
    log(f"    dims={dict(ds.sizes)}")
    log(f"    coords={list(ds.coords)}; data_vars={list(ds.data_vars)}")
    for v in ds.data_vars:
        a = ds[v].attrs
        log(f"    VAR {v}: dims={ds[v].dims} units={a.get('units','?')} long_name={a.get('long_name','?')}")
    tname = find(ds, ["time", "valid_time"])
    log(f"    time_coord={tname} values={ds[tname].values[:3]}..{ds[tname].values[-1]} n={ds.sizes[tname]}")
    lname = find(ds, ["level", "pressure_level"])
    log(f"    level_coord={lname} n={ds.sizes[lname]} values_head={list(ds[lname].values[:5])} last={ds[lname].values[-1] if lname else None}")
    for c in ["latitude", "longitude", "lat", "lon"]:
        if c in ds.dims:
            vv = ds[c].values
            log(f"    {c}: n={len(vv)} min={vv.min()} max={vv.max()} desc={bool(vv[0]>vv[-1])}")

# time-axis consistency across the four files
tname = find(dsets["t"], ["time", "valid_time"])
ref = dsets["t"][tname].values
same = all(np.array_equal(dsets[s][tname].values, ref) for s in dsets)
log(f"\n[TIME_CONSISTENCY] four files share identical time axis: {same}; n={len(ref)}; "
    f"range={ref[0]}..{ref[-1]}")

log("\n=== B. extract slice 2010-05-01T00..18, level [500,700,850], lat 20-45, lon 100-125 ===")
pieces, chunk_meta = [], []
for short, ds in dsets.items():
    tname = find(ds, ["time", "valid_time"])
    lname = find(ds, ["level", "pressure_level"])
    latname = find(ds, ["latitude", "lat"])
    lonname = find(ds, ["longitude", "lon"])
    var = [v for v in ds.data_vars][0]
    latv = ds[latname].values
    lat_sel = slice(45, 20) if latv[0] > latv[-1] else slice(20, 45)
    lonv = ds[lonname].values
    lon_sel = slice(100, 125) if lonv.min() >= 0 else slice(100, 125)
    t = time.time()
    da = ds[var].sel({tname: ["2010-05-01T00", "2010-05-01T06", "2010-05-01T12", "2010-05-01T18"],
                      lname: [500, 700, 850],
                      latname: lat_sel, lonname: lon_sel}).load()
    st = stats(da)
    log(f"[SLICE] {short}<-{var}: dims={dict(da.sizes)} in {time.time()-t:.1f}s "
        f"min={st['min']:.6g} max={st['max']:.6g} mean={st['mean']:.6g} nan={st['nan']}/{st['size']} ({st['nan_ratio']*100:.2f}%)")
    da = da.rename(var)  # normalize to short name t/q/u/v
    da.attrs["units"] = ds[var].attrs.get("units", da.attrs.get("units", ""))
    da.attrs["source_file"] = FILES[short]
    pieces.append(da)
    chunk_meta.append((short, var, st))

out = xr.merge(pieces)
log(f"[MERGE] dims={dict(out.sizes)}")
t_write = time.time()
out.to_netcdf(OUT_NC, engine="netcdf4")
write_s = time.time() - t_write
size_b = os.path.getsize(OUT_NC)
log(f"[WRITE] {OUT_NC} size={size_b} B ({size_b/1e6:.3f} MB) in {write_s:.1f}s")
total_s = time.time() - t0

# verify round-trip
chk = xr.open_dataset(OUT_NC)
log(f"[VERIFY] reopen dims={dict(chk.sizes)} vars={list(chk.data_vars)}")

lines = ["# Local ERA5 slice summary (Phase 0 / P2b, 2010-05-01)", "",
         f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')} Asia/Shanghai",
         "- **Read-only declaration**: source NetCDF files were opened read-only via xarray; "
         "nothing on /Volumes/Untitled or in the DBAS dir was modified, moved or re-written.",
         "- Source (local, external volume):"]
for short, long in {"t": "temperature", "q": "specific_humidity",
                    "u": "u_component_of_wind", "v": "v_component_of_wind"}.items():
    p = FILES[short]
    lines.append(f"  - `{short}` -> `{p}` ({os.path.getsize(p)} B / {os.path.getsize(p)/1e6:.1f} MB)")
lines += ["- Variable mapping: `t=temperature, q=specific_humidity, u=u_component_of_wind, "
          "v=v_component_of_wind` (data vars renamed to short names in the output file)",
          f"- Native resolution: 0.25 deg global, 37 pressure levels, hourly (see source attrs below)",
          f"- Extraction: time 2010-05-01 00/06/12/18 (4 steps), level [500,700,850] hPa, "
          f"lat 20-45N, lon 100-125E",
          f"- Time-axis consistency of the four source files: **{same}** (identical time arrays)",
          f"- Total read+write wall time: {total_s:.1f}s; NetCDF write: {write_s:.1f}s",
          "", "## dims", "", "```", str(dict(out.sizes)), "```", "",
          "## coords", "", "```",
          str({c: (out[c].dims, int(out[c].size)) for c in out.coords}), "```", "",
          "## variables (units / long_name)", "", "| var | dims | units | long_name |", "|---|---|---|---|"]
for v in out.data_vars:
    a = out[v].attrs
    lines.append(f"| {v} | {out[v].dims} | {a.get('units','?')} | {a.get('long_name','?')} |")
tvals = out["time"].values
lines += ["", "## time range", "", f"- start: `{tvals[0]}`", f"- end: `{tvals[-1]}`",
          f"- steps: {int(out.sizes['time'])} (step {tvals[1]-tvals[0]})",
          f"- level (hPa): {[int(x) for x in out['level'].values] if 'level' in out.coords else 'n/a'}"]
for c in ["latitude", "longitude"]:
    if c in out.coords:
        vv = out[c].values
        lines.append(f"- {c}: {vv.min()} .. {vv.max()} (n={len(vv)}, descending={bool(vv[0]>vv[-1])})")
lines += ["", "## value ranges / missing values (after subset)", "",
          "| var | min | max | mean | nan count | size | nan ratio |", "|---|---|---|---|---|---|---|"]
for short, var, st in chunk_meta:
    lines.append(f"| {short} ({var}) | {st['min']:.6g} | {st['max']:.6g} | {st['mean']:.6g} | "
                 f"{st['nan']} | {st['size']} | {st['nan_ratio']*100:.2f}% |")
lines += ["", "## file", "", f"- path: `{OUT_NC}`", f"- size: {size_b} bytes ({size_b/1e6:.3f} MB)",
          f"- <10MB: {'YES' if size_b < 10e6 else 'NO'}", ""]
open(OUT_MD, "w").write("\n".join(lines))
log(f"[SUMMARY] wrote {OUT_MD}")

log("\n=== C. cloud-water sample (clwc/ciwc) ===")
shutil.copyfile(CLOUD, CLOUD_NC)
log(f"[COPY] {CLOUD} -> {CLOUD_NC} ({os.path.getsize(CLOUD_NC)} B)")
cds = xr.open_dataset(CLOUD_NC)
log(f"[CLOUD] dims={dict(cds.sizes)} coords={list(cds.coords)} data_vars={list(cds.data_vars)}")
clines = ["# ERA5 cloud-water sample summary (Phase 0 / P2b)", "",
          f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')} Asia/Shanghai",
          "- **Read-only declaration**: the original DBAS download was NOT modified; a byte-copy "
          "was placed in data/raw/ for offline reference.",
          f"- Source: `{CLOUD}` ({os.path.getsize(CLOUD)} B)",
          f"- Copy: `{CLOUD_NC}` ({os.path.getsize(CLOUD_NC)} B)", "",
          "## dims", "", "```", str(dict(cds.sizes)), "```", "",
          "## coords", "", "```", str({c: (cds[c].dims, int(cds[c].size)) for c in cds.coords}), "```", "",
          "## variables", "", "| var | dims | units | long_name | min | max | nan ratio |", "|---|---|---|---|---|---|---|"]
for v in cds.data_vars:
    a = cds[v].attrs
    s = stats(cds[v])
    log(f"[CLOUD VAR] {v}: dims={cds[v].dims} units={a.get('units','?')} long_name={a.get('long_name','?')} "
        f"min={s['min']:.6g} max={s['max']:.6g} nan={s['nan']}/{s['size']}")
    clines.append(f"| {v} | {cds[v].dims} | {a.get('units','?')} | {a.get('long_name','?')} | "
                  f"{s['min']:.6g} | {s['max']:.6g} | {s['nan_ratio']*100:.2f}% |")
for c in ["valid_time", "time"]:
    if c in cds.coords:
        clines.append(f"- {c}: {cds[c].values[0]} .. {cds[c].values[-1]}")
if "pressure_level" in cds.coords or "level" in cds.coords:
    lc = "pressure_level" if "pressure_level" in cds.coords else "level"
    clines.append(f"- {lc}: {list(cds[lc].values)}")
for c in ["latitude", "longitude"]:
    if c in cds.coords:
        vv = cds[c].values
        clines.append(f"- {c}: {vv.min()} .. {vv.max()} (n={len(vv)})")
clines += ["", "NOTE: clwc/ciwc are the only cloud-water variables present here; "
              "crwc/cswc remain NOT_RUN (need CDS download, deferred to Phase 1).", ""]
open(CLOUD_MD, "w").write("\n".join(clines))
log(f"[SUMMARY] wrote {CLOUD_MD}")
log("[DONE] ok")
