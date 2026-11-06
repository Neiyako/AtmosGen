# Local ERA5 slice summary (Phase 0 / P2b, 2010-05-01)

- Generated: 2026-11-06 23:31:59 Asia/Shanghai
- **Read-only declaration**: source NetCDF files were opened read-only via xarray; nothing on /Volumes/Untitled or in the DBAS dir was modified, moved or re-written.
- Source (local, external volume):
  - `t` -> `/Volumes/Untitled/2010/ERA5_025dg_20100501_temperature.nc` (251416864 B / 251.4 MB)
  - `q` -> `/Volumes/Untitled/2010/ERA5_025dg_20100501_specific_humidity.nc` (251416884 B / 251.4 MB)
  - `u` -> `/Volumes/Untitled/2010/ERA5_025dg_20100501_u_component_of_wind.nc` (251416876 B / 251.4 MB)
  - `v` -> `/Volumes/Untitled/2010/ERA5_025dg_20100501_v_component_of_wind.nc` (251416876 B / 251.4 MB)
- Variable mapping: `t=temperature, q=specific_humidity, u=u_component_of_wind, v=v_component_of_wind` (data vars renamed to short names in the output file)
- Native resolution: 0.25 deg global, 37 pressure levels, hourly (see source attrs below)
- Extraction: time 2010-05-01 00/06/12/18 (4 steps), level [500,700,850] hPa, lat 20-45N, lon 100-125E
- Time-axis consistency of the four source files: **True** (identical time arrays)
- Total read+write wall time: 1.3s; NetCDF write: 0.3s

## dims

```
{'longitude': 101, 'latitude': 101, 'level': 3, 'time': 4}
```

## coords

```
{'longitude': (('longitude',), 101), 'latitude': (('latitude',), 101), 'level': (('level',), 3), 'time': (('time',), 4)}
```

## variables (units / long_name)

| var | dims | units | long_name |
|---|---|---|---|
| t | ('time', 'level', 'latitude', 'longitude') | K | Temperature |
| q | ('time', 'level', 'latitude', 'longitude') | kg kg**-1 | Specific humidity |
| u | ('time', 'level', 'latitude', 'longitude') | m s**-1 | U component of wind |
| v | ('time', 'level', 'latitude', 'longitude') | m s**-1 | V component of wind |

## time range

- start: `2010-05-01T00:00:00.000000000`
- end: `2010-05-01T18:00:00.000000000`
- steps: 4 (step 21600000000000 nanoseconds)
- level (hPa): [500, 700, 850]
- latitude: 20.0 .. 45.0 (n=101, descending=True)
- longitude: 100.0 .. 125.0 (n=101, descending=False)

## value ranges / missing values (after subset)

| var | min | max | mean | nan count | size | nan ratio |
|---|---|---|---|---|---|---|
| t (t) | 247.983 | 301.147 | 275.76 | 0 | 122412 | 0.00% |
| q (q) | 6.2454e-05 | 0.0157861 | 0.00341575 | 0 | 122412 | 0.00% |
| u (u) | -12.9677 | 40.6768 | 6.54294 | 0 | 122412 | 0.00% |
| v (v) | -24.4553 | 21.1455 | -1.87776 | 0 | 122412 | 0.00% |

## file

- path: `data/raw/era5_local_slice_20100501.nc`
- size: 1002210 bytes (1.002 MB)
- <10MB: YES
