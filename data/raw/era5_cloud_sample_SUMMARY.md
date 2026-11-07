# ERA5 cloud-water sample summary (Phase 0 / P2b)

- Generated: 2026-11-06 23:31:59 Asia/Shanghai
- **Read-only declaration**: the original DBAS download was NOT modified; a byte-copy was placed in data/raw/ for offline reference.
- Source: `<local>/era5数据dbas/data/downloads/reanalysis-era5-pressure-levels/2024/20240101_multi_fdb02e.nc` (45855 B)
- Copy: `data/raw/era5_cloud_sample_20240101.nc` (45855 B)

## dims

```
{'valid_time': 1, 'pressure_level': 3, 'latitude': 61, 'longitude': 141}
```

## coords

```
{'number': ((), 1), 'valid_time': (('valid_time',), 1), 'pressure_level': (('pressure_level',), 3), 'latitude': (('latitude',), 61), 'longitude': (('longitude',), 141), 'expver': ((), 1)}
```

## variables

| var | dims | units | long_name | min | max | nan ratio |
|---|---|---|---|---|---|---|
| clwc | ('valid_time', 'pressure_level', 'latitude', 'longitude') | kg kg**-1 | Specific cloud liquid water content | 0 | 0.000166178 | 0.00% |
| ciwc | ('valid_time', 'pressure_level', 'latitude', 'longitude') | kg kg**-1 | Specific cloud ice water content | 0 | 0.000104904 | 0.00% |
- valid_time: 2024-01-01T00:00:00.000000000 .. 2024-01-01T00:00:00.000000000
- pressure_level: [np.float64(500.0), np.float64(400.0), np.float64(300.0)]
- latitude: 25.0 .. 40.0 (n=61)
- longitude: 70.0 .. 105.0 (n=141)

NOTE: clwc/ciwc are the only cloud-water variables present here; crwc/cswc remain NOT_RUN (need CDS download, deferred to Phase 1).
