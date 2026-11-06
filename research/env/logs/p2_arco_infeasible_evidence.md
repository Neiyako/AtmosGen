# P2 阻塞证据：0.25° 全 37 层 Zarr 切片今晚不可行（实测，2026-11-06）

## 1. 两个候选 Zarr 的属性（.zmetadata 匿名 HTTP 获取，已存 JSON 证据）
| store | HTTP | .zmetadata bytes | arrays | 目标变量 |
|---|---|---|---|---|
| `weatherbench2/datasets/era5/1959-2022-full_37-6h-0p25deg-chunk-1.zarr-v2` | 200 | 31,673 | 36 | temperature/specific_humidity/u/v_component_of_wind/total_column_water_vapour 有；**clwc/ciwc/crwc/cswc 全无** |
| `gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3` | 200 | 132,785 | 277 | +specific_cloud_liquid_water_content(clwc)/specific_cloud_ice_water_content(ciwc)；**crwc/cswc 仍无（需 CDS）** |

- 短名 → 长名映射：t=temperature, q=specific_humidity, u=u_component_of_wind,
  v=v_component_of_wind, clwc=specific_cloud_liquid_water_content,
  ciwc=specific_cloud_ice_water_content, tcwv=total_column_water_vapour（ARCO/WB2 都不用 CDS 短名）。
- 两个 store 的 pressure-level 数组 chunk 均为 `[1, 37, 721, 1440]`（time=1, 全 37 层, 全球 0.25°）。

## 2. 单 chunk 体积（curl HEAD，ARCO temperature，2020-01-01T00 对应 index 1051896）
```
$ curl -sI ".../ar/full_37-1h-0p25deg-chunk-1.zarr-v3/temperature/1051896.0.0.0"
HTTP/2 200
content-length: 85235158            # 85,235,158 B ≈ 85.2 MB（单个变量单个时次！）
```
（同目录相邻 chunk 1000000.0.0.0 = 85,215,171 B，1000001.0.0.0 = 85,220,418 B）

chunk key 是 **4 段**：`temperature/<time_idx>.0.0.0`（4D 数组补 0；`dimension_separator` 未设置 = zarr v2 默认 "."）。
早前 `.../temperature/1051896.0.0` 报 404 就是少了第 4 段。

## 3. 本机到 GCS 的实测吞吐（curl Range 限时）
```
$ curl -s --max-time 120 -r 0-5242879 .../temperature/1051896.0.0.0   # 12s 内只拿到 4.6MB
http=206 bytes=4598275 time=120.004429s speed=38317B/s                # 38.3 KB/s
```
=> 单 chunk 85.2 MB 需 ≈ 85,235,158 / 38,317 ≈ **37.1 分钟**。
=> 7 变量 × 1 时次 ≈ 5×85MB ≈ 426 MB ≈ **3.1 小时**；× 8 时次 ≈ **25 小时**。今晚不可行。

## 4. ARCO 时间轴与索引（用 ds.time.sel/get_loc，别硬编码）
- `xr.open_zarr(mapper, consolidated=True)` 11.3s 打开；dims time=1,323,648, level=37, lat=721, lon=1440；nvars=273。
- time: 1900-01-01T00 … 2050-12-31T23，units `hours since 1900-01-01`。
- 2020-01-01T00 → index **1,051,896**（`ds.time.values` 读回实测确认；脚本里应改用
  `ds.indexes["time"].get_loc(pd.Timestamp("2020-01-01T00"))` 或 `ds.sel(time=..., method="nearest").time.item()`）。
- 时间坐标本身读取 ~0.0s（随 open_zarr 一并载入，67706/块）。

## 5. Phase 1 明日重跑方案（按优先级）
1. **先量别的 store 的 chunk**：`ar/1959-2022-6h-1440x721.zarr`、
   `ar/1959-2022-full_37-6h-0p25deg_derived.zarr`、`ar/1959-2022-wb13-6h-0p25deg-chunk-1.zarr-v2`
   （HEAD `.zarray` 看 chunks；若是空间分块则切片成本大降）。一条命令即可：
   `curl -s "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/<store>/.zmetadata" | python3 -c "..."`。
2. **流式 + 断点续传**：单变量单时次下载 → 立即 subset(level/lat/lon) → 释放；`curl -C -` 或 gcsfs
   `Range` 分片并发（先测 4 连接是否总吞吐线性提升；若瓶颈是单连接限速，并发可提速 4 倍）。
3. **CDS API**（若主人有 key）：`cdsapi` 请求 pressure-level 子集（t/q/u/v/clwc/ciwc/tcwv + crwc/cswc），
   服务端裁好再下载，几十 MB 级；无 key 则继续走 ARCO。
4. 明晚以前先落地 **WB2 轻量 store 真值切片**（已落，见 data/raw/era5_slice_SUMMARY.md）跑通 Test A。

## 6. 相关日志
- `research/env/logs/p2_wb2_zmetadata.json` / `p2_arco_zmetadata.json`（原始 .zmetadata）
- `research/env/logs/p2_store_metadata_check.log`（变量清单与 chunk 形状）
- `research/env/logs/p2_arco_probe.log`（open_zarr 11.3s、time 轴、index 计算）
- `research/env/logs/p2_wb2_read.log`（今晚降级源切片读取）
