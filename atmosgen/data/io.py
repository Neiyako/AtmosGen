"""NetCDF / Zarr IO behind the AtmosGen data contract.

Design rules (docs/DATA_SCHEMA.md):

1. Sources are opened **read-only**: ``open_source`` never offers a write path,
   and ``save_canonical`` refuses to write under ``data/raw/`` unless the
   caller is an ingest script passing ``allow_raw=True`` explicitly.
2. Canonical files always carry an embedded contract (``atmosgen:contract``)
   and a standalone ``contract.json`` sidecar.
3. Writes are atomic: data is written to a sibling temporary path and moved
   into place, so a crashed run cannot leave a half-written dataset that looks
   valid.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import xarray as xr

from ..contract import (
    CONTRACT_ATTR,
    CONTRACT_FILENAME,
    ContractError,
    attach_contract,
    read_contract,
    sha256_file,
    validate_contract,
    write_contract_file,
)

#: Repository root, i.e. the directory that contains ``atmosgen/``.
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

#: Directory that ingest scripts may write once and generation code must never touch.
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw"

_ENGINE_BY_FORMAT = {"netcdf": "netcdf4", "zarr": "zarr"}


class RawWriteError(PermissionError):
    """Raised when code tries to write into a raw (read-only) data directory."""


# --------------------------------------------------------------------------- #
# Paths and guards
# --------------------------------------------------------------------------- #
def project_root() -> Path:
    """Repository root inferred from the package location."""

    return REPO_ROOT


def default_raw_roots() -> tuple[Path, ...]:
    """Raw directories that must stay read-only.

    ``ATMOSGEN_RAW_DIRS`` (``os.pathsep``-separated) extends the default so
    tests and experiments can protect temporary raw trees as well.
    """

    roots = [DEFAULT_RAW_DIR.resolve()]
    extra = os.environ.get("ATMOSGEN_RAW_DIRS")
    if extra:
        roots.extend(Path(part).expanduser().resolve() for part in extra.split(os.pathsep) if part)
    return tuple(dict.fromkeys(roots))


def resolve_path(path: str | Path, *, base: str | Path | None = None) -> Path:
    """Resolve ``path`` against ``base`` (or the CWD) and normalise it."""

    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path(base).expanduser() / candidate if base is not None else candidate
    return candidate.resolve()


def is_within(path: str | Path, root: str | Path) -> bool:
    """True when ``path`` is ``root`` or lives below it."""

    try:
        Path(path).resolve().relative_to(Path(root).resolve())
    except ValueError:
        return False
    return True


def assert_writable_target(
    path: str | Path,
    *,
    allow_raw: bool = False,
    raw_roots: tuple[Path, ...] | None = None,
) -> Path:
    """Validate a write target and return its resolved path.

    Raises :class:`RawWriteError` when the target lives in a raw directory and
    ``allow_raw`` is false. Only ``scripts/ingest_*`` may pass ``allow_raw=True``.
    """

    target = resolve_path(path)
    if not allow_raw:
        roots = raw_roots if raw_roots is not None else default_raw_roots()
        for root in roots:
            if is_within(target, root):
                raise RawWriteError(
                    f"refusing to write {target} inside raw data directory {root}; "
                    "raw data is write-once (ingest scripts only). Write to data/processed/ or output/ instead"
                )
    return target


def file_fingerprint(path: str | Path, *, with_sha256: bool = False) -> dict[str, Any]:
    """Stat-based fingerprint used to prove that a source stayed untouched."""

    target = Path(path)
    stat = target.stat()
    fingerprint: dict[str, Any] = {
        "path": str(target),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if with_sha256:
        fingerprint["sha256"] = sha256_file(target)
    return fingerprint


# --------------------------------------------------------------------------- #
# Open / detect
# --------------------------------------------------------------------------- #
def detect_format(path: str | Path, *, format: str = "auto") -> str:
    """Return ``netcdf`` or ``zarr`` for ``path``."""

    if format not in {"auto", "netcdf", "zarr"}:
        raise ValueError(f"format must be auto|netcdf|zarr, got {format!r}")
    if format != "auto":
        return format
    target = Path(path)
    if target.suffix in {".nc", ".nc4", ".cdf", ".netcdf"}:
        return "netcdf"
    if target.name.endswith(".zarr") or target.is_dir():
        return "zarr"
    raise ValueError(f"cannot infer format from path {target}; pass format='netcdf' or 'zarr'")


def open_source(
    path: str | Path,
    *,
    chunks: Mapping[str, int] | None = None,
    format: str = "auto",
    decode_times: bool = True,
    variables: list[str] | None = None,
) -> xr.Dataset:
    """Open a source dataset read-only.

    The returned dataset is a lazy view: no data is read until accessed, and
    there is deliberately no write API here. Callers must never call
    ``to_netcdf`` on a dataset obtained from this function.
    """

    target = resolve_path(path)
    if not target.exists():
        raise FileNotFoundError(f"source dataset not found: {target}")
    resolved_format = detect_format(target, format=format)
    engine = _ENGINE_BY_FORMAT[resolved_format]
    ds = xr.open_dataset(
        target,
        engine=engine,
        chunks=dict(chunks) if chunks else None,
        decode_times=decode_times,
    )
    if variables:
        missing = [name for name in variables if name not in ds.variables]
        if missing:
            ds.close()
            raise KeyError(
                f"source {target} is missing requested variables {missing}; "
                f"available: {list(ds.data_vars)}"
            )
        ds = ds[list(variables)]
    return ds


# --------------------------------------------------------------------------- #
# Save / read canonical
# --------------------------------------------------------------------------- #
def _netcdf_encoding(
    ds: xr.Dataset,
    *,
    compression: Mapping[str, Any] | None,
    chunks: Mapping[str, int] | None,
) -> dict[str, dict[str, Any]]:
    encoding: dict[str, dict[str, Any]] = {}
    compression = dict(compression or {})
    zlib = bool(compression.get("zlib", True))
    complevel = int(compression.get("complevel", 4))
    for name, variable in ds.data_vars.items():
        var_encoding: dict[str, Any] = {"zlib": zlib}
        if zlib:
            var_encoding["complevel"] = complevel
        if chunks:
            # Unspecified dimensions fall back to their full size, clamped to
            # the actual dimension length so an over-large request is not an error.
            chunksizes = tuple(
                max(1, min(int(chunks.get(dim, ds.sizes[dim])), int(ds.sizes[dim])))
                for dim in variable.dims
            )
            var_encoding["chunksizes"] = chunksizes
        encoding[name] = var_encoding
    return encoding


def save_canonical(
    ds: xr.Dataset,
    path: str | Path,
    *,
    contract: Mapping[str, Any] | None = None,
    format: str = "auto",
    compression: Mapping[str, Any] | None = None,
    chunks: Mapping[str, int] | None = None,
    allow_raw: bool = False,
    overwrite: bool = False,
    write_contract: bool = True,
) -> Path:
    """Write ``ds`` as a canonical NetCDF/Zarr dataset with an embedded contract.

    Returns the written data path. The contract sidecar is written as
    ``<parent>/contract.json`` unless ``write_contract`` is false.
    """

    target = assert_writable_target(path, allow_raw=allow_raw)
    resolved_format = detect_format(target, format=format)

    if contract is not None:
        validate_contract(contract, ds)
        to_write = attach_contract(ds, contract)
    elif CONTRACT_ATTR in ds.attrs:
        validate_contract(read_contract(ds), ds)
        to_write = ds
    else:
        raise ContractError(
            "save_canonical requires a contract: pass contract=... or attach one with attach_contract()"
        )

    if target.exists():
        if not overwrite:
            raise FileExistsError(
                f"{target} already exists; pass overwrite=True (CLI --overwrite) to replace it"
            )
        _remove_target(target)

    target.parent.mkdir(parents=True, exist_ok=True)

    if resolved_format == "netcdf":
        encoding = _netcdf_encoding(to_write, compression=compression, chunks=chunks)
        tmp = target.with_name(f".{target.name}.tmp")
        if tmp.exists():
            tmp.unlink()
        to_write.to_netcdf(tmp, engine="netcdf4", encoding=encoding or None)
        os.replace(tmp, target)
    elif resolved_format == "zarr":
        tmp = target.with_name(f".{target.name}.tmp")
        if tmp.exists():
            _remove_target(tmp)
        to_write.to_zarr(tmp, mode="w", consolidated=True)
        os.replace(tmp, target)
    else:  # pragma: no cover - detect_format guards this
        raise ValueError(f"unsupported format {resolved_format!r}")

    if write_contract:
        contract_doc = to_write.attrs.get(CONTRACT_ATTR)
        if contract_doc is not None:
            import json

            write_contract_file(sidecar_contract_path(target), json.loads(contract_doc))
    return target


def read_canonical(
    path: str | Path,
    *,
    chunks: Mapping[str, int] | None = None,
    format: str = "auto",
    validate: bool = True,
) -> xr.Dataset:
    """Open a canonical dataset and (by default) validate its embedded contract."""

    ds = open_source(path, chunks=chunks, format=format)
    if validate:
        try:
            read_contract(ds)
        except ContractError:
            ds.close()
            raise
    return ds


def sidecar_contract_path(data_path: str | Path) -> Path:
    """Path of the standalone contract file belonging to ``data_path``.

    ``data.nc`` in a run directory keeps the schema-mandated ``contract.json``
    name; any other dataset gets ``<stem>.contract.json`` so two datasets living
    in the same directory cannot overwrite each other's sidecar.
    """

    target = Path(data_path)
    if target.stem == "data":
        return target.parent / CONTRACT_FILENAME
    return target.parent / f"{target.stem}.contract.json"


def _remove_target(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def dataset_size_bytes(path: str | Path) -> int:
    """Total size of a dataset on disk (directory total for Zarr)."""

    target = Path(path)
    if target.is_dir():
        return sum(p.stat().st_size for p in target.rglob("*") if p.is_file())
    return target.stat().st_size


__all__ = [
    "DEFAULT_RAW_DIR",
    "RawWriteError",
    "REPO_ROOT",
    "assert_writable_target",
    "dataset_size_bytes",
    "default_raw_roots",
    "detect_format",
    "file_fingerprint",
    "is_within",
    "open_source",
    "project_root",
    "read_canonical",
    "resolve_path",
    "save_canonical",
    "sidecar_contract_path",
]
