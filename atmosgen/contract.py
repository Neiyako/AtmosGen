"""Unified dataset contract for AtmosGen (docs/DATA_SCHEMA.md).

The contract is a JSON document attached to every canonical dataset:

- NetCDF: global attribute ``atmosgen:contract`` (plus a few flat ``atmosgen_*``
  attributes for convenience);
- Zarr: root attribute ``atmosgen:contract``;
- always duplicated as a standalone ``contract.json`` next to the data file so
  it can be read without xarray.

It declares the 14 fields required by the task specification:
variables / coordinates / units / source / generation_method / model_name /
model_version / random_seed / initial_conditions / boundary_conditions /
physical_parameters / generation_config / quality_metrics / metadata.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import xarray as xr

from . import __version__

SCHEMA_VERSION = "1.0"
CONTRACT_ATTR = "atmosgen:contract"
CONTRACT_FILENAME = "contract.json"

#: Fields that must exist in every contract document (14 + 2 bookkeeping fields).
REQUIRED_FIELDS: tuple[str, ...] = (
    "schema_version",
    "atmosgen_version",
    "variables",
    "coordinates",
    "units",
    "source",
    "generation_method",
    "model_name",
    "model_version",
    "random_seed",
    "initial_conditions",
    "boundary_conditions",
    "physical_parameters",
    "generation_config",
    "quality_metrics",
    "metadata",
)

GENERATION_METHODS: tuple[str, ...] = ("ingest", "statistical", "physics", "hybrid")
SOURCE_KINDS: tuple[str, ...] = ("real", "derived", "synthetic", "mixed")

#: Minimum set of coordinate descriptions that must be present.
REQUIRED_COORDS: tuple[str, ...] = ("time", "latitude", "longitude")


class ContractError(ValueError):
    """Raised when a dataset/contract does not satisfy the AtmosGen schema."""


@dataclass(frozen=True)
class VariableSpec:
    """Canonical definition of one AtmosGen variable."""

    name: str
    dims: tuple[str, ...]
    units: str
    long_name: str
    standard_name: str | None
    category: str


_VAR_SPECS: tuple[VariableSpec, ...] = (
    VariableSpec("t", ("time", "level", "latitude", "longitude"), "K", "Temperature", "air_temperature", "thermodynamic"),
    VariableSpec("q", ("time", "level", "latitude", "longitude"), "kg kg**-1", "Specific humidity", "specific_humidity", "thermodynamic"),
    VariableSpec("u", ("time", "level", "latitude", "longitude"), "m s**-1", "U component of wind", "eastward_wind", "dynamic"),
    VariableSpec("v", ("time", "level", "latitude", "longitude"), "m s**-1", "V component of wind", "northward_wind", "dynamic"),
    VariableSpec("clwc", ("time", "level", "latitude", "longitude"), "kg kg**-1", "Specific cloud liquid water content", "specific_cloud_liquid_water_content", "cloud"),
    VariableSpec("ciwc", ("time", "level", "latitude", "longitude"), "kg kg**-1", "Specific cloud ice water content", "specific_cloud_ice_water_content", "cloud"),
    VariableSpec("crwc", ("time", "level", "latitude", "longitude"), "kg kg**-1", "Specific rain water content", "specific_rain_water_content", "cloud"),
    VariableSpec("cswc", ("time", "level", "latitude", "longitude"), "kg kg**-1", "Specific snow water content", "specific_snow_water_content", "cloud"),
    VariableSpec("tcwv", ("time", "latitude", "longitude"), "kg m**-2", "Total column water vapour", "atmosphere_mass_content_of_water_vapor", "column"),
)

VAR_SPECS: dict[str, VariableSpec] = {spec.name: spec for spec in _VAR_SPECS}


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def known_variables() -> list[str]:
    """Return the canonical variable names in schema order."""

    return [spec.name for spec in _VAR_SPECS]


def spec_for(name: str) -> VariableSpec:
    """Return the ``VariableSpec`` for a canonical variable name."""

    try:
        return VAR_SPECS[name]
    except KeyError:
        raise ContractError(
            f"unknown variable {name!r}; canonical variables are {known_variables()}"
        ) from None


def utc_now_iso() -> str:
    """Current UTC timestamp in ISO-8601 (seconds resolution)."""

    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def canonical_json(obj: Mapping[str, Any]) -> str:
    """Deterministic JSON serialisation used for hashing and attribute storage."""

    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file (streamed, so large NetCDF files are fine)."""

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    """SHA-256 of a string."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def host_metadata() -> dict[str, Any]:
    """Environment fingerprint written into ``metadata``."""

    return {
        "host": f"{platform.system()} {platform.machine()}",
        "hostname": platform.node(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "xarray": xr.__version__,
        "platform": platform.platform(),
    }


def _coord_units(coord: xr.DataArray) -> str | None:
    units = coord.attrs.get("units")
    if units:
        return str(units)
    encoding_units = coord.encoding.get("units") if coord.encoding else None
    return str(encoding_units) if encoding_units else None


def _coord_step(values: np.ndarray, axis: str) -> float:
    if values.size < 2:
        return 0.0
    if axis == "time":
        deltas = np.diff(values.astype("datetime64[ns]").astype("int64"))
        return float(np.median(deltas) / 3.6e12)  # ns -> hours
    deltas = np.diff(np.asarray(values, dtype=float))
    if deltas.size == 0:
        return 0.0
    return float(np.round(np.median(np.abs(deltas)), 10))


def _time_description(time: xr.DataArray) -> dict[str, Any]:
    values = np.asarray(time.values)
    if values.dtype.kind != "M":
        raise ContractError(
            "time coordinate must be datetime64; pass decode_times=True when opening the source"
        )
    values = values.astype("datetime64[ns]")
    return {
        "name": time.name,
        "units": _coord_units(time) or "datetime64[ns] (UTC, naive)",
        "calendar": time.attrs.get("calendar", "proleptic_gregorian"),
        "range": [
            np.datetime_as_string(values.min(), unit="s").replace("T", "T"),
            np.datetime_as_string(values.max(), unit="s"),
        ],
        "step_hours": _coord_step(values, "time"),
    }


def _level_description(level: xr.DataArray, *, level_type: str | None = None) -> dict[str, Any]:
    values = np.asarray(level.values)
    units = (_coord_units(level) or "").lower()
    if level_type is not None:
        if level_type not in {"pressure_level", "model_level"}:
            raise ContractError(
                f"level_type must be pressure_level or model_level, got {level_type!r}"
            )
        return {
            "name": level.name,
            "type": level_type,
            "units": _coord_units(level) or ("hPa" if level_type == "pressure_level" else "1"),
            "values": [float(v) if not float(v).is_integer() else int(v) for v in values],
        }
    vertical = str(level.attrs.get("vertical", "")).lower()
    if vertical in {"pressure_level", "model_level"}:
        kind = vertical
    elif units in {"hpa", "mb", "millibars", "millibar", "pa", "hpaa"}:
        kind = "pressure_level"
    elif units.isdigit() or units == "":
        # No usable unit: pressure levels are the only supported vertical axis
        # with dimensionless-looking values in the ERA5 archive, so be explicit
        # in the contract instead of guessing silently.
        kind = "pressure_level"
    else:
        kind = "model_level"
    return {
        "name": level.name,
        "type": kind,
        "units": _coord_units(level) or ("hPa" if kind == "pressure_level" else "1"),
        "values": [float(v) if not float(v).is_integer() else int(v) for v in values],
    }


def _horizontal_description(coord: xr.DataArray, *, axis: str) -> dict[str, Any]:
    values = np.asarray(coord.values, dtype=float)
    step = _coord_step(values, axis)
    meta: dict[str, Any] = {
        "name": coord.name,
        "units": _coord_units(coord) or ("degrees_north" if axis == "latitude" else "degrees_east"),
        "range": [float(values.min()), float(values.max())],
        "step": step,
    }
    if axis == "latitude":
        meta["ascending"] = bool(values.size < 2 or values[-1] > values[0])
    else:
        meta["range_style"] = "0-360" if float(values.max()) > 180.0 else "-180-180"
        meta["ascending"] = bool(values.size < 2 or values[-1] > values[0])
    return meta


def _coordinates_block(ds: xr.Dataset, *, level_type: str | None = None) -> dict[str, Any]:
    block: dict[str, Any] = {}
    for name in ds.coords:
        coord = ds.coords[name]
        if name == "time":
            block["time"] = _time_description(coord)
        elif name == "level":
            block["level"] = _level_description(coord, level_type=level_type)
        elif name == "latitude":
            block["latitude"] = _horizontal_description(coord, axis="latitude")
        elif name == "longitude":
            block["longitude"] = _horizontal_description(coord, axis="longitude")
    problems = [name for name in REQUIRED_COORDS if name not in block]
    if problems:
        raise ContractError(f"dataset is missing required coordinates: {problems}")
    return block


# --------------------------------------------------------------------------- #
# Build / attach / read
# --------------------------------------------------------------------------- #
def build_contract(
    ds: xr.Dataset,
    *,
    generation_method: str,
    source: Mapping[str, Any],
    model_name: str | None = None,
    model_version: str | None = None,
    random_seed: int | None = None,
    initial_conditions: Mapping[str, Any] | None = None,
    boundary_conditions: Mapping[str, Any] | None = None,
    physical_parameters: Mapping[str, Any] | None = None,
    generation_config: Mapping[str, Any] | None = None,
    quality_metrics: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    units: Mapping[str, str] | None = None,
    level_type: str | None = None,
    atmosgen_version: str | None = None,
) -> dict[str, Any]:
    """Build a contract document describing ``ds``.

    Parameters not provided by the producer default to ``None`` (explicitly
    "not applicable") rather than being invented, so that readers can tell
    "statistical generation" from "physics generation".
    """

    if generation_method not in GENERATION_METHODS:
        raise ContractError(
            f"generation_method must be one of {GENERATION_METHODS}, got {generation_method!r}"
        )
    source_dict = dict(source)
    if source_dict.get("kind") not in SOURCE_KINDS:
        raise ContractError(
            f"source.kind must be one of {SOURCE_KINDS}, got {source_dict.get('kind')!r}"
        )

    variables = [name for name in ds.data_vars if name in VAR_SPECS]
    unknown = [name for name in ds.data_vars if name not in VAR_SPECS]
    if unknown:
        raise ContractError(
            f"dataset contains variables outside the AtmosGen schema: {unknown}; "
            f"canonical variables are {known_variables()}"
        )
    if not variables:
        raise ContractError("dataset has no canonical AtmosGen variables")

    unit_map: dict[str, str] = {}
    for name in variables:
        declared = (units or {}).get(name) or ds[name].attrs.get("units") or VAR_SPECS[name].units
        unit_map[name] = str(declared)

    metadata_dict = host_metadata()
    metadata_dict.update(
        {
            "created_at": utc_now_iso(),
            "env_lock": "research/env/logs/atmosgen_pip_freeze.txt",
            "notes": "",
        }
    )
    if metadata:
        metadata_dict.update(dict(metadata))

    return {
        "schema_version": SCHEMA_VERSION,
        "atmosgen_version": atmosgen_version or __version__,
        "variables": variables,
        "coordinates": _coordinates_block(ds, level_type=level_type),
        "units": unit_map,
        "source": source_dict,
        "generation_method": generation_method,
        "model_name": model_name,
        "model_version": model_version,
        "random_seed": random_seed,
        "initial_conditions": dict(initial_conditions) if initial_conditions else None,
        "boundary_conditions": dict(boundary_conditions) if boundary_conditions else None,
        "physical_parameters": dict(physical_parameters) if physical_parameters else None,
        "generation_config": dict(generation_config) if generation_config else None,
        "quality_metrics": dict(quality_metrics) if quality_metrics else None,
        "metadata": metadata_dict,
    }


def contract_problems(
    contract: Mapping[str, Any], ds: xr.Dataset | None = None
) -> list[str]:
    """Return every schema violation found in ``contract`` (empty = valid).

    Collecting all problems instead of failing on the first one keeps the
    repair loop cheap: one run tells the caller everything that is wrong.
    """

    problems: list[str] = []
    if not isinstance(contract, Mapping):
        return [f"contract must be a mapping, got {type(contract).__name__}"]

    missing = [field for field in REQUIRED_FIELDS if field not in contract]
    if missing:
        problems.append(f"missing required fields: {missing}")

    schema = contract.get("schema_version")
    if not isinstance(schema, str) or not schema:
        problems.append("schema_version must be a non-empty string")
    else:
        try:
            major = int(schema.split(".")[0])
        except ValueError:
            problems.append(f"schema_version {schema!r} is not '<major>.<minor>'")
        else:
            if major != int(SCHEMA_VERSION.split(".")[0]):
                problems.append(
                    f"unsupported schema major version {major} (reader supports {SCHEMA_VERSION})"
                )

    method = contract.get("generation_method")
    if method not in GENERATION_METHODS:
        problems.append(f"generation_method {method!r} not in {GENERATION_METHODS}")

    variables = contract.get("variables")
    if not isinstance(variables, list) or not variables:
        problems.append("variables must be a non-empty list")
        variables = []
    else:
        bad = [name for name in variables if name not in VAR_SPECS]
        if bad:
            problems.append(f"unknown variables {bad}")
        if len(set(variables)) != len(variables):
            problems.append("variables contains duplicates")

    coords = contract.get("coordinates")
    if not isinstance(coords, Mapping):
        problems.append("coordinates must be a mapping")
    else:
        for name in REQUIRED_COORDS:
            if name not in coords:
                problems.append(f"coordinates missing required coordinate {name!r}")
        level = coords.get("level")
        if level is not None:
            if not isinstance(level, Mapping):
                problems.append("coordinates.level must be a mapping")
            elif level.get("type") not in {"pressure_level", "model_level"}:
                problems.append(
                    f"coordinates.level.type {level.get('type')!r} must be pressure_level or model_level"
                )

    units = contract.get("units")
    if not isinstance(units, Mapping):
        problems.append("units must be a mapping")
    else:
        for name in variables:
            if not units.get(name):
                problems.append(f"units missing entry for variable {name!r}")

    source = contract.get("source")
    if not isinstance(source, Mapping):
        problems.append("source must be a mapping")
    elif source.get("kind") not in SOURCE_KINDS:
        problems.append(f"source.kind {source.get('kind')!r} not in {SOURCE_KINDS}")

    meta = contract.get("metadata")
    if not isinstance(meta, Mapping):
        problems.append("metadata must be a mapping")
    elif not meta.get("created_at"):
        problems.append("metadata.created_at is required")

    if ds is not None:
        for name in variables:
            if name not in ds.data_vars:
                problems.append(f"dataset does not contain declared variable {name!r}")
                continue
            expected = VAR_SPECS[name].dims
            if tuple(ds[name].dims) != expected:
                problems.append(
                    f"variable {name!r} has dims {tuple(ds[name].dims)}, schema requires {expected}"
                )

    try:
        canonical_json(contract)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        problems.append(f"contract is not JSON serialisable: {exc}")
    return problems


def validate_contract(contract: Mapping[str, Any], ds: xr.Dataset | None = None) -> None:
    """Raise :class:`ContractError` listing every violation of the schema."""

    problems = contract_problems(contract, ds)
    if problems:
        raise ContractError("contract validation failed: " + "; ".join(problems))


def attach_contract(ds: xr.Dataset, contract: Mapping[str, Any]) -> xr.Dataset:
    """Return ``ds`` with the contract (re)attached, replacing stale attributes."""

    validate_contract(contract, ds)
    out = ds.copy(deep=False)
    attrs = {k: v for k, v in out.attrs.items() if not k.startswith("atmosgen")}
    attrs[CONTRACT_ATTR] = canonical_json(dict(contract))
    # Flat duplicates: convenient for tools that do not parse JSON attributes.
    attrs["atmosgen_schema_version"] = str(contract["schema_version"])
    attrs["atmosgen_generation_method"] = str(contract["generation_method"])
    attrs["atmosgen_model_name"] = str(contract.get("model_name") or "null")
    attrs["atmosgen_variables"] = ",".join(contract["variables"])
    out.attrs = attrs
    return out


def read_contract(ds: xr.Dataset) -> dict[str, Any]:
    """Read and validate the embedded contract; raises if absent or malformed."""

    raw = ds.attrs.get(CONTRACT_ATTR)
    if raw is None:
        raise ContractError(
            f"dataset has no {CONTRACT_ATTR!r} attribute; it is not an AtmosGen canonical dataset"
        )
    if isinstance(raw, str):
        try:
            contract = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ContractError(f"{CONTRACT_ATTR} is not valid JSON: {exc}") from exc
    else:  # NetCDF may hand back bytes or a list of dicts for some backends
        raise ContractError(
            f"{CONTRACT_ATTR} must be stored as a JSON string, got {type(raw).__name__}"
        )
    validate_contract(contract, ds)
    return contract


def write_contract_file(path: str | Path, contract: Mapping[str, Any]) -> Path:
    """Write a standalone ``contract.json`` next to the data file."""

    validate_contract(contract, None)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def read_contract_file(path: str | Path) -> dict[str, Any]:
    """Read and validate a standalone ``contract.json``."""

    target = Path(path)
    try:
        contract = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ContractError(f"contract file not found: {target}") from exc
    except json.JSONDecodeError as exc:
        raise ContractError(f"contract file {target} is not valid JSON: {exc}") from exc
    validate_contract(contract, None)
    return contract


def with_sha256(contract: dict[str, Any], data_path: str | Path) -> dict[str, Any]:
    """Return a copy of ``contract`` with ``source.sha256`` set from the data file."""

    updated = json.loads(canonical_json(contract))
    updated.setdefault("source", {})["sha256"] = sha256_file(data_path)
    return updated


__all__ = [
    "CONTRACT_ATTR",
    "CONTRACT_FILENAME",
    "ContractError",
    "GENERATION_METHODS",
    "REQUIRED_FIELDS",
    "SCHEMA_VERSION",
    "SOURCE_KINDS",
    "VAR_SPECS",
    "VariableSpec",
    "attach_contract",
    "build_contract",
    "canonical_json",
    "contract_problems",
    "host_metadata",
    "known_variables",
    "read_contract",
    "read_contract_file",
    "sha256_file",
    "sha256_text",
    "spec_for",
    "utc_now_iso",
    "validate_contract",
    "with_sha256",
    "write_contract_file",
]
