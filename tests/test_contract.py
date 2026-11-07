"""Contract construction / validation / embedding."""

from __future__ import annotations

import json

import pytest
import xarray as xr

from atmosgen import contract as c
from atmosgen.contract import (
    CONTRACT_ATTR,
    ContractError,
    attach_contract,
    build_contract,
    contract_problems,
    read_contract,
    validate_contract,
    write_contract_file,
)


def test_build_contract_covers_required_fields(small_ds):
    doc = build_contract(
        small_ds,
        generation_method="statistical",
        source={"kind": "synthetic", "dataset": "unit-test"},
        model_name="sdv.GaussianCopulaSynthesizer",
        random_seed=42,
    )
    assert c.contract_problems(doc) == []
    for field in c.REQUIRED_FIELDS:
        assert field in doc, field
    assert doc["variables"] == ["t", "q", "u", "v", "tcwv"]
    assert doc["units"]["t"] == "K"
    assert doc["coordinates"]["time"]["step_hours"] == 6.0
    assert doc["coordinates"]["level"]["type"] == "pressure_level"
    assert doc["coordinates"]["latitude"]["ascending"] is True
    assert doc["generation_method"] == "statistical"


def test_unknown_variable_is_rejected(small_ds):
    ds = small_ds.assign(pressure=("time", [1.0, 2.0, 3.0]))
    with pytest.raises(ContractError) as excinfo:
        build_contract(ds, generation_method="ingest", source={"kind": "real"})
    assert "outside the AtmosGen schema" in str(excinfo.value)


def test_missing_required_coordinate_is_rejected(small_ds):
    ds = small_ds.drop_vars("longitude")
    with pytest.raises(ContractError) as excinfo:
        build_contract(ds, generation_method="ingest", source={"kind": "real"})
    assert "longitude" in str(excinfo.value)


def test_schema_major_mismatch_is_detected(small_ds):
    doc = build_contract(small_ds, generation_method="ingest", source={"kind": "real"})
    doc["schema_version"] = "9.0"
    problems = contract_problems(doc)
    assert any("unsupported schema major" in problem for problem in problems)


def test_attach_and_read_contract_roundtrip(small_ds):
    doc = build_contract(small_ds, generation_method="physics", source={"kind": "synthetic"})
    attached = attach_contract(small_ds, doc)
    assert isinstance(attached.attrs[CONTRACT_ATTR], str)
    json.loads(attached.attrs[CONTRACT_ATTR])
    assert read_contract(attached)["generation_method"] == "physics"
    # flat duplicates for non-JSON tooling
    assert attached.attrs["atmosgen_schema_version"] == c.SCHEMA_VERSION
    assert attached.attrs["atmosgen_variables"] == "t,q,u,v,tcwv"


def test_read_contract_requires_attribute(small_ds):
    with pytest.raises(ContractError):
        read_contract(small_ds)


def test_contract_file_roundtrip(tmp_path, small_ds):
    doc = build_contract(small_ds, generation_method="ingest", source={"kind": "real"})
    path = write_contract_file(tmp_path / "contract.json", doc)
    assert c.read_contract_file(path)["generation_method"] == "ingest"


def test_units_fall_back_to_schema(small_ds):
    ds = small_ds.copy()
    ds["t"].attrs.pop("units")
    doc = build_contract(ds, generation_method="ingest", source={"kind": "real"})
    assert doc["units"]["t"] == "K"


def test_level_type_override_is_recorded(small_ds):
    ds = small_ds.copy()
    ds["level"] = ds["level"].astype("float64").assign_attrs(units="1")
    inferred = build_contract(ds, generation_method="ingest", source={"kind": "real"})
    # dimensionless level units alone are ambiguous; the explicit config wins
    declared = build_contract(
        ds, generation_method="ingest", source={"kind": "real"}, level_type="model_level"
    )
    assert declared["coordinates"]["level"]["type"] == "model_level"
    assert inferred["coordinates"]["level"]["type"] in {"pressure_level", "model_level"}
    with pytest.raises(ContractError):
        build_contract(
            ds, generation_method="ingest", source={"kind": "real"}, level_type="sigma_level"
        )


def test_validate_contract_checks_dims(small_ds):
    doc = build_contract(small_ds, generation_method="ingest", source={"kind": "real"})
    broken = small_ds.copy()
    broken["t"] = broken["t"].isel(level=0)  # drop level dimension
    problems = contract_problems(doc, broken)
    assert any("dims" in problem for problem in problems)
