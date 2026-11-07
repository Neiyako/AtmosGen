"""Self-written 2D advection-diffusion solver (Mode 2, plan §4.2 / decision D7).

Equation::

    dc/dt + u * dc/dx + v * dc/dy = kappa * lap(c)

Discretisation: explicit Euler in time, first-order upwind advection, 5-point
central diffusion. Stability is checked **before** stepping (CFL and diffusion
number) and a violation raises — the solver never silently shrinks ``dt``.
Boundary conditions: ``periodic`` (wrap), ``open`` (zero-gradient, edge
replication) and ``fixed_from_real`` (edges held at the real-field edges).

The solver is deliberately small and dependency-free (numpy); it is validated
against analytic solutions by ``scripts/validate_solver_analytic.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import xarray as xr

from ... import __version__
from ...contract import attach_contract, build_contract
from ...runner import ResourceTracker, set_global_seed
from ..base import BaseGenerator, GenerationResult
from .conditions import (
    ConditionError,
    GridGeometry,
    build_geometry,
    build_initial_condition,
)

#: Synthetic time origin used when no real dataset supplies a clock (documented
#: in the contract as a synthetic axis: physics runs are not wall-clock data).
DEFAULT_START_TIME = np.datetime64("2000-01-01T00:00:00")


class PhysicsError(RuntimeError):
    """Raised when the solver cannot run (unstable or invalid configuration)."""



@dataclass
class SolverDiagnostics:
    cfl_x: float
    cfl_y: float
    diffusion_number: float
    steps: int
    saved_fields: int
    boundary: str
    initial_condition: Mapping[str, Any]
    mass_drift_max: float = 0.0
    mass_series: list[float] = field(default_factory=list)
    cfl_actual: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "cfl_x": self.cfl_x,
            "cfl_y": self.cfl_y,
            "diffusion_number": self.diffusion_number,
            "steps": self.steps,
            "saved_fields": self.saved_fields,
            "boundary": self.boundary,
            "initial_condition": dict(self.initial_condition),
            "mass_drift_max": self.mass_drift_max,
            "mass_series": self.mass_series[:20] + (["…"] if len(self.mass_series) > 20 else []),
            "cfl_actual": self.cfl_actual,
            "warnings": list(self.warnings),
        }


def check_stability(
    *, u0: float, v0: float, kappa: float, dt_s: float, dx_m: float, dy_m: float
) -> dict[str, float]:
    """Return stability numbers and raise :class:`PhysicsError` if violated."""

    if dt_s <= 0 or dx_m <= 0 or dy_m <= 0:
        raise PhysicsError(f"dt/dx/dy 必须 > 0，得到 dt={dt_s}, dx={dx_m}, dy={dy_m}")
    if kappa < 0:
        raise PhysicsError(f"kappa 必须 ≥ 0，得到 {kappa}")
    cfl_x = abs(u0) * dt_s / dx_m
    cfl_y = abs(v0) * dt_s / dy_m
    # 2-D explicit diffusion stability (von Neumann): the diffusion numbers of
    # BOTH axes must sum to <= 0.5, i.e. kappa*dt*(1/dx^2 + 1/dy^2) <= 0.5.
    # Checking kappa*dt/min(dx,dy)^2 <= 0.5 alone is wrong: on a square grid it
    # permits r=0.4 per axis (total 0.8), which explodes within a few steps —
    # caught by the convergence test, fixed here (documented in PHASE3_REPORT).
    diffusion = kappa * dt_s * (1.0 / dx_m**2 + 1.0 / dy_m**2)
    problems = []
    if cfl_x > 1.0:
        problems.append(f"CFL_x={cfl_x:.3f}>1")
    if cfl_y > 1.0:
        problems.append(f"CFL_y={cfl_y:.3f}>1")
    if diffusion > 0.5:
        problems.append(f"扩散稳定性 κ·dt·(1/dx²+1/dy²)={diffusion:.3f}>0.5")
    if problems:
        raise PhysicsError(
            "物理参数不满足显式格式稳定性要求: " + "; ".join(problems) + "（减小 dt_s 或增大网格间距）"
        )
    return {"cfl_x": cfl_x, "cfl_y": cfl_y, "diffusion_number": diffusion}


def _shift(array: np.ndarray, offset: int, axis: int, periodic: bool) -> np.ndarray:
    """``out[i] = array[i - offset]``; edge-replicated for non-periodic axes."""

    shifted = np.roll(array, offset, axis=axis)
    if periodic or offset == 0:
        return shifted
    n = array.shape[axis]
    index = [slice(None)] * array.ndim
    edge_value = np.take(array, 0 if offset > 0 else n - 1, axis=axis)
    edge_value = np.expand_dims(edge_value, axis=axis)
    if offset > 0:
        index[axis] = slice(0, min(offset, n))
    else:
        index[axis] = slice(max(n + offset, 0), n)
    shifted[tuple(index)] = edge_value
    return shifted


def _upwind_gradient(
    field: np.ndarray, velocity: float, spacing: float, axis: int, periodic: bool
) -> np.ndarray:
    """First-order upwind ``d(field)/d(axis)`` for one velocity component."""

    if velocity >= 0:  # backward difference
        return (field - _shift(field, 1, axis, periodic)) / spacing
    return (_shift(field, -1, axis, periodic) - field) / spacing


def _laplacian(field: np.ndarray, dx_m: float, dy_m: float, periodic: bool) -> np.ndarray:
    x_plus = _shift(field, -1, 1, periodic)
    x_minus = _shift(field, 1, 1, periodic)
    y_plus = _shift(field, -1, 0, periodic)
    y_minus = _shift(field, 1, 0, periodic)
    return (x_plus - 2.0 * field + x_minus) / dx_m**2 + (
        y_plus - 2.0 * field + y_minus
    ) / dy_m**2


def solve_advection_diffusion(
    *,
    grid: Mapping[str, Any],
    time: Mapping[str, Any],
    parameters: Mapping[str, Any],
    initial_condition: Mapping[str, Any],
    boundary_condition: Mapping[str, Any],
    fields: int | None = None,
    real: xr.Dataset | None = None,
) -> tuple[xr.Dataset, SolverDiagnostics]:
    """Run the solver and return ``(dataset, diagnostics)``.

    ``dataset`` has variable ``c`` with dims ``(time, latitude, longitude)``;
    ``fields`` controls how many snapshots are saved (evenly spaced, including
    the initial state).
    """

    nx = int(grid["nx"])
    ny = int(grid["ny"])
    dx_m = float(grid["dx_m"])
    dy_m = float(grid["dy_m"])
    dt_s = float(time["dt_s"])
    steps = int(time["steps"])
    u0 = float(parameters.get("u0", 0.0))
    v0 = float(parameters.get("v0", 0.0))
    kappa = float(parameters.get("kappa", 0.0))
    bc_kind = str(boundary_condition.get("kind", "periodic"))
    if bc_kind not in {"periodic", "open", "fixed_from_real"}:
        raise PhysicsError(f"未知 boundary_condition.kind={bc_kind!r}")
    periodic = bc_kind == "periodic"
    stability = check_stability(u0=u0, v0=v0, kappa=kappa, dt_s=dt_s, dx_m=dx_m, dy_m=dy_m)

    geometry: GridGeometry = build_geometry(
        nx=nx, ny=ny, dx_m=dx_m, dy_m=dy_m, center=initial_condition.get("center"), real=real
    )
    field, ic_details = build_initial_condition(
        initial_condition, geometry=geometry, real=real
    )
    if field.shape != (ny, nx):
        raise PhysicsError(f"初始场形状 {field.shape} 与网格 ({ny}, {nx}) 不符")
    if not np.all(np.isfinite(field)):
        raise PhysicsError("初始场包含 NaN/Inf；检查 initial_condition 与真实场插值范围")

    diagnostics = SolverDiagnostics(
        cfl_x=stability["cfl_x"],
        cfl_y=stability["cfl_y"],
        diffusion_number=stability["diffusion_number"],
        steps=steps,
        saved_fields=0,
        boundary=bc_kind,
        initial_condition=ic_details,
    )

    schedule_notes: list[str] = []
    save_count = int(fields) if fields is not None else steps + 1
    if save_count < 1:
        raise PhysicsError(f"fields 必须 ≥ 1，得到 {fields}")
    save_count = min(save_count, steps + 1)
    if save_count == steps + 1:
        save_indices = np.arange(0, steps + 1)
    else:
        # Uniform snapshot spacing: rounded linspace gave uneven dt deltas
        # (e.g. 24 fields over 96 steps -> 2400/2400/3000... s), which breaks
        # any residual check that assumes a uniform clock. Use the largest
        # uniform schedule with at most ``fields`` snapshots instead.
        step_every = max(1, steps // save_count)
        save_indices = np.arange(0, steps + 1, step_every)
        if save_indices.size > save_count:
            save_indices = save_indices[:save_count]
        if save_indices[-1] != steps:
            schedule_notes.append(
                f"snapshot schedule: {save_indices.size} fields at uniform "
                f"{step_every}-step intervals; final internal step {steps} "
                "not persisted (fields < steps+1)"
            )

    start_time = DEFAULT_START_TIME
    if real is not None and "time" in real.coords:
        start_time = np.asarray(real["time"].values)[0]
    snapshots: list[np.ndarray] = []
    mass_series: list[float] = []
    c_max = 0.0
    fixed_edges: np.ndarray | None = None
    if bc_kind == "fixed_from_real":
        if real is None:
            diagnostics.warnings.append(
                "boundary_condition=fixed_from_real 但未提供真实场；退化为 open（边缘复制）"
            )
        else:
            fixed_edges = field.copy()  # edges of the initial (real-interpolated) field
    current = field.copy()
    snapshots.append(current.copy())
    mass_series.append(float(current.sum()))

    for step in range(1, steps + 1):
        advection = -u0 * _upwind_gradient(current, u0, dx_m, 1, periodic) - v0 * _upwind_gradient(
            current, v0, dy_m, 0, periodic
        )
        diffusion = kappa * _laplacian(current, dx_m, dy_m, periodic)
        current = current + dt_s * (advection + diffusion)
        if fixed_edges is not None:
            current[0, :] = fixed_edges[0, :]
            current[-1, :] = fixed_edges[-1, :]
            current[:, 0] = fixed_edges[:, 0]
            current[:, -1] = fixed_edges[:, -1]
        if not np.all(np.isfinite(current)):
            raise PhysicsError(f"数值发散：第 {step} 步出现 NaN/Inf（CFL 或扩散数超限）")
        c_max = max(c_max, float(np.max(np.abs(current))))
        mass_series.append(float(current.sum()))
        if step in save_indices:
            snapshots.append(current.copy())

    mass0 = mass_series[0]
    if abs(mass0) > 0:
        diagnostics.mass_drift_max = float(
            max(abs(value - mass0) for value in mass_series) / abs(mass0)
        )
    diagnostics.mass_series = mass_series
    diagnostics.cfl_actual = float(
        max(stability["cfl_x"], stability["cfl_y"]) + 4.0 * stability["diffusion_number"]
    )
    diagnostics.saved_fields = len(snapshots)
    if schedule_notes:
        diagnostics.warnings.extend(schedule_notes)
    if c_max > 1e6:
        diagnostics.warnings.append(f"浓度峰值 {c_max:.3g} 偏大，检查 amplitude/真实场量纲")

    times = start_time + (save_indices.astype("timedelta64[s]") * int(dt_s))
    dataset = xr.Dataset(
        data_vars={
            "c": (
                ("time", "latitude", "longitude"),
                np.stack(snapshots, axis=0),
                {
                    "units": "1",
                    "long_name": "Tracer concentration (advection-diffusion)",
                    "solver": "explicit_euler_upwind_central_diffusion",
                },
            )
        },
        coords={
            "time": times,
            "latitude": geometry.latitude,
            "longitude": geometry.longitude,
        },
    )
    return dataset, diagnostics


class AdvectionDiffusionGenerator(BaseGenerator):
    """Mode 2 generator: runs the solver configured under ``physics:``."""

    name = "advection_diffusion"
    mode = "physics"
    description = "自研最小 2D 平流-扩散显式求解器（迎风 + 中心扩散）"
    implemented_phase = 3

    def __init__(
        self,
        config: Any = None,
        *,
        seed: int = 42,
        config_path: str | None = None,
        config_sha256: str | None = None,
        trained_from: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(config, seed=seed)
        set_global_seed(seed)
        self.config_path = config_path
        self.config_sha256 = config_sha256
        self.trained_from = dict(trained_from) if trained_from else None
        self._real: xr.Dataset | None = None
        self.fit_resources: dict[str, Any] = {}
        self.sample_resources: dict[str, Any] = {}

    @classmethod
    def dependency_problems(cls) -> list[str]:
        problems = []
        for module in ("numpy", "xarray"):
            try:
                __import__(module)
            except Exception as exc:  # pragma: no cover
                problems.append(f"missing module {module!r}: {exc}")
        return problems

    def fit(self, dataset: xr.Dataset | None = None) -> "AdvectionDiffusionGenerator":
        self._real = dataset
        self._fitted = True
        return self

    def sample(self) -> GenerationResult:
        self.require_fitted()
        physics = getattr(self.config, "physics", None)
        if physics is None:
            raise PhysicsError("mode=physics 需要 physics 段（equation/grid/time/parameters/IC/BC）")
        grid = physics.grid.model_dump()
        time = physics.time.model_dump()
        parameters = physics.parameters.model_dump()
        initial_condition = physics.initial_condition.model_dump()
        boundary_condition = physics.boundary_condition.model_dump()
        samples = getattr(getattr(self.config, "generation", None), "samples", None)
        fields = getattr(samples, "fields", None) if samples is not None else None

        tracker = ResourceTracker(label=f"physics:{self.name}").start()
        try:
            dataset, diagnostics = solve_advection_diffusion(
                grid=grid,
                time=time,
                parameters=parameters,
                initial_condition=initial_condition,
                boundary_condition=boundary_condition,
                fields=int(fields) if fields is not None else None,
                real=self._real,
            )
        finally:
            self.sample_resources = tracker.stop()

        contract = build_contract(
            dataset,
            generation_method="physics",
            source={
                "kind": "synthetic",
                "dataset": "physics.advection_diffusion",
                "uri": None,
                "trained_from": self.trained_from,
                "access_date": None,
                "readonly": True,
                "license": "AtmosGen self-written solver (no third-party code vendored)",
            },
            model_name="physics.advection_diffusion",
            model_version=__version__,
            random_seed=int(self.seed),
            initial_conditions={
                "kind": initial_condition.get("kind"),
                "details": diagnostics.initial_condition,
            },
            boundary_conditions={"kind": boundary_condition.get("kind")},
            physical_parameters={
                "u0": parameters.get("u0"),
                "v0": parameters.get("v0"),
                "kappa": parameters.get("kappa"),
                "grid": grid,
                "dt_s": time.get("dt_s"),
                "steps": time.get("steps"),
                "equation": physics.equation,
            },
            generation_config={
                "config_path": self.config_path,
                "config_sha256": self.config_sha256,
                "samples": {
                    "unit": "field",
                    "fields": diagnostics.saved_fields,
                    "requested_fields": fields,
                },
                "discretisation": {
                    "time_scheme": "explicit_euler",
                    "advection": "first_order_upwind",
                    "diffusion": "five_point_central",
                },
                "diagnostics": {
                    "cfl_x": diagnostics.cfl_x,
                    "cfl_y": diagnostics.cfl_y,
                    "diffusion_number": diagnostics.diffusion_number,
                    "mass_drift_max": diagnostics.mass_drift_max,
                },
            },
            metadata={
                "notes": (
                    "Mode 2 physics generation (self-written solver); time axis is synthetic "
                    "unless initial_condition.kind=from_real supplies a real clock. "
                    "Solver validated against analytic solutions by "
                    "scripts/validate_solver_analytic.py."
                ),
                "elapsed_seconds": self.sample_resources.get("elapsed_seconds"),
                "peak_rss_mb": self.sample_resources.get("peak_rss_mb"),
            },
        )
        dataset = attach_contract(dataset, contract)
        return GenerationResult(
            dataset=dataset,
            contract=contract,
            metadata={
                "diagnostics": diagnostics.as_dict(),
                "resources": self.sample_resources,
                "initial_condition": initial_condition,
                "boundary_condition": boundary_condition,
            },
        )


__all__ = [
    "AdvectionDiffusionGenerator",
    "DEFAULT_START_TIME",
    "PhysicsError",
    "SolverDiagnostics",
    "check_stability",
    "solve_advection_diffusion",
]
