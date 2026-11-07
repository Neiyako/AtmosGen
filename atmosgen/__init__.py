"""AtmosGen — atmospheric synthetic data generation framework.

Layering (see docs/IMPLEMENTATION_PLAN.md §2.1):

- ``contract``  : unified dataset contract (docs/DATA_SCHEMA.md)
- ``data``      : IO (NetCDF/Zarr), preprocessing, tabular views
- ``generators``: model adapters (SDV, Synthcity, self-written physics solver)
- ``validation``: statistical / temporal / spatial / physical metrics and reports
- ``config``    : pydantic configuration models and cross-field validation
- ``runner``    : run directory, seeding, resource accounting
- ``cli``       : typer entry points

Phase 1 implements: contract, IO, config validation, real-data ingest, model
registry and generator base class. Generation and validation arrive in Phase 2+.
"""

__version__ = "1.1.0"
__all__ = ["__version__"]
