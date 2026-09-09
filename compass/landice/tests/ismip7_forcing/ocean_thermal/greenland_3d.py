"""Build regional 3-D Greenland ocean thermal forcing for MALI.

Ported from the standalone ``greenland_thermal_forcing.py`` tool. The workflow
has three coupled stages:

1. Construct seven regional climatological profiles from monthly EN4 objective
   analyses.
2. Translate each regional profile cell-by-cell so it matches processed
   ISMIP7 OCX thermal forcing at the effective local seafloor.
3. Hold gamma0 fixed and calibrate one Jourdain et al. (2020) nonlocal
   temperature correction (deltaT) per region.

The large time-dependent output is streamed one record at a time and written
as NETCDF3_64BIT_OFFSET.

The 3-D-specific parameters are supplied through a JSON config file, while the
mesh, 2-D forcing, output, and diagnostics paths are injected by the compass
step (see ``build_3d_thermal_forcing.py``).
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.optimize import brentq
from scipy.spatial import cKDTree

REGION_NAMES = (
    "ISMIP6 Greenland Central East Shelf",
    "ISMIP6 Greenland Central West Shelf",
    "ISMIP6 Greenland North East Shelf",
    "ISMIP6 Greenland North Shelf",
    "ISMIP6 Greenland North West Shelf",
    "ISMIP6 Greenland South East Shelf",
    "ISMIP6 Greenland South West Shelf",
)

REGION_KEYS = (
    "central_east",
    "central_west",
    "north_east",
    "north",
    "north_west",
    "south_east",
    "south_west",
)

EN4_BIAS_CORRECTIONS = {"g10", "l09", "c13", "c14", "unknown"}
DATE_RE = re.compile(r"(?<!\d)(\d{4})(\d{2})(?!\d)")

MONTH_ABBR = (
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
)


def season_index(month: int, seasons_per_year: int) -> int:
    """Zero-based season for a 1-based calendar month.

    Months are partitioned into ``seasons_per_year`` equal blocks, e.g. with
    four seasons: Jan-Mar=0, Apr-Jun=1, Jul-Sep=2, Oct-Dec=3.
    """
    return (month - 1) // (12 // seasons_per_year)


def season_label(season: int, seasons_per_year: int) -> str:
    months_per_season = 12 // seasons_per_year
    start = season * months_per_season
    end = start + months_per_season - 1
    return f"{MONTH_ABBR[start]}-{MONTH_ABBR[end]}"


def require_xarray():
    """Import xarray late so mathematical unit tests need no NetCDF stack."""
    try:
        import xarray as xr
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "xarray is required. Install xarray, dask, scipy, h5netcdf, and "
            "h5py in the runtime environment."
        ) from exc
    return xr


def _required(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping or mapping[key] in (None, ""):
        raise ValueError(f"Missing required configuration value: {key}")
    return mapping[key]


def _as_path(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


@dataclass(frozen=True)
class Config:
    mesh_file: Path
    region_mask_file: Path
    forcing_2d_file: Path
    en4_directory: Path
    en4_source_region_geojson: Path
    output_file: Path
    melt_params_file: Path
    diagnostics_directory: Path
    scenario: str
    ocean_levels_m: np.ndarray
    source_max_depth_m: float
    profile_start_year: int
    profile_end_year: int
    calibration_start_year: int
    calibration_end_year: int
    en4_version: str
    en4_bias_correction: str
    en4_file_glob: str
    en4_max_mesh_distance_km: float
    en4_latitude_min: float
    en4_latitude_max: float
    gamma0_m_per_yr: float
    regional_melt_targets_m_per_yr: np.ndarray
    rho_ice: float
    rho_seawater: float
    cp_seawater: float
    latent_heat_ice: float
    flotation_tolerance_m: float
    minimum_ice_thickness_m: float
    freezing_a: float
    freezing_b: float
    freezing_c: float
    forcing_variable: str
    overwrite: bool
    output_years_per_file: int
    seasons_per_year: int

    @property
    def calibrate_delta_t(self) -> bool:
        """Whether this run should (re)calibrate the regional deltaT.

        DeltaT is calibrated once against the OCX reanalysis and held fixed
        for every ESM scenario, so calibration only runs when
        ``scenario == "OCX"``; other scenarios reuse the deltaT/gamma0/basin
        stored in ``melt_params_file``.
        """
        return self.scenario.strip().upper() == "OCX"

    @classmethod
    def from_json(cls, path: Path, overrides: dict | None = None) -> "Config":
        """Build a Config from JSON, optionally injecting compass paths.

        ``overrides`` may supply ``mesh_file``, ``forcing_2d_file``,
        ``output_file``, ``melt_params_file``, ``diagnostics_directory``,
        ``scenario``, and ``output_years_per_file``. When supplied, they take
        precedence over the corresponding JSON ``files``/top-level entries,
        which then become optional. The region-mask, EN4, and GeoJSON paths
        always come from the JSON.
        """
        overrides = overrides or {}
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        base = path.resolve().parent

        files = _required(raw, "files")
        en4 = raw.get("en4", {})
        calibration = raw.get("calibration", {})
        physical = raw.get("physical_constants", {})
        freezing = raw.get("freezing_point", {})

        def resolved(field_name, json_key, required=True, default=None):
            if overrides.get(field_name) is not None:
                return Path(overrides[field_name])
            if required:
                return _as_path(_required(files, json_key), base)
            value = files.get(json_key, default)
            return _as_path(value, base) if value else None

        if "ocean_levels_m" in raw:
            levels = np.asarray(raw["ocean_levels_m"], dtype=float)
        else:
            vertical_grid = raw.get("ocean_vertical_grid", {})
            number_of_levels = int(vertical_grid.get("number_of_levels", 30))
            surface_m = float(vertical_grid.get("surface_m", 0.0))
            bottom_m = float(vertical_grid.get("bottom_m", -1000.0))
            if number_of_levels < 2:
                raise ValueError(
                    "ocean_vertical_grid.number_of_levels must be at least 2"
                )
            levels = np.linspace(surface_m, bottom_m, number_of_levels)
        validate_ocean_levels(levels)

        targets_raw = calibration.get("regional_melt_targets_m_per_yr", 20.0)
        if isinstance(targets_raw, dict):
            missing = [key for key in REGION_KEYS if key not in targets_raw]
            if missing:
                raise ValueError(
                    f"Missing regional melt targets for: {', '.join(missing)}"
                )
            targets = np.asarray(
                [targets_raw[key] for key in REGION_KEYS], dtype=float
            )
        elif np.isscalar(targets_raw):
            targets = np.full(len(REGION_NAMES), float(targets_raw))
        else:
            targets = np.asarray(targets_raw, dtype=float)
        if targets.shape != (len(REGION_NAMES),) or np.any(targets < 0.0):
            raise ValueError(
                "Regional melt targets must be seven nonnegative values"
            )

        bias = str(en4.get("bias_correction", "unknown")).lower()
        if bias not in EN4_BIAS_CORRECTIONS:
            raise ValueError(
                f"EN4 bias correction must be one of "
                f"{sorted(EN4_BIAS_CORRECTIONS)}"
            )

        cfg = cls(
            mesh_file=resolved("mesh_file", "mesh"),
            region_mask_file=_as_path(_required(files, "region_masks"), base),
            forcing_2d_file=resolved("forcing_2d_file", "forcing_2d"),
            en4_directory=_as_path(_required(files, "en4_directory"), base),
            en4_source_region_geojson=_as_path(
                _required(files, "en4_source_region_geojson"), base
            ),
            output_file=resolved("output_file", "output"),
            melt_params_file=resolved("melt_params_file", "melt_params"),
            diagnostics_directory=resolved(
                "diagnostics_directory", "diagnostics",
                required=False, default="diagnostics"
            ),
            scenario=str(
                overrides.get("scenario") or raw.get("scenario", "OCX")
            ),
            ocean_levels_m=levels,
            source_max_depth_m=float(raw.get("source_max_depth_m", 1000.0)),
            profile_start_year=int(en4.get("profile_start_year", 2007)),
            profile_end_year=int(en4.get("profile_end_year", 2015)),
            calibration_start_year=int(calibration.get("start_year", 2007)),
            calibration_end_year=int(calibration.get("end_year", 2015)),
            en4_version=str(en4.get("version", "EN.4.2.2")),
            en4_bias_correction=bias,
            en4_file_glob=str(en4.get("file_glob", "**/*.nc")),
            en4_max_mesh_distance_km=float(
                en4.get("max_mesh_distance_km", 300.0)
            ),
            en4_latitude_min=float(en4.get("latitude_min", 55.0)),
            en4_latitude_max=float(en4.get("latitude_max", 90.0)),
            gamma0_m_per_yr=float(calibration.get("gamma0_m_per_yr", 14500.0)),
            regional_melt_targets_m_per_yr=targets,
            rho_ice=float(physical.get("rho_ice", 910.0)),
            rho_seawater=float(physical.get("rho_seawater", 1028.0)),
            cp_seawater=float(physical.get("cp_seawater", 3974.0)),
            latent_heat_ice=float(physical.get("latent_heat_ice", 335000.0)),
            flotation_tolerance_m=float(
                calibration.get("flotation_tolerance_m", 1.0)
            ),
            minimum_ice_thickness_m=float(
                calibration.get("minimum_ice_thickness_m", 0.0)
            ),
            freezing_a=float(freezing.get("a_degC_per_salinity", -0.0575)),
            freezing_b=float(freezing.get("b_degC", 0.0901)),
            freezing_c=float(freezing.get("c_degC_per_m", 7.61e-4)),
            forcing_variable=str(
                raw.get("forcing_2d_variable", "ismip6_2dThermalForcing")
            ),
            overwrite=bool(raw.get("overwrite", False)),
            output_years_per_file=int(
                overrides.get("output_years_per_file") or
                raw.get("output_years_per_file", 10)
            ),
            seasons_per_year=int(
                overrides.get("seasons_per_year") or
                raw.get("seasons_per_year", 4)
            ),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        for label, path in (
            ("mesh", self.mesh_file),
            ("region-mask", self.region_mask_file),
            ("2-D forcing", self.forcing_2d_file),
            ("EN4 directory", self.en4_directory),
            ("EN4 source-region GeoJSON", self.en4_source_region_geojson),
        ):
            if not path.exists():
                raise FileNotFoundError(
                    f"Configured {label} path does not exist: {path}"
                )
        if self.profile_start_year > self.profile_end_year:
            raise ValueError(
                "EN4 profile_start_year must not exceed profile_end_year"
            )
        if self.calibration_start_year > self.calibration_end_year:
            raise ValueError(
                "Calibration start_year must not exceed end_year"
            )
        if self.source_max_depth_m <= 0.0:
            raise ValueError("source_max_depth_m must be positive")
        if self.gamma0_m_per_yr <= 0.0:
            raise ValueError("gamma0_m_per_yr must be positive")
        if self.output_years_per_file < 1:
            raise ValueError("output_years_per_file must be at least 1")
        if self.seasons_per_year < 1 or 12 % self.seasons_per_year != 0:
            raise ValueError(
                "seasons_per_year must be a positive divisor of 12 "
                "(1, 2, 3, 4, 6, or 12)"
            )
        if self.en4_bias_correction == "unknown":
            warnings.warn(
                "EN4 bias correction is unknown. Processing will continue "
                "only if the discovered files contain at most one analysis "
                "per month.",
                stacklevel=2,
            )
        if not self.calibrate_delta_t and not self.melt_params_file.exists():
            raise FileNotFoundError(
                "melt_params_file must already exist when scenario is not "
                f"OCX: {self.melt_params_file}. Build it first with an OCX "
                "(scenario = OCX) run on this mesh."
            )


@dataclass
class MeshData:
    bed: np.ndarray
    thickness: np.ndarray
    area: np.ndarray
    lat_deg: np.ndarray
    lon_deg: np.ndarray


@dataclass
class RegionalProfiles:
    source_z_m: np.ndarray
    output_z_m: np.ndarray
    monthly_dates: list[str]
    monthly_temperature_degC: np.ndarray
    monthly_salinity: np.ndarray
    monthly_thermal_forcing_degC: np.ndarray
    temperature_degC: np.ndarray
    salinity: np.ndarray
    freezing_temperature_degC: np.ndarray
    thermal_forcing_degC: np.ndarray
    output_thermal_forcing_degC: np.ndarray
    seasons_per_year: int
    season_labels: tuple
    season_of_month: np.ndarray
    seasonal_temperature_degC: np.ndarray
    seasonal_salinity: np.ndarray
    seasonal_thermal_forcing_degC: np.ndarray
    seasonal_output_thermal_forcing_degC: np.ndarray
    valid_gridpoint_counts: np.ndarray
    temperature_observation_influence: np.ndarray
    salinity_observation_influence: np.ndarray
    mapped_lats: np.ndarray
    mapped_lons: np.ndarray
    mapped_basins: np.ndarray
    mapped_distances_km: np.ndarray


def validate_ocean_levels(levels: np.ndarray) -> None:
    if levels.ndim != 1 or levels.size < 2:
        raise ValueError("ocean_levels_m must contain at least two values")
    if not np.all(np.isfinite(levels)):
        raise ValueError("ocean_levels_m contains non-finite values")
    if not np.all(np.diff(levels) < 0.0):
        raise ValueError(
            "ocean_levels_m must be strictly decreasing (negative downward)"
        )
    if levels[0] > 0.0 or levels[-1] >= 0.0:
        raise ValueError(
            "ocean levels must begin at or below 0 m and extend below 0 m"
        )


def decode_char_rows(values: np.ndarray) -> list[str]:
    """Decode either raw S1 character rows or xarray-concatenated strings."""
    arr = np.asarray(values)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim == 1 and (arr.dtype.kind == "U" or arr.dtype.itemsize > 1):
        result = []
        for value in arr:
            if isinstance(value, bytes):
                text = value.decode("utf-8", errors="replace")
            else:
                text = str(value)
            result.append(text.rstrip("\x00 "))
        return result
    if arr.ndim == 1:
        arr = arr[None, :]
    result: list[str] = []
    for row in arr:
        if row.dtype.kind == "S":
            text = b"".join(row.tolist()).decode("utf-8", errors="replace")
        elif row.dtype.kind == "U":
            text = "".join(row.tolist())
        else:
            text = bytes(row.tolist()).decode("utf-8", errors="replace")
        result.append(text.rstrip("\x00 "))
    return result


def decode_region_names(values: np.ndarray) -> tuple[str, ...]:
    return tuple(decode_char_rows(values))


def radians_or_degrees_to_degrees(values: np.ndarray, kind: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    limit = math.pi / 2 + 1e-6 if kind == "latitude" else 2 * math.pi + 1e-6
    if np.nanmax(np.abs(values)) <= limit:
        return np.rad2deg(values)
    return values


def build_basin_ids(region_masks: np.ndarray) -> np.ndarray:
    masks = np.asarray(region_masks)
    if masks.ndim != 2 or masks.shape[1] != len(REGION_NAMES):
        raise ValueError(
            f"regionCellMasks must have shape (nCells, {len(REGION_NAMES)}); "
            f"got {masks.shape}"
        )
    membership = masks != 0
    counts = membership.sum(axis=1)
    overlapping = np.flatnonzero(counts > 1)
    unassigned = np.flatnonzero(counts == 0)
    if overlapping.size:
        print(
            f"{overlapping.size} cells belong to multiple regions; first "
            f"indices: {overlapping[:10].tolist()}"
        )
    if unassigned.size:
        print(
            f"{unassigned.size} cells have no region; first indices: "
            f"{unassigned[:10].tolist()}"
        )
    return np.argmax(membership, axis=1).astype(np.int32) + 1


def unit_sphere_xyz(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    lat = np.deg2rad(np.asarray(lat_deg, dtype=float))
    lon = np.deg2rad(np.asarray(lon_deg, dtype=float))
    cos_lat = np.cos(lat)
    return np.column_stack(
        (cos_lat * np.cos(lon), cos_lat * np.sin(lon), np.sin(lat))
    )


def chord_to_arc_km(
    chord: np.ndarray, radius_km: float = 6371.0
) -> np.ndarray:
    return 2.0 * radius_km * np.arcsin(
        np.clip(np.asarray(chord) / 2.0, 0.0, 1.0)
    )


def freezing_temperature(
    salinity: np.ndarray,
    z_m: np.ndarray,
    a: float = -0.0575,
    b: float = 0.0901,
    c: float = 7.61e-4,
) -> np.ndarray:
    return a * np.asarray(salinity) + b + c * np.asarray(z_m)


def interpolate_profile(
    z_source: np.ndarray, values: np.ndarray, z_target: np.ndarray
) -> np.ndarray:
    """Linearly interpolate with constant endpoint extension."""
    z_source = np.asarray(z_source, dtype=float)
    values = np.asarray(values, dtype=float)
    order = np.argsort(z_source)
    good = np.isfinite(z_source[order]) & np.isfinite(values[order])
    if good.sum() < 2:
        raise ValueError(
            "At least two finite source profile levels are required"
        )
    x = z_source[order][good]
    y = values[order][good]
    return np.interp(np.asarray(z_target, dtype=float), x, y, left=y[0],
                     right=y[-1])


def profiles_at_cell_depths(
    regional_profiles: np.ndarray,
    z_levels: np.ndarray,
    basin_ids: np.ndarray,
    cell_depths: np.ndarray,
) -> np.ndarray:
    """Evaluate piecewise-linear regional profiles at cell-specific depths."""
    regional_profiles = np.asarray(regional_profiles, dtype=float)
    basin_ids = np.asarray(basin_ids)
    cell_depths = np.asarray(cell_depths, dtype=float)
    result = np.empty(cell_depths.shape, dtype=float)
    for region in range(len(REGION_NAMES)):
        mask = basin_ids == region + 1
        if np.any(mask):
            result[mask] = interpolate_profile(
                z_levels, regional_profiles[region], cell_depths[mask]
            )
    return result


def floating_mask_and_draft(
    bed: np.ndarray,
    thickness: np.ndarray,
    rho_ice: float,
    rho_seawater: float,
    tolerance_m: float,
    minimum_thickness_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    bed = np.asarray(bed, dtype=float)
    thickness = np.asarray(thickness, dtype=float)
    flotation_thickness = np.where(
        bed < 0.0, -bed * rho_seawater / rho_ice, 0.0
    )
    floating = (
        np.isfinite(bed) &
        np.isfinite(thickness) &
        (thickness > minimum_thickness_m) &
        (thickness <= flotation_thickness + tolerance_m)
    )
    draft = -rho_ice / rho_seawater * thickness
    return floating, draft


def nonlocal_mean_melt(
    delta_t: float,
    monthly_mean_tf: np.ndarray,
    gamma0_m_per_yr: float,
    coefficient: float,
) -> float:
    corrected = np.asarray(monthly_mean_tf, dtype=float) + delta_t
    return float(
        gamma0_m_per_yr * coefficient**2 *
        np.mean(corrected * np.abs(corrected))
    )


def calibrate_delta_t(
    monthly_mean_tf: np.ndarray,
    target_melt_m_per_yr: float,
    gamma0_m_per_yr: float,
    coefficient: float,
) -> float:
    values = np.asarray(monthly_mean_tf, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("No finite monthly regional thermal-forcing means")
    if target_melt_m_per_yr < 0.0:
        raise ValueError("Target mean melt must be nonnegative")

    def residual(delta: float) -> float:
        return (
            nonlocal_mean_melt(delta, values, gamma0_m_per_yr, coefficient) -
            target_melt_m_per_yr
        )

    # The physically relevant root has a nonnegative corrected regional mean
    # in every month. Starting above -min(TF) selects that branch.
    lower = float(-np.min(values))
    if target_melt_m_per_yr == 0.0:
        return lower
    upper = max(lower + 1.0, 1.0)
    while residual(upper) < 0.0:
        upper = lower + 2.0 * (upper - lower)
        if upper - lower > 1000.0:
            raise RuntimeError("Could not bracket deltaT calibration root")

    return float(brentq(residual, lower, upper, xtol=1e-12, rtol=1e-12))


def parse_yyyymm_from_name(path: Path) -> tuple[int, int]:
    matches = DATE_RE.findall(path.name)
    valid = [(int(y), int(m)) for y, m in matches if 1 <= int(m) <= 12]
    if not valid:
        raise ValueError(f"Could not find YYYYMM in EN4 filename: {path.name}")
    return valid[-1]


def discover_en4_files(cfg: Config) -> list[Path]:
    candidates = sorted(cfg.en4_directory.glob(cfg.en4_file_glob))
    selected: dict[tuple[int, int], Path] = {}
    for path in candidates:
        if not path.is_file():
            continue
        name_lower = path.name.lower()
        if (cfg.en4_version.lower() not in name_lower or
                "analysis" not in name_lower):
            continue
        if (cfg.en4_bias_correction != "unknown" and
                f".{cfg.en4_bias_correction}." not in name_lower):
            continue
        try:
            year, month = parse_yyyymm_from_name(path)
        except ValueError:
            continue
        if not (cfg.profile_start_year <= year <= cfg.profile_end_year):
            continue
        key = (year, month)
        if key in selected:
            raise ValueError(
                f"Multiple EN4 analyses found for {year:04d}-{month:02d}: "
                f"{selected[key]} and {path}. Select an explicit bias "
                f"correction."
            )
        selected[key] = path
    if not selected:
        raise FileNotFoundError(
            "No EN4 objective-analysis files matched the configured "
            "directory, glob, years, version, and bias correction"
        )
    return [selected[key] for key in sorted(selected)]


def _array_with_nan(values: Any) -> np.ndarray:
    if np.ma.isMaskedArray(values):
        return np.asarray(values.filled(np.nan), dtype=float)
    return np.asarray(values, dtype=float)


def _find_variable(
    dataset: Any, requested: str, alternatives: Sequence[str] = ()
) -> Any:
    # dataset[name] (not dataset.variables[name]) so callers get a full
    # DataArray; the low-level Variable lacks .name and isel(..., drop=True)
    for name in (requested, *alternatives):
        if name in dataset.variables:
            return dataset[name]
    raise KeyError(
        f"None of these variables is present: {(requested, *alternatives)}"
    )


def _read_3d_en4_variable(
    variable: Any, depth_name: str, lat_name: str, lon_name: str
) -> np.ndarray:
    target_dims = [depth_name, lat_name, lon_name]
    data_array = variable
    for dim in tuple(data_array.dims):
        if dim not in target_dims:
            if data_array.sizes[dim] != 1:
                raise ValueError(
                    f"Unexpected non-singleton EN4 dimension {dim} in "
                    f"{variable.name}"
                )
            data_array = data_array.isel({dim: 0}, drop=True)
    if set(data_array.dims) != set(target_dims):
        raise ValueError(
            f"Cannot arrange {variable.name} dimensions {data_array.dims} as "
            f"{target_dims}"
        )
    return _array_with_nan(data_array.transpose(*target_dims).values)


def _temperature_to_deg_c(values: np.ndarray, units: str) -> np.ndarray:
    normalized = units.strip().lower()
    if (normalized in {"k", "kelvin", "degrees_k", "degree_k"} or
            "kelvin" in normalized):
        return values - 273.15
    if "c" in normalized or normalized == "":
        return values
    raise ValueError(f"Unsupported EN4 temperature units: {units!r}")


def points_in_geojson(
    longitude_deg: np.ndarray,
    latitude_deg: np.ndarray,
    geojson_path: Path,
) -> np.ndarray:
    """Return points inside or on the boundary of a WGS84 GeoJSON geometry."""
    try:
        from shapely import covers, points
        from shapely.geometry import shape
        from shapely.ops import unary_union
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Shapely >=2.0 is required to apply the EN4 source-region GeoJSON"
        ) from exc

    with geojson_path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    kind = document.get("type")
    if kind == "FeatureCollection":
        geometries = [
            shape(feature["geometry"])
            for feature in document.get("features", [])
            if feature.get("geometry") is not None
        ]
        geometry = unary_union(geometries) if geometries else None
    elif kind == "Feature":
        raw_geometry = document.get("geometry")
        geometry = shape(raw_geometry) if raw_geometry is not None else None
    else:
        geometry = shape(document)
    if geometry is None or geometry.is_empty:
        raise ValueError(
            f"GeoJSON contains no usable geometry: {geojson_path}"
        )
    if geometry.geom_type not in {"Polygon", "MultiPolygon"}:
        raise ValueError(
            "EN4 source-region GeoJSON must contain polygonal geometry; got "
            f"{geometry.geom_type}"
        )
    if not geometry.is_valid:
        raise ValueError(
            f"EN4 source-region GeoJSON geometry is invalid: {geojson_path}"
        )

    # GeoJSON uses longitude in the conventional -180..180 range, whereas
    # EN4 longitude coordinates may use either that convention or 0..360.
    longitude = (
        np.asarray(longitude_deg, dtype=float) + 180.0
    ) % 360.0 - 180.0
    latitude = np.asarray(latitude_deg, dtype=float)
    return np.asarray(
        covers(geometry, points(longitude, latitude)), dtype=bool
    )


def load_mesh_and_basins(cfg: Config) -> tuple[MeshData, np.ndarray]:
    xr = require_xarray()
    with xr.open_dataset(
        cfg.mesh_file, decode_times=False, concat_characters=False
    ) as ds:
        bed_var = _find_variable(ds, "bedTopography")
        thk_var = _find_variable(ds, "thickness")
        bed = _array_with_nan(
            bed_var.isel(Time=0).values if "Time" in bed_var.dims
            else bed_var.values
        )
        thickness = _array_with_nan(
            thk_var.isel(Time=0).values if "Time" in thk_var.dims
            else thk_var.values
        )
        area = _array_with_nan(_find_variable(ds, "areaCell").values)
        lat = _array_with_nan(_find_variable(ds, "latCell").values)
        lon = _array_with_nan(_find_variable(ds, "lonCell").values)
    lat_deg = radians_or_degrees_to_degrees(lat, "latitude")
    lon_deg = radians_or_degrees_to_degrees(lon, "longitude")

    with xr.open_dataset(
        cfg.region_mask_file, decode_times=False, concat_characters=False
    ) as ds:
        masks = np.asarray(_find_variable(ds, "regionCellMasks").values)
        names = decode_region_names(_find_variable(ds, "regionNames").values)
    if names != REGION_NAMES:
        details = "\n".join(
            f"  {index + 1}: found={found!r}, expected={expected!r}"
            for index, (found, expected) in enumerate(zip(names, REGION_NAMES))
        )
        raise ValueError(
            f"Region names/order do not match the configured convention:\n"
            f"{details}"
        )
    basin_ids = build_basin_ids(masks)

    n_cells = bed.size
    for name, values in (
        ("thickness", thickness),
        ("areaCell", area),
        ("latCell", lat),
        ("lonCell", lon),
    ):
        if values.size != n_cells:
            raise ValueError(
                f"Mesh variable {name} has {values.size} cells, expected "
                f"{n_cells}"
            )
    return MeshData(bed, thickness, area, lat_deg, lon_deg), basin_ids


def _prepare_en4_mapping(
    dataset: Any,
    mesh: MeshData,
    basin_ids: np.ndarray,
    cfg: Config,
) -> dict[str, np.ndarray]:
    lat_var = _find_variable(dataset, "lat", ("latitude",))
    lon_var = _find_variable(dataset, "lon", ("longitude",))
    source_lat = _array_with_nan(lat_var.values)
    source_lon = _array_with_nan(lon_var.values)
    if source_lat.ndim != 1 or source_lon.ndim != 1:
        raise ValueError(
            "EN4 latitude and longitude coordinates must be one-dimensional"
        )
    lon_grid, lat_grid = np.meshgrid(source_lon, source_lat)
    lat_flat = lat_grid.ravel()
    lon_flat = lon_grid.ravel()
    display_lon_flat = (lon_flat + 180.0) % 360.0 - 180.0
    source_region_ok = points_in_geojson(
        display_lon_flat, lat_flat, cfg.en4_source_region_geojson
    )
    latitude_ok = (
        (lat_flat >= cfg.en4_latitude_min) & (lat_flat <= cfg.en4_latitude_max)
    )
    candidate_flat = np.flatnonzero(latitude_ok & source_region_ok)

    tree = cKDTree(unit_sphere_xyz(mesh.lat_deg, mesh.lon_deg))
    chord, nearest = tree.query(
        unit_sphere_xyz(lat_flat[candidate_flat], lon_flat[candidate_flat]),
        k=1,
    )
    distance_km = chord_to_arc_km(chord)
    keep = distance_km <= cfg.en4_max_mesh_distance_km
    selected_flat = candidate_flat[keep]
    if selected_flat.size == 0:
        raise ValueError(
            "No EN4 grid points inside the source-region GeoJSON fall within "
            "max_mesh_distance_km of the MALI mesh"
        )
    nearest_cells = nearest[keep]
    return {
        "flat_indices": selected_flat,
        "lat": lat_flat[selected_flat],
        "lon": display_lon_flat[selected_flat],
        "basin": basin_ids[nearest_cells],
        "distance_km": distance_km[keep],
        "area_weight": np.clip(
            np.cos(np.deg2rad(lat_flat[selected_flat])), 0.0, None
        ),
        "nlat": np.asarray([source_lat.size]),
        "nlon": np.asarray([source_lon.size]),
    }


def build_regional_profiles(
    cfg: Config, mesh: MeshData, basin_ids: np.ndarray, logger
) -> RegionalProfiles:
    xr = require_xarray()
    files = discover_en4_files(cfg)
    logger.info(
        f"Found {len(files)} EN4 monthly analyses for regional profiles"
    )

    mapping: dict[str, np.ndarray] | None = None
    source_z: np.ndarray | None = None
    monthly_temp: list[np.ndarray] = []
    monthly_salinity: list[np.ndarray] = []
    monthly_tf: list[np.ndarray] = []
    monthly_counts: list[np.ndarray] = []
    monthly_temp_influence: list[np.ndarray] = []
    monthly_sal_influence: list[np.ndarray] = []
    dates: list[str] = []

    for index, path in enumerate(files):
        year, month = parse_yyyymm_from_name(path)
        with xr.open_dataset(
            path,
            decode_times=False,
            mask_and_scale=True,
            concat_characters=False,
        ) as ds:
            depth_var = _find_variable(ds, "depth")
            lat_var = _find_variable(ds, "lat", ("latitude",))
            lon_var = _find_variable(ds, "lon", ("longitude",))
            depth = _array_with_nan(depth_var.values)
            z = -np.abs(depth)
            if source_z is None:
                source_z = z
                mapping = _prepare_en4_mapping(ds, mesh, basin_ids, cfg)
            elif (source_z.shape != z.shape or
                    not np.allclose(source_z, z, equal_nan=True)):
                raise ValueError(f"EN4 depth coordinate changed in {path}")

            assert mapping is not None
            temp_var = _find_variable(ds, "temperature")
            sal_var = _find_variable(ds, "salinity")
            temperature = _read_3d_en4_variable(
                temp_var, depth_var.name, lat_var.name, lon_var.name
            )
            salinity = _read_3d_en4_variable(
                sal_var, depth_var.name, lat_var.name, lon_var.name
            )
            temperature = _temperature_to_deg_c(
                temperature, str(temp_var.attrs.get("units", ""))
            )
            if "temperature_observation_weights" in ds:
                temp_influence = _read_3d_en4_variable(
                    ds["temperature_observation_weights"],
                    depth_var.name,
                    lat_var.name,
                    lon_var.name,
                )
            else:
                temp_influence = np.full_like(temperature, np.nan)
            if "salinity_observation_weights" in ds:
                sal_influence = _read_3d_en4_variable(
                    ds["salinity_observation_weights"],
                    depth_var.name,
                    lat_var.name,
                    lon_var.name,
                )
            else:
                sal_influence = np.full_like(salinity, np.nan)

        n_depth = z.size
        flat_indices = mapping["flat_indices"]
        temp_selected = temperature.reshape(n_depth, -1)[:, flat_indices]
        sal_selected = salinity.reshape(n_depth, -1)[:, flat_indices]
        temp_influence_selected = temp_influence.reshape(
            n_depth, -1
        )[:, flat_indices]
        sal_influence_selected = sal_influence.reshape(
            n_depth, -1
        )[:, flat_indices]
        tf_selected = temp_selected - freezing_temperature(
            sal_selected,
            z[:, None],
            cfg.freezing_a,
            cfg.freezing_b,
            cfg.freezing_c,
        )
        weights = mapping["area_weight"]
        basins = mapping["basin"]

        region_temp = np.full((len(REGION_NAMES), n_depth), np.nan)
        region_sal = np.full_like(region_temp, np.nan)
        region_tf = np.full_like(region_temp, np.nan)
        region_count = np.zeros_like(region_temp, dtype=np.int32)
        region_temp_influence = np.full_like(region_temp, np.nan)
        region_sal_influence = np.full_like(region_temp, np.nan)
        for region in range(len(REGION_NAMES)):
            regional = basins == region + 1
            for level in range(n_depth):
                good = (
                    regional &
                    np.isfinite(temp_selected[level]) &
                    np.isfinite(sal_selected[level]) &
                    np.isfinite(tf_selected[level])
                )
                region_count[region, level] = int(good.sum())
                if np.any(good):
                    w = weights[good]
                    region_temp[region, level] = np.average(
                        temp_selected[level, good], weights=w
                    )
                    region_sal[region, level] = np.average(
                        sal_selected[level, good], weights=w
                    )
                    region_tf[region, level] = np.average(
                        tf_selected[level, good], weights=w
                    )
                temp_good = (
                    regional & np.isfinite(temp_influence_selected[level])
                )
                if np.any(temp_good):
                    region_temp_influence[region, level] = np.average(
                        temp_influence_selected[level, temp_good],
                        weights=weights[temp_good],
                    )
                sal_good = (
                    regional & np.isfinite(sal_influence_selected[level])
                )
                if np.any(sal_good):
                    region_sal_influence[region, level] = np.average(
                        sal_influence_selected[level, sal_good],
                        weights=weights[sal_good],
                    )

        monthly_temp.append(region_temp)
        monthly_salinity.append(region_sal)
        monthly_tf.append(region_tf)
        monthly_counts.append(region_count)
        monthly_temp_influence.append(region_temp_influence)
        monthly_sal_influence.append(region_sal_influence)
        dates.append(f"{year:04d}-{month:02d}")
        if index == 0 or (index + 1) % 12 == 0 or index + 1 == len(files):
            logger.info(
                f"  processed EN4 month {index + 1}/{len(files)}: {dates[-1]}"
            )

    assert source_z is not None and mapping is not None
    month_temp_array = np.asarray(monthly_temp)
    month_sal_array = np.asarray(monthly_salinity)
    month_tf_array = np.asarray(monthly_tf)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_temp = np.nanmean(month_temp_array, axis=0)
        mean_sal = np.nanmean(month_sal_array, axis=0)
        mean_tf = np.nanmean(month_tf_array, axis=0)
    mean_freeze = freezing_temperature(
        mean_sal, source_z[None, :], cfg.freezing_a, cfg.freezing_b,
        cfg.freezing_c
    )
    output_tf = np.vstack(
        [
            interpolate_profile(source_z, mean_tf[region], cfg.ocean_levels_m)
            for region in range(len(REGION_NAMES))
        ]
    )

    # Seasonally-varying vertical structure: group the monthly profiles into
    # seasons_per_year equal calendar blocks and average within each block, so
    # strong upper-ocean seasonality is preserved rather than smeared into one
    # annual mean.
    n_seasons = cfg.seasons_per_year
    n_source = source_z.size
    season_of_month = np.array(
        [season_index(int(date[5:7]), n_seasons) for date in dates]
    )
    seasonal_temp = np.full(
        (n_seasons, len(REGION_NAMES), n_source), np.nan
    )
    seasonal_sal = np.full_like(seasonal_temp, np.nan)
    seasonal_tf = np.full_like(seasonal_temp, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for season in range(n_seasons):
            members = np.flatnonzero(season_of_month == season)
            if members.size == 0:
                raise ValueError(
                    f"No EN4 months fall in season "
                    f"{season_label(season, n_seasons)}; cannot build a "
                    "seasonal profile"
                )
            seasonal_temp[season] = np.nanmean(
                month_temp_array[members], axis=0
            )
            seasonal_sal[season] = np.nanmean(
                month_sal_array[members], axis=0
            )
            seasonal_tf[season] = np.nanmean(
                month_tf_array[members], axis=0
            )
    seasonal_output_tf = np.stack(
        [
            np.vstack(
                [
                    interpolate_profile(
                        source_z, seasonal_tf[season, region],
                        cfg.ocean_levels_m
                    )
                    for region in range(len(REGION_NAMES))
                ]
            )
            for season in range(n_seasons)
        ]
    )
    season_labels = tuple(
        season_label(season, n_seasons) for season in range(n_seasons)
    )

    return RegionalProfiles(
        source_z_m=source_z,
        output_z_m=cfg.ocean_levels_m,
        monthly_dates=dates,
        monthly_temperature_degC=month_temp_array,
        monthly_salinity=month_sal_array,
        monthly_thermal_forcing_degC=month_tf_array,
        temperature_degC=mean_temp,
        salinity=mean_sal,
        freezing_temperature_degC=mean_freeze,
        thermal_forcing_degC=mean_tf,
        output_thermal_forcing_degC=output_tf,
        seasons_per_year=n_seasons,
        season_labels=season_labels,
        season_of_month=season_of_month,
        seasonal_temperature_degC=seasonal_temp,
        seasonal_salinity=seasonal_sal,
        seasonal_thermal_forcing_degC=seasonal_tf,
        seasonal_output_thermal_forcing_degC=seasonal_output_tf,
        valid_gridpoint_counts=np.asarray(monthly_counts),
        temperature_observation_influence=np.asarray(monthly_temp_influence),
        salinity_observation_influence=np.asarray(monthly_sal_influence),
        mapped_lats=mapping["lat"],
        mapped_lons=mapping["lon"],
        mapped_basins=mapping["basin"],
        mapped_distances_km=mapping["distance_km"],
    )


def forcing_times(dataset: Any) -> list[str]:
    if "xtime" not in dataset.variables:
        raise KeyError("2-D forcing file must contain xtime(Time, StrLen)")
    return decode_char_rows(dataset["xtime"].values)


def year_from_xtime(value: str) -> int:
    match = re.match(r"\s*(\d{4})", value)
    if match is None:
        raise ValueError(f"Could not parse year from xtime value {value!r}")
    return int(match.group(1))


def month_from_xtime(value: str) -> int:
    match = re.match(r"\s*\d{4}-(\d{2})", value)
    if match is None:
        raise ValueError(
            f"Could not parse month from xtime value {value!r}"
        )
    return int(match.group(1))


def calibration_time_indices(
    times: Sequence[str], start_year: int, end_year: int
) -> np.ndarray:
    years = np.asarray([year_from_xtime(value) for value in times])
    result = np.flatnonzero((years >= start_year) & (years <= end_year))
    if result.size == 0:
        raise ValueError(
            f"No forcing records fall within calibration years "
            f"{start_year}-{end_year}"
        )
    return result


def validate_forcing_schema(dataset: Any, cfg: Config, n_cells: int) -> Any:
    if cfg.forcing_variable not in dataset.variables:
        raise KeyError(
            f"2-D forcing variable {cfg.forcing_variable!r} is missing"
        )
    variable = dataset[cfg.forcing_variable]
    if variable.ndim != 2 or variable.shape[1] != n_cells:
        raise ValueError(
            f"{cfg.forcing_variable} must have shape "
            f"(Time, nCells={n_cells}); got {variable.shape}"
        )
    return variable


def calibrate_regional_delta_t(
    cfg: Config,
    mesh: MeshData,
    basin_ids: np.ndarray,
    profiles: RegionalProfiles,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xr = require_xarray()
    floating, draft = floating_mask_and_draft(
        mesh.bed,
        mesh.thickness,
        cfg.rho_ice,
        cfg.rho_seawater,
        cfg.flotation_tolerance_m,
        cfg.minimum_ice_thickness_m,
    )
    anchor = np.clip(mesh.bed, -cfg.source_max_depth_m, 0.0)
    # Seasonal base profiles at the effective seafloor (anchor) and the ice
    # draft; the offset uses whichever season each calibration month is in.
    seasonal_out = profiles.seasonal_output_thermal_forcing_degC
    n_seasons = seasonal_out.shape[0]
    base_at_anchor_seasonal = np.stack(
        [
            profiles_at_cell_depths(
                seasonal_out[season], cfg.ocean_levels_m, basin_ids, anchor
            )
            for season in range(n_seasons)
        ]
    )
    base_at_draft_seasonal = np.stack(
        [
            profiles_at_cell_depths(
                seasonal_out[season], cfg.ocean_levels_m, basin_ids, draft
            )
            for season in range(n_seasons)
        ]
    )

    floating_counts = np.asarray(
        [
            np.count_nonzero(floating & (basin_ids == region + 1))
            for region in range(len(REGION_NAMES))
        ]
    )
    empty = np.flatnonzero(floating_counts == 0)
    empty_regions = set(empty.tolist())
    if empty.size:
        names = ", ".join(REGION_KEYS[index] for index in empty)
        print(
            "Cannot calibrate regional deltaT because the initial geometry "
            f"has no floating cells in: {names}. Setting deltaT=0 for these "
            "regions."
        )

    with xr.open_dataset(
        cfg.forcing_2d_file,
        decode_times=False,
        mask_and_scale=True,
        concat_characters=False,
    ) as ds:
        forcing_var = validate_forcing_schema(ds, cfg, mesh.bed.size)
        times = forcing_times(ds)
        indices = calibration_time_indices(
            times, cfg.calibration_start_year, cfg.calibration_end_year
        )
        monthly_means = np.full((indices.size, len(REGION_NAMES)), np.nan)
        for output_index, time_index in enumerate(indices):
            forcing = _array_with_nan(
                forcing_var.isel(Time=int(time_index)).values
            )
            season = season_index(
                month_from_xtime(times[time_index]), n_seasons
            )
            offset = forcing - base_at_anchor_seasonal[season]
            tf_draft = base_at_draft_seasonal[season] + offset
            for region in range(len(REGION_NAMES)):
                if region in empty_regions:
                    continue
                cells = (
                    floating &
                    (basin_ids == region + 1) &
                    np.isfinite(tf_draft)
                )
                if not np.any(cells):
                    raise ValueError(
                        f"No finite draft thermal forcing for "
                        f"{REGION_KEYS[region]} at {times[time_index]}"
                    )
                monthly_means[output_index, region] = np.average(
                    tf_draft[cells], weights=mesh.area[cells]
                )

    coefficient = cfg.rho_seawater * cfg.cp_seawater / (
        cfg.rho_ice * cfg.latent_heat_ice
    )
    delta_t = np.zeros(len(REGION_NAMES))
    achieved = np.full(len(REGION_NAMES), np.nan)
    for region in range(len(REGION_NAMES)):
        if region in empty_regions:
            continue
        delta_t[region] = calibrate_delta_t(
            monthly_means[:, region],
            cfg.regional_melt_targets_m_per_yr[region],
            cfg.gamma0_m_per_yr,
            coefficient,
        )
        achieved[region] = nonlocal_mean_melt(
            delta_t[region], monthly_means[:, region], cfg.gamma0_m_per_yr,
            coefficient
        )
    return delta_t, achieved, monthly_means


def write_melt_params(
    cfg: Config,
    basin_ids: np.ndarray,
    regional_delta_t: np.ndarray,
    logger,
) -> None:
    """Write the calibrated deltaT/gamma0/basin fields.

    Kept in a file separate from the time-varying 3-D thermal forcing so
    every ESM scenario can reuse the same OCX-calibrated values unchanged
    (see ``Config.calibrate_delta_t``).
    """
    xr = require_xarray()
    output = cfg.melt_params_file
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not cfg.overwrite:
        logger.info(
            f"Melt-parameters file already exists; skipping (set "
            f"overwrite=true to regenerate): {output}"
        )
        return
    temporary = output.with_name(output.name + ".partial")
    if temporary.exists():
        raise FileExistsError(
            f"Partial melt-parameters output already exists: {temporary}. "
            "Remove or rename it after inspecting it."
        )
    delta_t_by_cell = regional_delta_t[basin_ids - 1]
    target = xr.Dataset(
        data_vars={
            "ismip6shelfMelt_basin": xr.DataArray(
                basin_ids.astype(np.int32),
                dims=("nCells",),
                attrs={
                    "description": "One-based basin number for regional "
                    "ISMIP6 shelf-melt forcing"
                },
            ),
            "ismip6shelfMelt_gamma0": xr.DataArray(
                np.float32(cfg.gamma0_m_per_yr),
                attrs={
                    "units": "m yr^-1",
                    "description": "Uniform gamma0 for nonlocal Jourdain "
                    "et al. (2020) shelf melt",
                },
            ),
            "ismip6shelfMelt_deltaT": xr.DataArray(
                delta_t_by_cell.astype(np.float32),
                dims=("nCells",),
                attrs={
                    "units": "K",
                    "description": "Regionally calibrated, cellwise "
                    "temperature-bias correction, calibrated once against "
                    "OCX and held fixed for every ESM",
                },
            ),
        },
        attrs={
            "title": "ISMIP6 shelf-melt parameters (deltaT, gamma0, basin) "
            "for Greenland, calibrated against OCX",
            "region_names": " | ".join(REGION_NAMES),
            "regional_deltaT_K": ", ".join(
                f"{value:.8g}" for value in regional_delta_t
            ),
            "deltaT_calibration_period": f"{cfg.calibration_start_year}-"
            f"{cfg.calibration_end_year}",
            "history": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ) + ": " + " ".join(sys.argv),
        },
    )
    encoding = {
        "ismip6shelfMelt_basin": {"dtype": "int32", "_FillValue": None},
        "ismip6shelfMelt_gamma0": {"dtype": "float32", "_FillValue": None},
        "ismip6shelfMelt_deltaT": {"dtype": "float32", "_FillValue": None},
    }
    try:
        logger.info(f"Writing calibrated melt parameters to {temporary}")
        target.to_netcdf(
            temporary, engine="netcdf4", format="NETCDF3_64BIT",
            encoding=encoding,
        )
        target.close()
        os.replace(temporary, output)
        logger.info(f"Created {output}")
    except Exception:
        logger.warning(
            f"Melt-parameters output was not finalized; partial file, if "
            f"any, is at {temporary}"
        )
        raise


def read_melt_params(
    cfg: Config,
    basin_ids: np.ndarray,
    logger,
) -> np.ndarray:
    """Load the OCX-calibrated per-region deltaT for reuse by other ESMs.

    Validates that the stored basin assignment and gamma0 match the current
    mesh/config, since deltaT is only meaningful alongside the exact basin
    layout and gamma0 it was calibrated with.
    """
    xr = require_xarray()
    path = cfg.melt_params_file
    with xr.open_dataset(
        path, decode_times=False, mask_and_scale=False
    ) as ds:
        stored_basin = ds["ismip6shelfMelt_basin"].values
        if stored_basin.shape != basin_ids.shape or not np.array_equal(
            stored_basin, basin_ids
        ):
            raise ValueError(
                f"Basin assignment in {path} does not match the current "
                "mesh; rebuild it with an OCX (scenario = OCX) run on this "
                "mesh before reuse."
            )
        gamma0 = float(ds["ismip6shelfMelt_gamma0"].values)
        if not math.isclose(gamma0, cfg.gamma0_m_per_yr, rel_tol=1e-6):
            raise ValueError(
                f"gamma0 in {path} ({gamma0} m/yr) does not match the "
                f"configured gamma0_m_per_yr ({cfg.gamma0_m_per_yr} m/yr)"
            )
        delta_t_by_cell = ds["ismip6shelfMelt_deltaT"].values.astype(float)
    regional_delta_t = np.zeros(len(REGION_NAMES))
    for region in range(len(REGION_NAMES)):
        cells = basin_ids == region + 1
        if np.any(cells):
            regional_delta_t[region] = delta_t_by_cell[cells][0]
    logger.info(f"Loaded calibrated melt parameters from {path}")
    return regional_delta_t


def year_chunks(
    years: np.ndarray, years_per_file: int
) -> "list[tuple[np.ndarray, int, int]]":
    """Group record indices into consecutive blocks of whole years.

    Blocks are aligned to the first year so their boundaries match a MALI
    ``filename_interval`` anchored at ``first_year``. Returns
    ``(record_indices, block_start_year, block_last_year)`` tuples.
    """
    years = np.asarray(years)
    first_year = int(years.min())
    block_id = (years - first_year) // years_per_file
    chunks = []
    for block in np.unique(block_id):
        indices = np.flatnonzero(block_id == block)
        start_year = first_year + int(block) * years_per_file
        chunks.append((indices, start_year, int(years[indices].max())))
    return chunks


def chunk_output_path(output_file: Path, start_year: int) -> Path:
    """Name a per-chunk file by its block start year (MALI ``$Y`` template).

    Any trailing ``_YYYY-YYYY`` range in ``output_file`` is replaced with the
    single start year so the run side can address the whole series with one
    ``filename_template`` plus a ``filename_interval``.
    """
    stem = re.sub(r"_\d{4}-\d{4}$", "", output_file.stem)
    name = f"{stem}_{start_year:04d}{output_file.suffix}"
    return output_file.with_name(name)


def _convert_to_cdf2(hdf5_path: Path, cdf2_path: Path, logger) -> None:
    """Convert an HDF5 (NETCDF4) file to CDF-2 (64-bit offset) with nccopy.

    MALI/pnetcdf reads classic formats only. Writing HDF5 then converting is
    much faster than writing CDF-2 directly for large record variables, and
    nccopy preserves the unlimited Time dimension MALI expects.
    """
    nccopy = shutil.which("nccopy")
    if nccopy is None:
        raise RuntimeError(
            "nccopy is required to convert the 3-D forcing to CDF-2 but was "
            "not found on PATH."
        )
    logger.info(f"Converting {hdf5_path.name} to CDF-2 (64-bit offset)")
    subprocess.run(
        [nccopy, "-k", "nc6", str(hdf5_path), str(cdf2_path)], check=True
    )


def _write_forcing_chunk(
    cfg: Config,
    forcing_3d: Any,
    chunk_times: "list[str]",
    regional_delta_t: np.ndarray,
    start_year: int,
    last_year: int,
    output_path: Path,
    logger,
    dask,
) -> None:
    xr = require_xarray()
    if output_path.exists() and not cfg.overwrite:
        logger.info(
            f"Forcing chunk already exists; skipping (set overwrite=true to "
            f"regenerate): {output_path}"
        )
        return
    temporary = output_path.with_name(output_path.name + ".partial")
    if temporary.exists():
        raise FileExistsError(
            f"Partial output already exists: {temporary}. Remove or rename "
            "it after inspecting it."
        )
    hdf5_temporary = output_path.with_name(output_path.name + ".h5.partial")
    target = xr.Dataset(
        data_vars={
            "xtime": xr.DataArray(
                np.asarray(
                    [value.ljust(64) for value in chunk_times], dtype="S"
                ),
                dims=("Time",),
            ),
            "ismip6shelfMelt_zOcean": xr.DataArray(
                cfg.ocean_levels_m.astype(np.float32),
                dims=("nISMIP6OceanLayers",),
                attrs={"units": "m", "positive": "up"},
            ),
            "ismip6shelfMelt_3dThermalForcing": forcing_3d,
        },
        attrs={
            "title": "Regional three-dimensional Greenland ocean thermal "
            "forcing for MALI",
            "source_2d_forcing": str(cfg.forcing_2d_file),
            "seasonal_vertical_structure": f"{cfg.seasons_per_year} seasons",
            "melt_params_file": str(cfg.melt_params_file),
            "source_en4_version": cfg.en4_version,
            "source_en4_bias_correction": cfg.en4_bias_correction,
            "source_en4_region_geojson": str(
                cfg.en4_source_region_geojson
            ),
            "en4_profile_period": f"{cfg.profile_start_year}-"
            f"{cfg.profile_end_year}",
            "deltaT_calibration_period": f"{cfg.calibration_start_year}"
            f"-{cfg.calibration_end_year}",
            "forcing_period": f"{start_year}-{last_year}",
            "source_ocean_max_depth_m": cfg.source_max_depth_m,
            "region_names": " | ".join(REGION_NAMES),
            "regional_deltaT_K": ", ".join(
                f"{value:.8g}" for value in regional_delta_t
            ),
            "history": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ) + ": " + " ".join(sys.argv),
        },
    )
    # Dimension coordinates are implementation details, not MALI input
    # fields. Drop them while retaining the named dimensions.
    target = target.drop_vars(
        [
            name
            for name in ("nCells", "nISMIP6OceanLayers")
            if name in target.coords
        ]
    )
    encoding = {
        "xtime": {"char_dim_name": "StrLen"},
        "ismip6shelfMelt_zOcean": {"dtype": "float32", "_FillValue": None},
        "ismip6shelfMelt_3dThermalForcing": {
            "dtype": "float32", "_FillValue": None
        },
    }
    try:
        logger.info(
            f"Writing {len(chunk_times)} monthly records "
            f"({start_year}-{last_year}) to {output_path.name}"
        )
        # Writing NETCDF3_64BIT directly with an unlimited Time dimension and
        # multiple record variables is several times slower because the
        # classic writer interleaves each record and writes with a stride.
        # Write HDF5 first (each variable stored contiguously, much faster),
        # then convert to the CDF-2 format MALI/pnetcdf requires with nccopy.
        with dask.config.set(scheduler="single-threaded"):
            target.to_netcdf(
                hdf5_temporary,
                engine="netcdf4",
                format="NETCDF4",
                unlimited_dims=["Time"],
                encoding=encoding,
            )
        target.close()
        _convert_to_cdf2(hdf5_temporary, temporary, logger)
        os.replace(temporary, output_path)
        logger.info(f"Created {output_path}")
    except Exception:
        logger.warning(
            f"Output was not finalized; partial files, if any, are at "
            f"{hdf5_temporary} and {temporary}"
        )
        raise
    finally:
        if hdf5_temporary.exists():
            hdf5_temporary.unlink()


def write_output(
    cfg: Config,
    mesh: MeshData,
    basin_ids: np.ndarray,
    profiles: RegionalProfiles,
    regional_delta_t: np.ndarray,
    logger,
) -> None:
    xr = require_xarray()
    try:
        import dask
        import dask.array as darray
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Dask is required to construct and stream the multi-gigabyte "
            "xarray output"
        ) from exc
    cfg.output_file.parent.mkdir(parents=True, exist_ok=True)

    anchor = np.clip(mesh.bed, -cfg.source_max_depth_m, 0.0)
    # Per-season vertical structure of the column relative to the anchor. The
    # 2-D forcing sets the value at the anchor each month, so this delta plus
    # the 2-D forcing reconstructs the full column; the season used varies
    # with each output month.
    seasonal_out = profiles.seasonal_output_thermal_forcing_degC
    n_seasons = seasonal_out.shape[0]
    base_by_cell_seasonal = seasonal_out[:, basin_ids - 1, :]
    base_at_anchor_seasonal = np.stack(
        [
            profiles_at_cell_depths(
                seasonal_out[season], cfg.ocean_levels_m, basin_ids, anchor
            )
            for season in range(n_seasons)
        ]
    )
    delta_seasonal = (
        base_by_cell_seasonal - base_at_anchor_seasonal[:, :, None]
    ).astype(np.float32)

    with xr.open_dataset(
        cfg.forcing_2d_file,
        decode_times=False,
        mask_and_scale=True,
        concat_characters=False,
        chunks={"Time": 1},
    ) as source:
        forcing_var = validate_forcing_schema(source, cfg, mesh.bed.size)
        times = forcing_times(source)
        years = np.asarray([year_from_xtime(value) for value in times])
        season_of_time = np.asarray(
            [season_index(month_from_xtime(t), n_seasons) for t in times]
        )
        # Fail before initiating a multi-gigabyte write if any source value
        # is absent. This reduction stays lazy until compute() and does not
        # load the full forcing array into memory.
        invalid_count = int(
            (~np.isfinite(forcing_var)).sum().compute().item()
        )
        if invalid_count:
            raise ValueError(
                f"2-D forcing contains {invalid_count} invalid values; "
                "explicit missing-value handling is required before "
                "building MALI forcing"
            )

        delta_by_season = xr.DataArray(
            darray.from_array(
                delta_seasonal,
                chunks=(1, mesh.bed.size, cfg.ocean_levels_m.size),
            ),
            dims=("season", "nCells", "nISMIP6OceanLayers"),
        )
        # Gather each month's seasonal column shape; stays lazy so the write
        # streams record by record.
        delta_by_time = delta_by_season.isel(
            season=xr.DataArray(season_of_time, dims="Time")
        )
        forcing_3d = (
            forcing_var.astype(np.float32) + delta_by_time
        ).transpose("Time", "nCells", "nISMIP6OceanLayers").assign_attrs(
            units="C",
            long_name="3D thermal forcing for nonlocal ISMIP6 ice-shelf "
            "melt method",
        )

        chunks = year_chunks(years, cfg.output_years_per_file)
        logger.info(
            f"Writing {len(times)} monthly records to {len(chunks)} file(s) "
            f"of up to {cfg.output_years_per_file} year(s) each"
        )
        for indices, start_year, last_year in chunks:
            output_path = chunk_output_path(cfg.output_file, start_year)
            _write_forcing_chunk(
                cfg,
                forcing_3d.isel(Time=indices),
                [times[int(index)] for index in indices],
                regional_delta_t,
                start_year,
                last_year,
                output_path,
                logger,
                dask,
            )


def write_diagnostics(
    cfg: Config,
    mesh: MeshData,
    basin_ids: np.ndarray,
    profiles: RegionalProfiles,
    regional_delta_t: np.ndarray,
    achieved_melt: np.ndarray | None,
    calibration_monthly_tf: np.ndarray | None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = cfg.diagnostics_directory
    directory.mkdir(parents=True, exist_ok=True)

    with (directory / "regional_profiles.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "season",
                "region_id",
                "region_key",
                "region_name",
                "z_m",
                "temperature_degC",
                "salinity",
                "freezing_temperature_degC",
                "thermal_forcing_degC",
            ]
        )
        for season in range(profiles.seasons_per_year):
            season_freeze = freezing_temperature(
                profiles.seasonal_salinity[season],
                profiles.source_z_m[None, :],
                cfg.freezing_a, cfg.freezing_b, cfg.freezing_c,
            )
            for region in range(len(REGION_NAMES)):
                for level, z in enumerate(profiles.source_z_m):
                    writer.writerow(
                        [
                            profiles.season_labels[season],
                            region + 1,
                            REGION_KEYS[region],
                            REGION_NAMES[region],
                            float(z),
                            float(
                                profiles.seasonal_temperature_degC[
                                    season, region, level
                                ]
                            ),
                            float(
                                profiles.seasonal_salinity[
                                    season, region, level
                                ]
                            ),
                            float(season_freeze[region, level]),
                            float(
                                profiles.seasonal_thermal_forcing_degC[
                                    season, region, level
                                ]
                            ),
                        ]
                    )

    with (directory / "en4_coverage.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "date",
                "region_id",
                "region_key",
                "z_m",
                "valid_gridpoint_count",
                "temperature_observation_influence",
                "salinity_observation_influence",
            ]
        )
        for month, date in enumerate(profiles.monthly_dates):
            for region in range(len(REGION_NAMES)):
                for level, z in enumerate(profiles.source_z_m):
                    writer.writerow(
                        [
                            date,
                            region + 1,
                            REGION_KEYS[region],
                            float(z),
                            int(
                                profiles.valid_gridpoint_counts[
                                    month, region, level
                                ]
                            ),
                            float(
                                profiles.temperature_observation_influence[
                                    month, region, level
                                ]
                            ),
                            float(
                                profiles.salinity_observation_influence[
                                    month, region, level
                                ]
                            ),
                        ]
                    )

    floating, _ = floating_mask_and_draft(
        mesh.bed,
        mesh.thickness,
        cfg.rho_ice,
        cfg.rho_seawater,
        cfg.flotation_tolerance_m,
        cfg.minimum_ice_thickness_m,
    )
    floating_counts = [
        int(np.count_nonzero(floating & (basin_ids == region + 1)))
        for region in range(len(REGION_NAMES))
    ]
    floating_areas_km2 = [
        float(np.sum(mesh.area[floating & (basin_ids == region + 1)]) / 1.0e6)
        for region in range(len(REGION_NAMES))
    ]

    # Only meaningful when this run actually calibrated deltaT (scenario ==
    # OCX); other scenarios reuse deltaT from an existing melt_params_file
    # and have no achieved-melt/calibration-TF statistics to report.
    if achieved_melt is not None and calibration_monthly_tf is not None:
        calibration_summary = {
            "gamma0_m_per_yr": cfg.gamma0_m_per_yr,
            "calibration_period": [
                cfg.calibration_start_year,
                cfg.calibration_end_year,
            ],
            "profile_period": [cfg.profile_start_year, cfg.profile_end_year],
            "regions": [
                {
                    "id": region + 1,
                    "key": REGION_KEYS[region],
                    "name": REGION_NAMES[region],
                    "target_melt_m_per_yr": float(
                        cfg.regional_melt_targets_m_per_yr[region]
                    ),
                    "achieved_melt_m_per_yr": float(achieved_melt[region]),
                    "deltaT_K": float(regional_delta_t[region]),
                    "mean_calibration_TF_degC": float(
                        np.mean(calibration_monthly_tf[:, region])
                    ),
                    "minimum_calibration_TF_degC": float(
                        np.min(calibration_monthly_tf[:, region])
                    ),
                    "maximum_calibration_TF_degC": float(
                        np.max(calibration_monthly_tf[:, region])
                    ),
                    "floating_cell_count": floating_counts[region],
                    "floating_area_km2": floating_areas_km2[region],
                }
                for region in range(len(REGION_NAMES))
            ],
        }
        with (directory / "deltaT_calibration.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(calibration_summary, handle, indent=2)
            handle.write("\n")

    colors = plt.get_cmap("tab10")(np.arange(len(REGION_NAMES)))
    fig, ax = plt.subplots(figsize=(8, 8))
    mesh_stride = max(1, mesh.lat_deg.size // 100_000)
    ax.scatter(
        mesh.lon_deg[::mesh_stride],
        mesh.lat_deg[::mesh_stride],
        c=colors[basin_ids[::mesh_stride] - 1],
        s=0.15,
        alpha=0.2,
        linewidths=0,
    )
    ax.scatter(
        profiles.mapped_lons,
        profiles.mapped_lats,
        c=colors[profiles.mapped_basins - 1],
        s=5,
        edgecolors="none",
    )
    ax.set_xlabel("Longitude (degrees east)")
    ax.set_ylabel("Latitude (degrees north)")
    ax.set_title("EN4 grid points assigned to Greenland shelf regions")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(directory / "en4_region_assignment.png", dpi=180)
    plt.close(fig)

    # Validate the profile translation at the effective seafloor and map one
    # representative monthly field at all output depths.
    xr = require_xarray()
    with xr.open_dataset(
        cfg.forcing_2d_file,
        decode_times=False,
        mask_and_scale=True,
        concat_characters=False,
    ) as source:
        times = forcing_times(source)
        indices = calibration_time_indices(
            times, cfg.calibration_start_year, cfg.calibration_end_year
        )
        time_index = int(indices[0])
        forcing_2d = _array_with_nan(
            validate_forcing_schema(source, cfg, mesh.bed.size)
            .isel(Time=time_index)
            .values
        )
    rep_season = season_index(
        month_from_xtime(times[time_index]), profiles.seasons_per_year
    )
    seasonal_out = profiles.seasonal_output_thermal_forcing_degC[rep_season]
    anchor = np.clip(mesh.bed, -cfg.source_max_depth_m, 0.0)
    base_at_anchor = profiles_at_cell_depths(
        seasonal_out,
        cfg.ocean_levels_m,
        basin_ids,
        anchor,
    )
    base_by_cell = seasonal_out[basin_ids - 1]
    offset = forcing_2d - base_at_anchor
    forcing_3d = base_by_cell + offset[:, None]
    reconstructed_anchor = base_at_anchor + offset
    max_anchor_error = float(
        np.nanmax(np.abs(reconstructed_anchor - forcing_2d))
    )
    with (directory / "forcing_validation.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "representative_time": times[time_index],
                "representative_season": profiles.season_labels[rep_season],
                "maximum_absolute_anchor_error_degC": max_anchor_error,
                "effective_anchor_depth_range_m": [
                    float(np.nanmin(anchor)),
                    float(np.nanmax(anchor)),
                ],
                "source_max_depth_m": cfg.source_max_depth_m,
            },
            handle,
            indent=2,
        )
        handle.write("\n")

    n_levels = cfg.ocean_levels_m.size
    plotted_levels = np.unique(
        np.rint(np.linspace(0, n_levels - 1, min(n_levels, 6))).astype(int)
    )
    ncols = min(2, plotted_levels.size)
    nrows = int(math.ceil(plotted_levels.size / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(7 * ncols, 6 * nrows), squeeze=False
    )
    plot_stride = max(1, mesh.lat_deg.size // 150_000)
    color_limits = np.nanpercentile(forcing_3d[::plot_stride], [2.0, 98.0])
    for plot_index, ax in enumerate(axes.ravel()):
        if plot_index >= plotted_levels.size:
            ax.set_visible(False)
            continue
        level = int(plotted_levels[plot_index])
        scatter = ax.scatter(
            mesh.lon_deg[::plot_stride],
            mesh.lat_deg[::plot_stride],
            c=forcing_3d[::plot_stride, level],
            s=0.5,
            linewidths=0,
            cmap="coolwarm",
            vmin=color_limits[0],
            vmax=color_limits[1],
        )
        ax.set_title(f"z = {cfg.ocean_levels_m[level]:g} m")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(scatter, ax=ax, label="Thermal forcing (°C)")
    fig.suptitle(
        f"Translated 3-D forcing: {times[time_index]} "
        f"(season {profiles.season_labels[rep_season]})"
    )
    fig.tight_layout()
    fig.savefig(
        directory / "thermal_forcing_at_representative_ocean_levels.png",
        dpi=180,
    )
    plt.close(fig)

    season_colors = plt.get_cmap("turbo")(
        np.linspace(0.05, 0.95, profiles.seasons_per_year)
    )
    for region in range(len(REGION_NAMES)):
        fig, axes = plt.subplots(1, 2, figsize=(10, 7), sharey=True)
        # Faint monthly profiles, colored by season, behind the seasonal
        # means so the seasonality that motivates the seasonal structure is
        # visible.
        for entry, monthly in enumerate(
            profiles.monthly_temperature_degC[:, region, :]
        ):
            axes[0].plot(
                monthly, profiles.source_z_m,
                color=season_colors[profiles.season_of_month[entry]],
                alpha=0.06, linewidth=0.5
            )
        for season in range(profiles.seasons_per_year):
            axes[0].plot(
                profiles.seasonal_temperature_degC[season, region],
                profiles.source_z_m,
                color=season_colors[season],
                linewidth=2,
                label=profiles.season_labels[season],
            )
            selected_temp = interpolate_profile(
                profiles.source_z_m,
                profiles.seasonal_temperature_degC[season, region],
                profiles.output_z_m,
            )
            axes[0].scatter(
                selected_temp, profiles.output_z_m,
                color=season_colors[season], marker="*", zorder=3,
            )
        axes[0].set_xlabel("Potential temperature (°C)")
        axes[0].set_ylabel("Elevation (m)")
        axes[0].legend(loc="best", fontsize=8, title="season / MALI levels")

        for entry, monthly in enumerate(
            profiles.monthly_thermal_forcing_degC[:, region, :]
        ):
            axes[1].plot(
                monthly, profiles.source_z_m,
                color=season_colors[profiles.season_of_month[entry]],
                alpha=0.06, linewidth=0.5
            )
        for season in range(profiles.seasons_per_year):
            axes[1].plot(
                profiles.seasonal_thermal_forcing_degC[season, region],
                profiles.source_z_m,
                color=season_colors[season],
                linewidth=2,
            )
            axes[1].scatter(
                profiles.seasonal_output_thermal_forcing_degC[
                    season, region
                ],
                profiles.output_z_m,
                color=season_colors[season],
                marker="*",
                zorder=3,
            )
        axes[1].set_xlabel("Thermal forcing (°C)")
        for ax in axes:
            ax.grid(alpha=0.3)
            ax.set_ylim(
                min(-cfg.source_max_depth_m, profiles.output_z_m[-1]), 0.0
            )
        fig.suptitle(REGION_NAMES[region])
        fig.tight_layout()
        fig.savefig(
            directory / f"profile_{region + 1:02d}_{REGION_KEYS[region]}.png",
            dpi=180,
        )
        plt.close(fig)


def print_summary(
    cfg: Config,
    regional_delta_t: np.ndarray,
    achieved_melt: np.ndarray | None,
    logger,
) -> None:
    logger.info("Regional deltaT calibration")
    if achieved_melt is None:
        logger.info("region          deltaT_K")
        for region, key in enumerate(REGION_KEYS):
            logger.info(f"{key:15s} {regional_delta_t[region]:9.5f}")
        return
    logger.info("region          target_m/yr  achieved_m/yr  deltaT_K")
    for region, key in enumerate(REGION_KEYS):
        logger.info(
            f"{key:15s} {cfg.regional_melt_targets_m_per_yr[region]:11.5f} "
            f"{achieved_melt[region]:14.5f} {regional_delta_t[region]:9.5f}"
        )


# A near-floor TF gradient above this magnitude means clamping the anchor to
# source_max_depth_m biases a sloped column rather than a flat one.
_BOTTOM_GRADIENT_THRESHOLD_C_PER_M = 5.0e-4


def warn_if_seafloor_below_max_depth(
    cfg: Config,
    mesh: MeshData,
    basin_ids: np.ndarray,
    profiles: RegionalProfiles,
    logger,
) -> None:
    """Warn when marine ice sits below the clamped anchor depth.

    Where the seafloor is deeper than ``source_max_depth_m`` the anchor is
    clamped, so TF_3d matches TF_2d at ``source_max_depth_m`` rather than the
    true seafloor. If the regional profile still has a vertical gradient at
    that depth, the per-cell offset (and thus the whole column) is biased.
    """
    max_depth = cfg.source_max_depth_m
    marine_ice = (
        (mesh.thickness > cfg.minimum_ice_thickness_m) & (mesh.bed < 0.0)
    )
    deep = marine_ice & (mesh.bed < -max_depth)
    n_deep = int(np.count_nonzero(deep))
    if n_deep == 0:
        return
    marine_area = float(np.sum(mesh.area[marine_ice]))
    deep_area = float(np.sum(mesh.area[deep]))
    fraction = deep_area / marine_area if marine_area > 0.0 else 0.0
    deepest = float(-np.min(mesh.bed[deep]))

    # Near-floor gradient (degC/m) from the deepest two output levels tells
    # us which affected regions still slope (biased) versus are flat (benign).
    z = cfg.ocean_levels_m
    tf = profiles.output_thermal_forcing_degC
    dz = z[-1] - z[-2]
    gradients = (tf[:, -1] - tf[:, -2]) / dz
    sloped_regions = sorted({
        REGION_KEYS[int(region) - 1]
        for region in np.unique(basin_ids[deep])
        if abs(gradients[int(region) - 1]) >=
        _BOTTOM_GRADIENT_THRESHOLD_C_PER_M
    })

    message = (
        f"{n_deep} marine ice cells ({100.0 * fraction:.1f}% of marine-ice "
        f"area; deepest {deepest:.0f} m) have seafloor below "
        f"source_max_depth_m={max_depth:g} m, so their anchor is clamped to "
        f"{max_depth:g} m and TF_3d matches TF_2d there rather than at the "
        "true seafloor."
    )
    if sloped_regions:
        message += (
            " The regional profile still slopes at that depth in: "
            f"{', '.join(sloped_regions)}; the clamp biases the whole "
            "reconstructed column for those cells. Consider increasing "
            "ocean_vertical_grid.bottom_m and source_max_depth_m together to "
            "cover the deepest seafloor."
        )
    else:
        message += (
            " The regional profile is effectively flat at that depth, so the "
            "clamp is benign here."
        )
    logger.warning(message)


def run(cfg: Config, logger, prepare_only: bool = False) -> None:
    mesh, basin_ids = load_mesh_and_basins(cfg)
    profiles = build_regional_profiles(cfg, mesh, basin_ids, logger)
    warn_if_seafloor_below_max_depth(cfg, mesh, basin_ids, profiles, logger)
    if cfg.calibrate_delta_t:
        delta_t, achieved, calibration_monthly_tf = (
            calibrate_regional_delta_t(cfg, mesh, basin_ids, profiles)
        )
        write_melt_params(cfg, basin_ids, delta_t, logger)
    else:
        logger.info(
            f"scenario={cfg.scenario!r} is not OCX; reusing calibrated "
            f"deltaT/gamma0 from {cfg.melt_params_file} instead of "
            "recalibrating"
        )
        delta_t = read_melt_params(cfg, basin_ids, logger)
        achieved, calibration_monthly_tf = None, None
    print_summary(cfg, delta_t, achieved, logger)
    write_diagnostics(
        cfg, mesh, basin_ids, profiles, delta_t, achieved,
        calibration_monthly_tf
    )
    if not prepare_only:
        write_output(cfg, mesh, basin_ids, profiles, delta_t, logger)
    else:
        logger.info(
            "Preparation-only run complete; the multi-gigabyte forcing file "
            "was not written"
        )
    logger.info(f"Diagnostics: {cfg.diagnostics_directory}")
