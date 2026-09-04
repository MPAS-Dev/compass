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
    diagnostics_directory: Path
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

    @classmethod
    def from_json(cls, path: Path, overrides: dict | None = None) -> "Config":
        """Build a Config from JSON, optionally injecting compass paths.

        ``overrides`` may supply ``mesh_file``, ``forcing_2d_file``,
        ``output_file``, and ``diagnostics_directory`` (as ``Path`` objects).
        When supplied, they take precedence over the corresponding JSON
        ``files`` entries, which then become optional. The region-mask, EN4,
        and GeoJSON paths always come from the JSON.
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
            diagnostics_directory=resolved(
                "diagnostics_directory", "diagnostics",
                required=False, default="diagnostics"
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
        if self.en4_bias_correction == "unknown":
            warnings.warn(
                "EN4 bias correction is unknown. Processing will continue "
                "only if the discovered files contain at most one analysis "
                "per month.",
                stacklevel=2,
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
        raise ValueError(
            f"{overlapping.size} cells belong to multiple regions; first "
            f"indices: {overlapping[:10].tolist()}"
        )
    if unassigned.size:
        raise ValueError(
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
    for name in (requested, *alternatives):
        if name in dataset.variables:
            return dataset.variables[name]
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
    base_at_anchor = profiles_at_cell_depths(
        profiles.output_thermal_forcing_degC, cfg.ocean_levels_m, basin_ids,
        anchor
    )
    base_at_draft = profiles_at_cell_depths(
        profiles.output_thermal_forcing_degC, cfg.ocean_levels_m, basin_ids,
        draft
    )

    floating_counts = np.asarray(
        [
            np.count_nonzero(floating & (basin_ids == region + 1))
            for region in range(len(REGION_NAMES))
        ]
    )
    empty = np.flatnonzero(floating_counts == 0)
    if empty.size:
        names = ", ".join(REGION_KEYS[index] for index in empty)
        raise ValueError(
            "Cannot calibrate regional deltaT because the initial geometry "
            f"has no floating cells in: {names}"
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
            offset = forcing - base_at_anchor
            tf_draft = base_at_draft + offset
            for region in range(len(REGION_NAMES)):
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
    delta_t = np.asarray(
        [
            calibrate_delta_t(
                monthly_means[:, region],
                cfg.regional_melt_targets_m_per_yr[region],
                cfg.gamma0_m_per_yr,
                coefficient,
            )
            for region in range(len(REGION_NAMES))
        ]
    )
    achieved = np.asarray(
        [
            nonlocal_mean_melt(
                delta_t[region], monthly_means[:, region], cfg.gamma0_m_per_yr,
                coefficient
            )
            for region in range(len(REGION_NAMES))
        ]
    )
    return delta_t, achieved, monthly_means


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
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Dask is required to construct and stream the multi-gigabyte "
            "xarray output"
        ) from exc
    output = cfg.output_file
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not cfg.overwrite:
        raise FileExistsError(f"Output exists and overwrite=false: {output}")
    temporary = output.with_name(output.name + ".partial")
    if temporary.exists():
        raise FileExistsError(
            f"Partial output already exists: {temporary}. Remove or rename it "
            f"after inspecting it."
        )

    anchor = np.clip(mesh.bed, -cfg.source_max_depth_m, 0.0)
    base_at_anchor = profiles_at_cell_depths(
        profiles.output_thermal_forcing_degC, cfg.ocean_levels_m, basin_ids,
        anchor
    )
    base_by_cell = profiles.output_thermal_forcing_degC[basin_ids - 1, :]
    delta_t_by_cell = regional_delta_t[basin_ids - 1]

    try:
        with xr.open_dataset(
            cfg.forcing_2d_file,
            decode_times=False,
            mask_and_scale=True,
            concat_characters=False,
            chunks={"Time": 1},
        ) as source:
            forcing_var = validate_forcing_schema(source, cfg, mesh.bed.size)
            times = forcing_times(source)
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

            cell_coord = np.arange(mesh.bed.size, dtype=np.int32)
            layer_coord = np.arange(cfg.ocean_levels_m.size, dtype=np.int32)
            base_cells = xr.DataArray(
                base_by_cell.astype(np.float32),
                dims=("nCells", "nISMIP6OceanLayers"),
                coords={
                    "nCells": cell_coord,
                    "nISMIP6OceanLayers": layer_coord,
                },
            )
            anchor_cells = xr.DataArray(
                base_at_anchor.astype(np.float32),
                dims=("nCells",),
                coords={"nCells": cell_coord},
            )
            forcing_3d = (
                base_cells + (forcing_var.astype(np.float32) - anchor_cells)
            ).transpose("Time", "nCells", "nISMIP6OceanLayers").assign_attrs(
                units="C",
                long_name="3D thermal forcing for nonlocal ISMIP6 ice-shelf "
                "melt method",
            )

            target = xr.Dataset(
                data_vars={
                    "xtime": source["xtime"],
                    "ismip6shelfMelt_basin": xr.DataArray(
                        basin_ids.astype(np.int32),
                        dims=("nCells",),
                        attrs={
                            "description": "One-based basin number for "
                            "regional ISMIP6 shelf-melt forcing"
                        },
                    ),
                    "ismip6shelfMelt_gamma0": xr.DataArray(
                        np.float32(cfg.gamma0_m_per_yr),
                        attrs={
                            "units": "m yr^-1",
                            "description": "Uniform gamma0 for nonlocal "
                            "Jourdain et al. (2020) shelf melt",
                        },
                    ),
                    "ismip6shelfMelt_deltaT": xr.DataArray(
                        delta_t_by_cell.astype(np.float32),
                        dims=("nCells",),
                        attrs={
                            "units": "K",
                            "description": "Regionally calibrated, cellwise "
                            "temperature-bias correction",
                        },
                    ),
                    "ismip6shelfMelt_zOcean": xr.DataArray(
                        cfg.ocean_levels_m.astype(np.float32),
                        dims=("nISMIP6OceanLayers",),
                        attrs={"units": "m", "positive": "up"},
                    ),
                    "ismip6shelfMelt_3dThermalForcing": forcing_3d,
                },
                attrs={
                    "title": "Regional three-dimensional Greenland ocean "
                    "thermal forcing for MALI",
                    "source_2d_forcing": str(cfg.forcing_2d_file),
                    "source_en4_version": cfg.en4_version,
                    "source_en4_bias_correction": cfg.en4_bias_correction,
                    "source_en4_region_geojson": str(
                        cfg.en4_source_region_geojson
                    ),
                    "en4_profile_period": f"{cfg.profile_start_year}-"
                    f"{cfg.profile_end_year}",
                    "deltaT_calibration_period":
                        f"{cfg.calibration_start_year}"
                        f"-{cfg.calibration_end_year}",
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
                "ismip6shelfMelt_basin": {
                    "dtype": "int32", "_FillValue": None
                },
                "ismip6shelfMelt_gamma0": {
                    "dtype": "float32", "_FillValue": None
                },
                "ismip6shelfMelt_deltaT": {
                    "dtype": "float32", "_FillValue": None
                },
                "ismip6shelfMelt_zOcean": {
                    "dtype": "float32", "_FillValue": None
                },
                "ismip6shelfMelt_3dThermalForcing": {
                    "dtype": "float32", "_FillValue": None
                },
            }
            logger.info(
                f"Writing {len(times)} monthly records to {temporary} with "
                "xarray (NETCDF3_64BIT, float32)"
            )
            with dask.config.set(scheduler="single-threaded"):
                target.to_netcdf(
                    temporary,
                    engine="scipy",
                    format="NETCDF3_64BIT",
                    unlimited_dims=["Time"],
                    encoding=encoding,
                )
            target.close()
        os.replace(temporary, output)
    except Exception:
        logger.warning(
            f"Output was not finalized; partial file, if any, is at "
            f"{temporary}"
        )
        raise


def write_diagnostics(
    cfg: Config,
    mesh: MeshData,
    basin_ids: np.ndarray,
    profiles: RegionalProfiles,
    regional_delta_t: np.ndarray,
    achieved_melt: np.ndarray,
    calibration_monthly_tf: np.ndarray,
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
        for region in range(len(REGION_NAMES)):
            for level, z in enumerate(profiles.source_z_m):
                writer.writerow(
                    [
                        region + 1,
                        REGION_KEYS[region],
                        REGION_NAMES[region],
                        float(z),
                        float(profiles.temperature_degC[region, level]),
                        float(profiles.salinity[region, level]),
                        float(
                            profiles.freezing_temperature_degC[region, level]
                        ),
                        float(profiles.thermal_forcing_degC[region, level]),
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
    anchor = np.clip(mesh.bed, -cfg.source_max_depth_m, 0.0)
    base_at_anchor = profiles_at_cell_depths(
        profiles.output_thermal_forcing_degC,
        cfg.ocean_levels_m,
        basin_ids,
        anchor,
    )
    base_by_cell = profiles.output_thermal_forcing_degC[basin_ids - 1]
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
    fig.suptitle(f"Translated 3-D forcing: {times[time_index]}")
    fig.tight_layout()
    fig.savefig(
        directory / "thermal_forcing_at_representative_ocean_levels.png",
        dpi=180,
    )
    plt.close(fig)

    for region in range(len(REGION_NAMES)):
        fig, axes = plt.subplots(1, 2, figsize=(10, 7), sharey=True)
        for monthly in profiles.monthly_temperature_degC[:, region, :]:
            axes[0].plot(
                monthly, profiles.source_z_m, color="tab:blue", alpha=0.08,
                linewidth=0.5
            )
        axes[0].plot(
            profiles.temperature_degC[region],
            profiles.source_z_m,
            color="black",
            linewidth=2,
            label="Monthly mean climatology",
        )
        selected_temp = interpolate_profile(
            profiles.source_z_m, profiles.temperature_degC[region],
            profiles.output_z_m
        )
        axes[0].scatter(
            selected_temp, profiles.output_z_m, color="black", marker="*",
            zorder=3, label="MALI levels"
        )
        axes[0].set_xlabel("Potential temperature (°C)")
        axes[0].set_ylabel("Elevation (m)")
        axes[0].legend(loc="best", fontsize=8)

        for monthly in profiles.monthly_thermal_forcing_degC[:, region, :]:
            axes[1].plot(
                monthly, profiles.source_z_m, color="tab:red", alpha=0.08,
                linewidth=0.5
            )
        axes[1].plot(
            profiles.thermal_forcing_degC[region],
            profiles.source_z_m,
            color="black",
            linewidth=2,
        )
        axes[1].scatter(
            profiles.output_thermal_forcing_degC[region],
            profiles.output_z_m,
            color="black",
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
    achieved_melt: np.ndarray,
    logger,
) -> None:
    logger.info("Regional deltaT calibration")
    logger.info("region          target_m/yr  achieved_m/yr  deltaT_K")
    for region, key in enumerate(REGION_KEYS):
        logger.info(
            f"{key:15s} {cfg.regional_melt_targets_m_per_yr[region]:11.5f} "
            f"{achieved_melt[region]:14.5f} {regional_delta_t[region]:9.5f}"
        )


def run(cfg: Config, logger, prepare_only: bool = False) -> None:
    mesh, basin_ids = load_mesh_and_basins(cfg)
    profiles = build_regional_profiles(cfg, mesh, basin_ids, logger)
    delta_t, achieved, calibration_monthly_tf = calibrate_regional_delta_t(
        cfg, mesh, basin_ids, profiles
    )
    print_summary(cfg, delta_t, achieved, logger)
    write_diagnostics(
        cfg, mesh, basin_ids, profiles, delta_t, achieved,
        calibration_monthly_tf
    )
    if not prepare_only:
        write_output(cfg, mesh, basin_ids, profiles, delta_t, logger)
        logger.info(f"Created {cfg.output_file}")
    else:
        logger.info(
            "Preparation-only run complete; the multi-gigabyte forcing file "
            "was not written"
        )
    logger.info(f"Diagnostics: {cfg.diagnostics_directory}")
