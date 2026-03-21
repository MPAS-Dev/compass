"""Compass-specific hooks for ``mache deploy run``."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from packaging.version import Version

if TYPE_CHECKING:
    from mache.deploy.hooks import DeployContext


def pre_pixi(ctx: DeployContext) -> dict[str, Any] | None:
    compass_version = _get_version()
    mpi = _get_pixi_mpi(ctx.machine, ctx.machine_config)
    return {
        'project': {'version': compass_version},
        'pixi': {'mpi': mpi},
    }


def pre_spack(ctx: DeployContext) -> dict[str, Any] | None:
    toolchain_pairs = _get_toolchain_pairs(ctx)
    _check_unsupported(ctx.machine, toolchain_pairs)

    updates: dict[str, Any] = {}
    exclude_packages = _get_spack_exclude_packages(ctx.config)
    _maybe_exclude_e3sm_hdf5_netcdf(
        exclude_packages=exclude_packages, machine_config=ctx.machine_config
    )
    _maybe_exclude_cmake(
        exclude_packages=exclude_packages, machine_config=ctx.machine_config
    )

    spack_path = _get_spack_path(ctx.config, ctx.machine, ctx.machine_config)
    if spack_path is not None:
        updates['spack'] = {'spack_path': spack_path}
    if exclude_packages:
        updates.setdefault('spack', {})['exclude_packages'] = exclude_packages

    if _with_albany(ctx):
        _check_albany_support(ctx.machine, toolchain_pairs)
        _set_albany_config(ctx)

    return updates


def _get_version() -> str:
    here = Path(__file__).resolve().parent
    version_path = here.parent / 'compass' / 'version.py'
    namespace: dict[str, str] = {}
    exec(version_path.read_text(encoding='utf-8'), namespace)
    return namespace['__version__']


def _get_pixi_mpi(machine: str | None, machine_config) -> str:
    if machine is not None and not machine.startswith('conda'):
        return 'nompi'

    if not machine_config.has_section('deploy'):
        raise ValueError("Missing 'deploy' section in machine config")

    compiler = machine_config.get('deploy', 'compiler', fallback='').strip()
    if not compiler:
        raise ValueError("Missing 'compiler' option in 'deploy' section")

    mpi_option = f'mpi_{compiler.replace("-", "_")}'
    mpi = machine_config.get('deploy', mpi_option, fallback='').strip()
    if not mpi:
        raise ValueError(
            f"Missing '{mpi_option}' option in 'deploy' section"
        )
    return mpi


def _get_spack_path(config, machine: str | None, machine_config) -> str | None:
    spack_cfg = config.get('spack', {})
    if isinstance(spack_cfg, dict):
        spack_path = spack_cfg.get('spack_path')
        if spack_path not in (None, '', 'null', 'None'):
            return None

    if machine is None or machine.startswith('conda'):
        return None

    if not machine_config.has_section('deploy'):
        raise ValueError("Missing 'deploy' section in machine config")

    spack_base = machine_config.get('deploy', 'spack', fallback='').strip()
    if not spack_base:
        raise ValueError("Missing 'spack' option in 'deploy' section")

    release_version = Version(_get_version()).base_version
    return os.path.join(spack_base, f'dev_compass_{release_version}')


def _get_toolchain_pairs(ctx: DeployContext) -> list[tuple[str, str]]:
    runtime_toolchain = ctx.runtime.get('toolchain', {})
    if not isinstance(runtime_toolchain, dict):
        return []
    pairs = runtime_toolchain.get('pairs', [])
    if not isinstance(pairs, list):
        return []

    toolchain_pairs: list[tuple[str, str]] = []
    for pair in pairs:
        if not isinstance(pair, dict):
            continue
        compiler = str(pair.get('compiler', '')).strip()
        mpi = str(pair.get('mpi', '')).strip()
        if compiler and mpi:
            toolchain_pairs.append((compiler, mpi))
    return toolchain_pairs


def _get_spack_exclude_packages(config) -> list[str]:
    spack_cfg = config.get('spack', {})
    if not isinstance(spack_cfg, dict):
        return []

    exclude_packages = spack_cfg.get('exclude_packages', [])
    if exclude_packages is None:
        return []
    if isinstance(exclude_packages, str):
        return [exclude_packages]

    return [str(package) for package in exclude_packages]


def _maybe_exclude_e3sm_hdf5_netcdf(
    exclude_packages: list[str], machine_config
) -> None:
    use_bundle = False
    if machine_config.has_section('deploy') and machine_config.has_option(
        'deploy', 'use_e3sm_hdf5_netcdf'
    ):
        use_bundle = machine_config.getboolean(
            'deploy', 'use_e3sm_hdf5_netcdf'
        )

    if not use_bundle and 'hdf5_netcdf' not in exclude_packages:
        exclude_packages.append('hdf5_netcdf')


def _maybe_exclude_cmake(
    exclude_packages: list[str], machine_config
) -> None:
    exclude_cmake = False
    if machine_config.has_section('deploy') and machine_config.has_option(
        'deploy', 'exclude_system_cmake'
    ):
        exclude_cmake = machine_config.getboolean(
            'deploy', 'exclude_system_cmake'
        )

    if exclude_cmake and 'cmake' not in exclude_packages:
        exclude_packages.append('cmake')


def _check_unsupported(
    machine: str | None, toolchain_pairs: list[tuple[str, str]]
) -> None:
    if machine is None or not toolchain_pairs:
        return

    unsupported = _read_triplets(Path('deploy') / 'unsupported.txt', machine)
    for compiler, mpi in toolchain_pairs:
        if (compiler, mpi) in unsupported:
            raise ValueError(
                f'{compiler} with {mpi} is not supported on {machine}'
            )


def _check_albany_support(
    machine: str | None, toolchain_pairs: list[tuple[str, str]]
) -> None:
    if machine is None:
        raise ValueError('Albany deployment requires a known machine')
    if not toolchain_pairs:
        raise ValueError('Albany deployment requires a compiler and MPI pair')

    supported = _read_triplets(
        Path('deploy') / 'albany_supported.txt', machine
    )
    for compiler, mpi in toolchain_pairs:
        if (compiler, mpi) not in supported:
            raise ValueError(
                f'{compiler} with {mpi} is not supported with albany on '
                f'{machine}'
            )


def _set_albany_config(ctx: DeployContext) -> None:
    spack_cfg = ctx.config.setdefault('spack', {})
    spack_cfg['env_name_prefix'] = 'compass_albany'

    spack_pins = ctx.pins.setdefault('spack', {})
    spack_pins['albany_enabled'] = 'true'
    spack_pins['albany_variants'] = _get_machine_override(
        ctx.machine_config,
        option='albany_variants',
        fallback=spack_pins.get('albany_variants', '+mpas~py+unit_tests'),
    )
    spack_pins['trilinos_variants'] = _get_machine_override(
        ctx.machine_config,
        option='trilinos_variants',
        fallback=spack_pins.get('trilinos_variants', ''),
    )


def _get_machine_override(machine_config, option: str, fallback: str) -> str:
    if not machine_config.has_section('deploy'):
        return fallback
    value = machine_config.get('deploy', option, fallback=fallback)
    return str(value).strip()


def _with_albany(ctx: DeployContext) -> bool:
    # The custom CLI flag takes precedence; the env var remains a fallback for
    # compatibility with older workflows.
    return (
        bool(getattr(ctx.args, 'with_albany', False)) or
        _with_albany_from_env()
    )


def _with_albany_from_env() -> bool:
    return os.environ.get('COMPASS_DEPLOY_WITH_ALBANY', '').lower() in (
        '1',
        'true',
        'yes',
        'on',
    )


def _read_triplets(
    filename: Path, machine: str
) -> set[tuple[str, str]]:
    triples: set[tuple[str, str]] = set()
    for line in filename.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = [part.strip() for part in line.split(',')]
        if len(parts) != 3:
            raise ValueError(f'Bad line in "{filename.name}": {line}')
        if parts[0] == machine:
            triples.add((parts[1], parts[2]))
    return triples
