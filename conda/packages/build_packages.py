#!/usr/bin/env python3
import argparse
import os
import platform
import shlex
import subprocess
import tempfile
from pathlib import Path

import yaml


def get_variant_configs(variants_dir, platform_tag):
    config_files = sorted(Path(variants_dir).glob("*.yaml"))
    filtered = []
    for config in config_files:
        name = config.name
        if platform_tag and not name.startswith(platform_tag):
            continue
        filtered.append(str(config))
    return filtered


def apply_channel_overrides(variant_config_path, channel_sources, outputs_dir,
                            overrides=None):
    with open(variant_config_path) as handle:
        variant_config = yaml.safe_load(handle) or {}

    variant_config["channel_sources"] = [",".join(channel_sources)]
    if overrides:
        variant_config.update(overrides)

    override_dir = outputs_dir / "variant_overrides"
    override_dir.mkdir(parents=True, exist_ok=True)
    override_path = (
        override_dir / f"{Path(variant_config_path).stem}_labels.yaml"
    )
    with open(override_path, "w") as handle:
        yaml.safe_dump(variant_config, handle, sort_keys=False)
    return str(override_path)


def get_platform_tag():
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "linux":
        if machine in {"x86_64", "amd64"}:
            return "linux_64"
        if machine in {"aarch64", "arm64"}:
            return "linux_aarch64"
    if system == "darwin":
        if machine in {"x86_64", "amd64"}:
            return "osx_64"
        if machine in {"aarch64", "arm64"}:
            return "osx_arm64"
    return None


def get_variant_values(variant_file):
    with open(variant_file) as handle:
        data = yaml.safe_load(handle) or {}
    python_spec = None
    mpi = None
    if "python_min" in data and data["python_min"]:
        python_spec = f"{data['python_min'][0]}"
    elif "python" in data and data["python"]:
        python_spec = str(data["python"][0]).split()[0]
    if "mpi" in data and data["mpi"]:
        mpi = str(data["mpi"][0])
    return python_spec, mpi


def normalize_spec(spec, mpi_prefix):
    rendered = spec.replace("${{ mpi_prefix }}", mpi_prefix)
    tokens = rendered.split()
    if len(tokens) <= 2:
        return rendered
    return " ".join(tokens[:2])


def parse_run_requirements(recipe_path, mpi_prefix, platform_tag):
    with open(recipe_path) as handle:
        recipe = yaml.safe_load(handle) or {}
    requirements = recipe.get("requirements", {})
    run_reqs = requirements.get("run", [])
    specs = []
    for entry in run_reqs:
        if isinstance(entry, str):
            specs.append(normalize_spec(entry, mpi_prefix))
        elif isinstance(entry, dict) and "if" in entry and "then" in entry:
            condition = str(entry["if"]).lower()
            if condition == "linux" and platform_tag.startswith("linux"):
                for then_entry in entry.get("then", []):
                    specs.append(normalize_spec(then_entry, mpi_prefix))
    return specs


def get_conda_sh():
    conda_sh = os.environ.get("CONDA_SH")
    if conda_sh:
        return Path(conda_sh).expanduser()
    return Path.home() / "miniforge3" / "etc" / "profile.d" / "conda.sh"


def run_conda_command(args, extra_env=None):
    conda_sh = get_conda_sh()
    if not conda_sh.exists():
        raise FileNotFoundError(
            f"Conda activation script not found: {conda_sh}"
        )
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    cmd = " ".join(shlex.quote(arg) for arg in args)
    subprocess.run(
        [
            "bash",
            "-lc",
            f"source '{conda_sh}' && conda {cmd}",
        ],
        check=True,
        env=env,
    )


def prefetch_with_conda(variant_file, recipe_dir, outputs_dir, platform_tag,
                        channels, mpi_override=None):
    python_spec, mpi = get_variant_values(variant_file)
    if python_spec is None:
        python_spec = ">=3.10"
    if mpi_override is not None:
        mpi = mpi_override
    elif mpi is None:
        mpi = "nompi"
    mpi_prefix = "nompi" if mpi == "nompi" else f"mpi_{mpi}"
    recipe_path = Path(recipe_dir) / "recipe.yaml"
    run_specs = parse_run_requirements(recipe_path, mpi_prefix, platform_tag)
    specs = [f"python {python_spec}", "pip", "setuptools"] + run_specs
    prefetch_root = outputs_dir / "prefetch"
    env_dir = prefetch_root / f"conda_env_{Path(variant_file).stem}"
    pkgs_dir = prefetch_root / "conda_pkgs"
    args = [
        "create",
        "--download-only",
        "--yes",
        "--override-channels",
        "--repodata-fn",
        "repodata.json",
        "--prefix",
        str(env_dir),
    ]
    for channel in channels:
        args.extend(["-c", channel])
    args.extend(specs)
    print("Prefetching packages with conda...")
    try:
        run_conda_command(
            args,
            extra_env={
                "CONDA_PKGS_DIRS": str(pkgs_dir),
            },
        )
    except subprocess.CalledProcessError:
        print("Conda prefetch failed; continuing with remote downloads.")
        return None
    return pkgs_dir


def run_build(variant_file, recipe_dir, outputs_dir, config_file, cache_dir):
    cmd = [
        "rattler-build",
        "build",
        "--config-file",
        str(config_file),
        "--io-concurrency-limit",
        "1",
        "-m",
        variant_file,
        "-r",
        str(recipe_dir),
        "--output-dir",
        str(outputs_dir),
    ]
    env = os.environ.copy()
    env.setdefault("REQWEST_DISABLE_HTTP2", "1")
    env.setdefault("RATTLER_IO_CONCURRENCY_LIMIT", "1")
    env.setdefault("RATTLER_CACHE_DIR", str(cache_dir))
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def main():
    parser = argparse.ArgumentParser(
        description="Build compass and OTPS conda packages with rattler-build."
    )
    parser.add_argument(
        "--otps",
        action="store_true",
        help="Only build OTPS variants."
    )
    parser.add_argument(
        "--compass",
        action="store_true",
        help="Only build COMPASS variants."
    )
    parser.add_argument(
        "--prefetch",
        action="store_true",
        help="Prefetch dependencies using conda and reuse its package cache."
    )
    parser.add_argument(
        "--mpi",
        nargs="+",
        help="MPI variant(s) to build for (overrides default matrix)."
    )
    args = parser.parse_args()

    recipe_root = Path(__file__).parent
    config_file = recipe_root / "rattler-build-config.toml"
    outputs_dir = recipe_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    build_otps = args.otps or not args.compass
    build_compass = args.compass or not args.otps

    platform_tag = get_platform_tag()
    if platform_tag is None:
        raise ValueError(
            "Unsupported platform for automatic variant selection."
        )

    default_cache_dir = Path(
        os.environ.get("TMPDIR", tempfile.gettempdir())
    ) / "rattler_cache"

    if build_otps:
        otps_dir = recipe_root / "otps"
        otps_variants_dir = otps_dir / "variants"
        otps_recipe_dir = otps_dir / "recipe"
        otps_variants = get_variant_configs(
            variants_dir=otps_variants_dir,
            platform_tag=platform_tag,
        )
        if not otps_variants:
            raise ValueError(
                "No OTPS variant config files matched the requested filters."
            )
        otps_channel_sources = ["https://conda.anaconda.org/conda-forge"]
        otps_prefetch_sources = otps_channel_sources
        for variant_file in otps_variants:
            effective_sources = otps_channel_sources
            cache_dir = default_cache_dir
            if args.prefetch:
                prefetch_cache_dir = prefetch_with_conda(
                    variant_file=variant_file,
                    recipe_dir=otps_recipe_dir,
                    outputs_dir=outputs_dir,
                    platform_tag=platform_tag,
                    channels=otps_prefetch_sources,
                )
                if prefetch_cache_dir:
                    cache_dir = Path(prefetch_cache_dir)
            override_file = apply_channel_overrides(
                variant_config_path=variant_file,
                channel_sources=effective_sources,
                outputs_dir=outputs_dir,
            )
            run_build(
                override_file,
                otps_recipe_dir,
                outputs_dir,
                config_file,
                cache_dir=cache_dir,
            )

    if build_compass:
        compass_dir = recipe_root / "compass"
        compass_variants_dir = compass_dir / "variants"
        compass_recipe_dir = compass_dir / "recipe"
        compass_build_config = compass_recipe_dir / "conda_build_config.yaml"
        with open(compass_build_config) as handle:
            compass_build_data = yaml.safe_load(handle) or {}
        default_mpis = compass_build_data.get("mpi", ["nompi"])
        mpi_variants = args.mpi or default_mpis
        compass_variants = get_variant_configs(
            variants_dir=compass_variants_dir,
            platform_tag=platform_tag,
        )
        if not compass_variants:
            raise ValueError(
                "No COMPASS variant config files matched the requested "
                "filters."
            )
        channel_sources = [
            "https://conda.anaconda.org/conda-forge",
            "e3sm/label/compass",
        ]
        prefetch_sources = channel_sources
        for variant_file in compass_variants:
            for mpi_variant in mpi_variants:
                effective_sources = channel_sources
                cache_dir = default_cache_dir
                if args.prefetch:
                    prefetch_cache_dir = prefetch_with_conda(
                        variant_file=variant_file,
                        recipe_dir=compass_recipe_dir,
                        outputs_dir=outputs_dir,
                        platform_tag=platform_tag,
                        channels=prefetch_sources,
                        mpi_override=mpi_variant,
                    )
                    if prefetch_cache_dir:
                        cache_dir = Path(prefetch_cache_dir)
                override_file = apply_channel_overrides(
                    variant_config_path=variant_file,
                    channel_sources=effective_sources,
                    outputs_dir=outputs_dir,
                    overrides={"mpi": [str(mpi_variant)]},
                )
                run_build(
                    override_file,
                    compass_recipe_dir,
                    outputs_dir,
                    config_file,
                    cache_dir=cache_dir,
                )


if __name__ == "__main__":
    main()
