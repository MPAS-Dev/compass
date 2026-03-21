#!/usr/bin/env python3
"""
Target software deployment entrypoint.

- Reads pinned mache version from deploy/pins.cfg
- Reads CLI spec from deploy/cli_spec.json and builds argparse CLI
- Downloads mache/deploy/bootstrap.py for either:
    * a given mache fork/branch, or
    * the pinned mache version
- Calls bootstrap.py with routed args (bootstrap|both) and stops
"""

import argparse
import configparser
import json
import os
import shlex
import shutil
import stat
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PINS_CFG = os.path.join('deploy', 'pins.cfg')
CLI_SPEC_JSON = os.path.join('deploy', 'cli_spec.json')
DEPLOY_TMP_DIR = 'deploy_tmp'
BOOTSTRAP_PATH = os.path.join(DEPLOY_TMP_DIR, 'bootstrap.py')

# Default upstream repo for release/tag downloads
DEFAULT_MACHE_REPO = 'E3SM-Project/mache'

# Where bootstrap.py lives inside the mache repo
BOOTSTRAP_RELPATH = 'mache/deploy/bootstrap.py'


def main():
    _check_location()

    pinned_mache_version, pinned_python_version = _read_pins(PINS_CFG)
    cli_spec = _read_cli_spec(CLI_SPEC_JSON)

    parser = _build_parser_from_cli_spec(cli_spec)
    args = parser.parse_args(sys.argv[1:])

    if args.python:
        python_version = args.python
    else:
        python_version = pinned_python_version

    _validate_fork_branch_pair(args)

    using_fork = getattr(args, 'mache_fork', None) is not None
    requested_mache_version = str(
        getattr(args, 'mache_version', '') or ''
    ).strip()

    if not using_fork:
        _validate_cli_spec_matches_pins(cli_spec, pinned_mache_version)

    bootstrap_mache_version = pinned_mache_version
    if not using_fork and requested_mache_version:
        bootstrap_mache_version = requested_mache_version

    # remove tmp dir
    if os.path.exists(DEPLOY_TMP_DIR):
        shutil.rmtree(DEPLOY_TMP_DIR)

    os.makedirs(DEPLOY_TMP_DIR)

    bootstrap_url = _bootstrap_url(
        mache_version=bootstrap_mache_version,
        mache_fork=getattr(args, 'mache_fork', None),
        mache_branch=getattr(args, 'mache_branch', None),
    )

    _download_file(bootstrap_url, BOOTSTRAP_PATH)

    # Make sure it's executable (nice-to-have). We'll still run with
    # sys.executable.
    _make_executable(BOOTSTRAP_PATH)

    bootstrap_argv = _build_routed_argv(cli_spec, args, route_key='bootstrap')

    software = str(cli_spec.get('meta', {}).get('software', '')).strip()
    if not software:
        raise SystemExit(
            'ERROR: deploy/cli_spec.json meta.software must be set to the '
            'target software name.'
        )
    # Always include target software name (not user-facing).
    bootstrap_argv = [
        '--software',
        software,
        '--python',
        python_version,
    ] + bootstrap_argv

    # Only pass a mache version when using a tagged release. If a fork/branch
    # is requested, bootstrap must take dependencies from the branch's
    # pixi.toml (not from a pinned release).
    if not using_fork:
        if '--mache-version' not in bootstrap_argv:
            bootstrap_argv += ['--mache-version', pinned_mache_version]

    cmd = [sys.executable, BOOTSTRAP_PATH] + bootstrap_argv
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        raise SystemExit(
            f'\nERROR: Bootstrap step failed (exit code {e.returncode}). '
            f'See the error output above.'
        ) from None

    if args.bootstrap_only:
        pixi_exe = _get_pixi_executable(getattr(args, 'pixi', None))
        bootstrap_dir = os.path.join(DEPLOY_TMP_DIR, 'bootstrap_pixi')
        update_cmd = f'mache deploy update --software {software}'
        if requested_mache_version:
            update_cmd = (
                f'{update_cmd} --mache-version '
                f'{shlex.quote(requested_mache_version)}'
            )
        print(
            '\nBootstrap environment is ready. To use it interactively:\n'
            f'  pixi shell -m {bootstrap_dir}/pixi.toml\n\n'
            'Then, you can run:\n'
            f'  {update_cmd}\n'
            'After update, edit deploy/pins.cfg to set [pixi] mache to the '
            'new version.\n'
            f'  exit\n'
        )

    # Now that the bootstrap env exists and has mache installed, run
    # deployment. Forward args routed to "mache".
    mache_run_argv = _build_routed_argv(cli_spec, args, route_key='run')

    if not args.bootstrap_only:
        pixi_exe = _get_pixi_executable(getattr(args, 'pixi', None))
        if '--pixi' not in mache_run_argv:
            mache_run_argv = ['--pixi', pixi_exe] + mache_run_argv
        _run_mache_deploy_run(
            pixi_exe=pixi_exe,
            repo_root='.',
            mache_run_argv=mache_run_argv,
        )


def _check_location():
    """Fail fast if not run from repo root."""
    expected = [
        'deploy.py',
        PINS_CFG,
        CLI_SPEC_JSON,
    ]
    missing = [p for p in expected if not os.path.exists(p)]
    if missing:
        missing_str = '\n  - ' + '\n  - '.join(missing)
        raise SystemExit(
            f'ERROR: deploy.py must be run from the root of the target '
            f'software repository.\n'
            f'Current location: {os.getcwd()}\n'
            f'Missing expected files:{missing_str}'
        )


def _read_pins(pins_path):
    if not os.path.exists(pins_path):
        raise SystemExit(f'ERROR: Required pins file not found: {pins_path}')

    cfg = configparser.ConfigParser(interpolation=None)
    try:
        with open(pins_path, 'r', encoding='utf-8') as f:
            cfg.read_file(f)
    except OSError as e:
        raise SystemExit(f'ERROR: Failed to read {pins_path}: {e!r}') from e

    section = None
    if cfg.has_section('pixi') and cfg.has_option('pixi', 'mache'):
        section = 'pixi'

    if section is None:
        raise SystemExit(f'ERROR: {pins_path} must contain [pixi] mache')

    mache_version = cfg.get(section, 'mache').strip()
    if not mache_version:
        raise SystemExit(
            f'ERROR: {pins_path} option [{section}] mache is empty'
        )

    python_version = cfg.get(section, 'python').strip()
    if not python_version:
        raise SystemExit(
            f'ERROR: {pins_path} option [{section}] python is empty'
        )

    return mache_version, python_version


def _read_cli_spec(spec_path):
    if not os.path.exists(spec_path):
        raise SystemExit(f'ERROR: Required CLI spec not found: {spec_path}')

    try:
        with open(spec_path, 'r', encoding='utf-8') as f:
            spec = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise SystemExit(f'ERROR: Failed to parse {spec_path}: {e!r}') from e

    if 'meta' not in spec or 'arguments' not in spec:
        raise SystemExit(
            f"ERROR: {spec_path} must contain top-level keys 'meta' and "
            f"'arguments'"
        )

    if 'mache_version' not in spec['meta']:
        raise SystemExit(
            f"ERROR: {spec_path} meta must include 'mache_version'"
        )

    if not isinstance(spec['arguments'], list):
        raise SystemExit(f"ERROR: {spec_path} 'arguments' must be a list")

    return spec


def _build_parser_from_cli_spec(cli_spec):
    description = cli_spec.get('meta', {}).get(
        'description', 'Deploy E3SM software environment'
    )
    parser = argparse.ArgumentParser(description=description)

    for entry, flags in _iter_routed_cli_spec_entries(
        cli_spec, route_key='deploy'
    ):
        # Build kwargs for argparse. Only allow a small, safe subset.
        kwargs = {}
        for key in (
            'dest',
            'help',
            'action',
            'default',
            'required',
            'choices',
            'nargs',
        ):
            if key in entry:
                kwargs[key] = entry[key]

        # NOTE: intentionally not supporting arbitrary 'type' here to keep it
        # simple/stdlib-only. If you need types later, you can support a
        # limited string->callable mapping.

        try:
            parser.add_argument(*flags, **kwargs)
        except TypeError as e:
            raise SystemExit(
                f'ERROR: Bad argparse spec for flags {flags}: {e}'
            ) from e

    return parser


def _iter_routed_cli_spec_entries(cli_spec, route_key):
    """Yield (entry, flags) for entries whose route contains route_key.

    This function centralizes CLI-spec validation shared between parser
    construction and argv forwarding.
    """
    for entry in cli_spec['arguments']:
        flags = entry.get('flags')
        route = entry.get('route')

        if not isinstance(route, list):
            raise SystemExit(
                f'ERROR: cli_spec.json argument {entry.get("flags")} has '
                f"invalid 'route'; must be a list"
            )

        if route_key not in route:
            continue

        if not flags or not isinstance(flags, list):
            raise SystemExit("ERROR: cli_spec.json entry missing 'flags' list")

        yield entry, flags


def _validate_fork_branch_pair(args):
    fork = getattr(args, 'mache_fork', None)
    branch = getattr(args, 'mache_branch', None)
    if (fork is None) != (branch is None):
        raise SystemExit(
            'ERROR: You must supply both --mache-fork and --mache-branch, or '
            'neither.'
        )


def _validate_cli_spec_matches_pins(cli_spec, pinned_mache_version):
    meta_version = str(cli_spec['meta'].get('mache_version', '')).strip()
    if not meta_version:
        raise SystemExit('ERROR: cli_spec.json meta.mache_version is empty')

    if meta_version != pinned_mache_version:
        raise SystemExit(
            f'ERROR: Mache version mismatch.\n'
            f'  deploy/pins.cfg pins mache = {pinned_mache_version}\n'
            f'  deploy/cli_spec.json meta.mache_version = {meta_version}\n\n'
            f'Fix: copy deploy/cli_spec.json from the matching mache version '
            f'into this repo (or update both together).'
        )


def _bootstrap_url(
    mache_version,
    mache_fork=None,
    mache_branch=None,
):
    override_url = str(os.environ.get('MACHE_BOOTSTRAP_URL', '')).strip()
    if override_url:
        return override_url

    if mache_fork is not None and mache_branch is not None:
        # Raw file from a fork/branch
        return f'https://raw.githubusercontent.com/{mache_fork}/{mache_branch}/{BOOTSTRAP_RELPATH}'  # noqa: E501

    # Raw file from a version tag. Convention: tags are "X.Y.Z".
    return f'https://raw.githubusercontent.com/{DEFAULT_MACHE_REPO}/{mache_version}/{BOOTSTRAP_RELPATH}'  # noqa: E501


def _download_file(url, dest_path):
    # Avoid stale/cached responses from proxies/CDNs (common on HPC networks).
    # GitHub raw content supports query strings; adding a cache-buster forces a
    # fresh fetch even if an intermediate cache is misbehaving.
    effective_url = url
    if 'raw.githubusercontent.com' in url:
        sep = '&' if '?' in url else '?'
        effective_url = f'{url}{sep}_cb={int(time.time())}'

    req = Request(
        effective_url,
        headers={
            'User-Agent': 'Mozilla/5.0',
            'Cache-Control': 'no-cache, no-store, max-age=0',
            'Pragma': 'no-cache',
        },
    )
    try:
        with urlopen(req, timeout=60) as resp:
            data = resp.read()
    except HTTPError as e:
        raise SystemExit(
            f'ERROR: Failed to download bootstrap.py (HTTP {e.code}) from '
            f'{effective_url}'
        ) from e
    except URLError as e:
        raise SystemExit(
            f'ERROR: Failed to download bootstrap.py from {effective_url}: '
            f'{e.reason}'
        ) from e
    except Exception as e:
        raise SystemExit(
            f'ERROR: Unexpected error downloading bootstrap.py from '
            f'{effective_url}: '
            f'{e!r}'
        ) from e

    # Basic sanity check: should look like a python script.
    first_line = data.splitlines()[0].strip() if data else b''
    if b'python' not in first_line and b'#!/' not in first_line:
        raise SystemExit(
            f'ERROR: Downloaded bootstrap.py does not look like a python '
            f'script.\n'
            f'URL: {effective_url}\n'
            f'This may indicate a proxy/redirect issue.'
        )

    try:
        with open(dest_path, 'wb') as f:
            f.write(data)
    except OSError as e:
        raise SystemExit(f'ERROR: Failed to write {dest_path}: {e!r}') from e


def _make_executable(path):
    try:
        st = os.stat(path)
        os.chmod(path, st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    except OSError:
        # Not fatal; we run via sys.executable anyway.
        pass


def _get_pixi_executable(pixi):
    if pixi:
        pixi = os.path.abspath(os.path.expanduser(pixi))
        if not os.path.exists(pixi):
            raise SystemExit(f'ERROR: pixi executable not found: {pixi}')
        return pixi

    which = shutil.which('pixi')
    if which is not None:
        return which

    default_pixi = os.path.join(
        os.path.expanduser('~'), '.pixi', 'bin', 'pixi'
    )
    if os.path.isfile(default_pixi) and os.access(default_pixi, os.X_OK):
        return default_pixi

    raise SystemExit(
        'ERROR: pixi executable not found on PATH or default install '
        'location (~/.pixi/bin). Install pixi or pass --pixi.'
    )


def _build_routed_argv(cli_spec, args, route_key):
    """Build forwarded argv from args for entries routed to route_key."""
    argv = []
    for entry, flags in _iter_routed_cli_spec_entries(
        cli_spec, route_key=route_key
    ):
        dest = entry.get('dest')
        if not dest:
            raise SystemExit(
                f"ERROR: cli_spec.json argument {flags} missing 'dest'"
            )

        value = getattr(args, dest, None)
        action = entry.get('action')

        # Use the first flag as the canonical one when forwarding.
        flag0 = flags[0]

        if action == 'store_true':
            if value:
                argv.append(flag0)
        else:
            if value is None:
                continue

            # If the argparse entry used `nargs` (or otherwise produced a
            # list), expand into repeated tokens: `--flag a b c`.
            if isinstance(value, (list, tuple)):
                if len(value) == 0:
                    continue
                argv.append(flag0)
                argv.extend(str(v) for v in value)
            else:
                argv.extend([flag0, str(value)])

    return argv


def _run_mache_deploy_run(pixi_exe, repo_root, mache_run_argv):
    """
    Run `mache deploy run ...` inside the bootstrap pixi environment.
    """
    repo_root = os.path.abspath(repo_root)

    bootstrap_dir = os.path.abspath(
        os.path.join(DEPLOY_TMP_DIR, 'bootstrap_pixi')
    )
    pixi_toml = os.path.join(bootstrap_dir, 'pixi.toml')
    if not os.path.exists(pixi_toml):
        raise SystemExit(
            f'ERROR: bootstrap pixi project not found. Expected: {pixi_toml}'
        )

    # Build a bash command that runs mache inside pixi, then cd's to repo.
    mache_cmd = 'mache deploy run'
    if mache_run_argv:
        mache_cmd = f'{mache_cmd} ' + ' '.join(
            shlex.quote(a) for a in mache_run_argv
        )

    cmd = (
        f'env -u PIXI_PROJECT_MANIFEST -u PIXI_PROJECT_ROOT '
        f'-u PIXI_ENVIRONMENT_NAME -u PIXI_IN_SHELL '
        f'{shlex.quote(pixi_exe)} run -m {shlex.quote(pixi_toml)} bash -lc '
        f'{shlex.quote("cd " + repo_root + " && " + mache_cmd)}'
    )
    try:
        subprocess.check_call(['/bin/bash', '-lc', cmd])
    except subprocess.CalledProcessError as e:
        raise SystemExit(
            f'\nERROR: Deployment step failed (exit code {e.returncode}). '
            f'See the error output above.'
        ) from None


if __name__ == '__main__':
    main()
