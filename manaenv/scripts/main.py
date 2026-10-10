#!/usr/bin/env python3
"""Lightweight bootstrap for the ManaEnv pipeline CLI."""

from __future__ import annotations

import os
import site
import sys

from manaenv.utils.isaac_runtime_env import reexec_with_isaac_runtime_env
from manaenv.utils.kit_app_runtime import close_kit_app

_DEPENDENCY_SITE = os.environ.get("MANAENV_DEPENDENCY_SITE_PACKAGES")
if _DEPENDENCY_SITE:
    site.addsitedir(_DEPENDENCY_SITE)


def _run_runtime() -> int:
    """Load the runtime only after bootstrap and always release a started Kit app."""
    from manaenv.scripts import pipeline_cli

    exit_code = 1
    try:
        exit_code = pipeline_cli.main()
        return exit_code
    finally:
        simulation_app = pipeline_cli.simulation_app
        if simulation_app is not None:
            close_kit_app(simulation_app, exit_code=exit_code)


if __name__ == "__main__":
    # Guarded: a re-exec must never fire when this module is merely imported;
    # only the real entrypoint may restart.
    for _isaac_env_note in reexec_with_isaac_runtime_env():
        print(f"[INFO] Isaac runtime env: {_isaac_env_note}", flush=True)

    exit_code = _run_runtime()
    sys.exit(exit_code)
