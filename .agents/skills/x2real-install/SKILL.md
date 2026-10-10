---
name: x2real-install
description: Plan, install, repair, or verify the ManaEnv evaluator from the current checkout using its repository-owned installer. Use when asked to set up an environment, prepare dependency sources, reinstall or reuse the simulation stack, diagnose an installation failure, or confirm installation provenance. Excludes assets, policy models/checkpoints, Policy Space service deployment, and evaluation execution.
---

# Install ManaEnv

Use the current checkout's installer as the source of truth. Select the environment,
source URLs, update behavior, and repair mode from the user's request and live machine
state rather than embedding one developer setup in the skill.

## Read the installer first

Resolve the repository root and inspect these live interfaces before composing a plan:

```bash
bash ./install.sh --help
```

- `install.sh` owns environment selection, hardware checks, Arena core installation,
  activation, and final verification.
- `install.py` owns the core dependency checkouts, the X-Square-Robot GitHub Policy Space
  `main` checkout, and editable ManaEnv/source installs.
- `README.md` and `README.zh-CN.md` describe the supported workflow.
- `pyproject.toml` declares the Python and direct-package requirements.

Run `install.py` directly only for a source-level diagnostic the shell installer cannot
express. Normal installations go through `install.sh`.

## Resolve the requested installation

1. Inspect the parent branch/worktree and the existing dependency checkouts under the
   installer-selected dependency root. Preserve unrelated changes. A dirty Newton,
   Arena, IsaacLab, or Policy Space checkout is a stopping condition because the
   installer intentionally refuses to update it.
2. Select exactly one environment mode from the current `--help` output:
   Conda, repository-local uv, or an existing compatible interpreter. Preserve a mode
   or environment name supplied by the user. When none is supplied, show the live
   choices and use the repository default only when the user asks for the default
   installation.
3. Select the operation:
   full installation, sources only, reuse installed sources/packages, forced core
   reinstall, or verification/diagnosis of an existing environment.
4. Resolve every required source URL from explicit flags, documented environment
   variables, the installer's declared default, or an existing checkout remote. Core
   simulation repositories use their configured branch; Policy Space uses the X-Square-Robot GitHub repository's
   `main` branch. Use `--policy-space-url` or `MANAENV_POLICY_SPACE_URL` for a mirror.
5. Check available disk space, the chosen environment tool, network reachability when
   sources or packages are needed, and the installer-required NVIDIA GPU/driver state.
   Use the hardware-check bypass only for an explicitly requested source-only or
   non-GPU diagnostic path.

Environment recreation is destructive. `--yes` authorizes unattended Conda removal,
so use it only when the user explicitly requests recreation. `--force` keeps the
selected environment but reinstalls core packages; use it only for an explicit repair
or reinstall. Preserve an existing environment by choosing the installer's reuse path
rather than deleting it manually.

## Dry-run the exact plan

Run the same backend, environment name, source overrides, and mode intended for the
real operation, with `--dry-run` added:

```bash
source ./install.sh <selected-options> --dry-run
```

Confirm the plan names the intended environment/interpreter, dependency branches and
sources, package version, source-update behavior, and verification policy. A dry-run
must not be reported as an installation.

## Execute through the shell installer

For a full install, source the script so its selected environment remains active in
that shell:

```bash
source ./install.sh <selected-options>
```

For source preparation without activation needs, the installer also supports its
source-only mode as shown by current `--help`.

When tools run commands in separate shells, activation from one call will not carry to
the next. Record the exact target interpreter reported by the installer and use that
interpreter for later verification instead of relying on a shell prompt or the word
`uv` in logs. Keep `UV_HTTP_TIMEOUT` or other package-network settings supplied by the
user; large NVIDIA wheels may need the installer-supported timeout path.

Let the installer manage these operations:

- preparing core dependency checkouts, the X-Square-Robot GitHub Policy Space `main` checkout,
  and the Arena-owned IsaacLab gitlink;
- creating or selecting the environment;
- installing the Isaac Sim/Arena core stack;
- installing the lightweight Policy Space protocol and ManaEnv editables;
- restoring the original working directory and printing the later activation command.

Assets, policy-specific dependencies, checkpoints, and policy services remain separate
from the ManaEnv client installation.

## Verify in layers

Retain the installer's final verification unless the user explicitly accepts the
`--skip-verify` gap. After success, use the exact target interpreter to verify:

1. Python matches the version range declared by the checkout.
2. The installed Isaac Sim distribution matches the installer's selected version.
3. Newton schemas/runtime, IsaacLab, Arena, visualizer, Policy Space protocol, and
   ManaEnv import from the intended source or environment locations.
4. Package dependency consistency (`python -m pip check`).
5. The evaluator's static catalog:

   ```bash
   <target-python> manaenv/scripts/x2real_eval.py --check
   ```

6. When a local asset bundle is present or was part of the request:

   ```bash
   <target-python> manaenv/scripts/x2real_eval.py --asset-check
   ```

Import and catalog checks do not establish Kit startup, camera rendering, physics, or
model inference. Run a Kit smoke test or X2Real episode only when the user requests
that separate runtime validation.

## Diagnose at the failing boundary

Preserve the first causal error and classify it before changing the system:

- source checkout/branch/dirty-state or inaccessible configured URL;
- environment tool, interpreter version, or activation;
- NVIDIA driver/hardware preflight;
- Arena/Isaac Sim core package installation;
- editable source installation or package dependency conflict;
- import provenance/final verification;
- static evaluator or local asset closure.

For an existing environment, prove the interpreter from the installer's reported path
and `sys.executable`. Repair the selected boundary; do not recreate the environment,
reset a dependency checkout, skip verification, or modify nested IsaacLab merely to
silence warnings unless the user specifically authorizes that action.

## Completion

Report the selected mode and environment, exact installer command, source revisions or
remotes prepared, target interpreter, verification results, and any skipped gate. Call
the installation complete only after the installer and its requested verification
layers succeed. State runtime, asset, model-service, and evaluation checks separately.
