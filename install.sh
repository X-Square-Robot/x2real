#!/usr/bin/env bash
# Install ManaEnv and the publication simulation stack.
# Sourcing this script leaves the selected Conda/uv environment active.

if [[ -n "${BASH_VERSION:-}" ]]; then
    MANAENV_INSTALL_SOURCE="${BASH_SOURCE[0]}"
elif [[ -n "${ZSH_VERSION:-}" ]]; then
    eval 'MANAENV_INSTALL_SOURCE="${(%):-%x}"'
else
    printf '%s\n' \
        '[ERROR] install.sh requires Bash or zsh.' \
        '        Run: source ./install.sh [options]' >&2
    # shellcheck disable=SC2317
    return 1 2>/dev/null || exit 1
fi

MANAENV_ROOT="$(cd -- "$(dirname -- "$MANAENV_INSTALL_SOURCE")" &>/dev/null && pwd)"
MANAENV_DEFAULT_ENV="manaenv"
MANAENV_ISAACSIM_VERSION="6.0.1.0"
MANAENV_UV_ENV_ROOT="$MANAENV_ROOT/.venv"
MANAENV_ARENA_DIR="$MANAENV_ROOT/3rd/isaaclabarena"
MANAENV_ARENA_INSTALLER="$MANAENV_ARENA_DIR/setup_arena_local.sh"

_MANAENV_RED=$'\033[0;31m'
_MANAENV_GREEN=$'\033[0;32m'
_MANAENV_YELLOW=$'\033[1;33m'
_MANAENV_BLUE=$'\033[0;34m'
_MANAENV_RESET=$'\033[0m'

print_info() { printf '%s\n' "${_MANAENV_BLUE}[INFO]${_MANAENV_RESET} $*"; }
print_success() { printf '%s\n' "${_MANAENV_GREEN}[SUCCESS]${_MANAENV_RESET} $*"; }
print_warning() { printf '%s\n' "${_MANAENV_YELLOW}[WARNING]${_MANAENV_RESET} $*" >&2; }
print_error() { printf '%s\n' "${_MANAENV_RED}[ERROR]${_MANAENV_RESET} $*" >&2; }

show_help() {
    cat <<EOF
Install ManaEnv and its x2real-branch simulation dependencies.
Isaac Sim is installed from NVIDIA's pip index at version 6.0.1.0; no
IsaacSim source checkout is required.

Usage:
  source ./install.sh [options]
  bash ./install.sh [options]

Environment options:
  --conda <name>          Use/create a Conda environment.
                          Default: --conda $MANAENV_DEFAULT_ENV
  --uv <name>             Use/create .venv/<name> under this repository.
  --python <path>         Use an existing Python 3.12 interpreter; do not
                          create an environment or run the Arena core installer.
  -f, --force             Keep the selected environment and force-reinstall
                          IsaacLab and Arena.
  -y, --yes               Recreate an existing Conda environment without asking.

Source options:
  --arena-url <url>       Override the public Arena x2real URL.
  --isaaclab-url <url>    Override the public IsaacLab x2real URL.
  --isaacsim-url <url>    Override the public Isaac Sim x2real URL.
  --policy-space-url <url>
                          Override the GitHub Policy Space main-branch URL.
  --skip-source-update    Keep clean existing source revisions.
  --skip-sources          Use already-installed simulation packages.
  --sources-only          Prepare dependency source checkouts without package installs.

Other options:
  --skip-hardware-check   Skip NVIDIA GPU/driver checks.
  --skip-verify           Skip final import verification.
  --dry-run               Print the selected plan and mutating commands.
  -h, --help              Show this help message.

Examples:
  source ./install.sh
  source ./install.sh --conda $MANAENV_DEFAULT_ENV
  source ./install.sh --uv $MANAENV_DEFAULT_ENV
  source ./install.sh --conda $MANAENV_DEFAULT_ENV --force
  bash ./install.sh --sources-only
EOF
}

_require_arg() {
    if [[ -z "${2-}" ]]; then
        print_error "$1 requires an argument."
        return 1
    fi
}

_run_or_fail() {
    local message="$1"
    shift
    "$@" || {
        print_error "$message"
        return 1
    }
}

_find_conda_executable() {
    local candidate
    if [[ -n "${CONDA_EXE:-}" && -x "$CONDA_EXE" ]]; then
        printf '%s\n' "$CONDA_EXE"
        return 0
    fi
    if command -v conda >/dev/null 2>&1; then
        command -v conda
        return 0
    fi
    for candidate in \
        "$HOME/miniforge3/bin/conda" \
        "$HOME/miniconda3/bin/conda" \
        "$HOME/anaconda3/bin/conda" \
        /opt/conda/bin/conda; do
        if [[ -x "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

_strip_sim_paths() {
    local remaining="$1" cleaned="" entry is_last
    while :; do
        case "$remaining" in
            *:)
                entry="${remaining%%:*}"
                remaining="${remaining#*:}"
                is_last=false
                ;;
            *:*)
                entry="${remaining%%:*}"
                remaining="${remaining#*:}"
                is_last=false
                ;;
            *)
                entry="$remaining"
                remaining=""
                is_last=true
                ;;
        esac
        case "$entry" in
            *"/_isaac_sim/"* | *"/_isaac_sim" | "") ;;
            *) cleaned="${cleaned:+$cleaned:}$entry" ;;
        esac
        [[ "$is_last" == true ]] && break
    done
    printf '%s' "$cleaned"
}

# The environment changes intentionally live only for the child command.
# shellcheck disable=SC2030,SC2031
_cleaned_command() (
    local clean_pythonpath clean_librarypath
    clean_pythonpath="$(_strip_sim_paths "${PYTHONPATH:-}")"
    clean_librarypath="$(_strip_sim_paths "${LD_LIBRARY_PATH:-}")"
    if [[ -n "$clean_pythonpath" ]]; then
        export PYTHONPATH="$clean_pythonpath"
    else
        unset PYTHONPATH
    fi
    if [[ -n "$clean_librarypath" ]]; then
        export LD_LIBRARY_PATH="$clean_librarypath"
    else
        unset LD_LIBRARY_PATH
    fi
    "$@"
)

# This function deliberately applies the cleaned values to the caller's shell.
# shellcheck disable=SC2031
_strip_active_sim_paths() {
    local clean_pythonpath clean_librarypath
    clean_pythonpath="$(_strip_sim_paths "${PYTHONPATH:-}")"
    clean_librarypath="$(_strip_sim_paths "${LD_LIBRARY_PATH:-}")"
    if [[ -n "$clean_pythonpath" ]]; then
        export PYTHONPATH="$clean_pythonpath"
    else
        unset PYTHONPATH
    fi
    if [[ -n "$clean_librarypath" ]]; then
        export LD_LIBRARY_PATH="$clean_librarypath"
    else
        unset LD_LIBRARY_PATH
    fi
}

init_conda() {
    local conda_executable shell_kind
    conda_executable="$(_find_conda_executable)" || {
        print_error 'Conda was not found. Install Conda or select --uv <name>.'
        return 1
    }
    shell_kind=bash
    [[ -n "${ZSH_VERSION:-}" ]] && shell_kind=zsh
    eval "$("$conda_executable" shell."$shell_kind" hook)" || {
        print_error 'Unable to initialize the Conda shell hook.'
        return 1
    }
    command -v conda >/dev/null 2>&1 || {
        print_error 'Conda shell initialization did not expose conda.'
        return 1
    }
}

target_uv_dir() {
    printf '%s/%s\n' "$MANAENV_UV_ENV_ROOT" "$ENV_NAME"
}

target_env_exists() {
    case "$ENV_BACKEND" in
        conda)
            conda env list 2>/dev/null | awk -v wanted="$ENV_NAME" \
                '$1 == wanted { found = 1 } END { exit !found }'
            ;;
        uv)
            [[ -x "$(target_uv_dir)/bin/python" ]]
            ;;
        python)
            [[ -x "$PYTHON_BIN" ]]
            ;;
    esac
}

activate_target_env() {
    case "$ENV_BACKEND" in
        conda)
            init_conda || return 1
            _strip_active_sim_paths
            conda activate "$ENV_NAME" || {
                print_error "Unable to activate Conda environment: $ENV_NAME"
                return 1
            }
            ;;
        uv)
            local activate_script
            activate_script="$(target_uv_dir)/bin/activate"
            [[ -f "$activate_script" ]] || {
                print_error "uv environment is not ready: $(target_uv_dir)"
                return 1
            }
            _strip_active_sim_paths
            # shellcheck disable=SC1090
            source "$activate_script"
            ;;
        python)
            [[ -x "$PYTHON_BIN" ]] || {
                print_error "Python interpreter is not executable: $PYTHON_BIN"
                return 1
            }
            ;;
    esac
}

target_python() {
    case "$ENV_BACKEND" in
        conda)
            command -v python
            ;;
        uv)
            printf '%s/bin/python\n' "$(target_uv_dir)"
            ;;
        python)
            printf '%s\n' "$PYTHON_BIN"
            ;;
    esac
}

target_activation_hint() {
    case "$ENV_BACKEND" in
        conda) printf 'conda activate %s' "$ENV_NAME" ;;
        uv) printf 'source %s/bin/activate' "$(target_uv_dir)" ;;
        python) printf 'use %s' "$PYTHON_BIN" ;;
    esac
}

_bootstrap_python() {
    if command -v python3 >/dev/null 2>&1; then
        command -v python3
    elif command -v python >/dev/null 2>&1; then
        command -v python
    else
        print_error 'Python is required to prepare publication sources.'
        return 1
    fi
}

prepare_sources() {
    local bootstrap_python
    local -a args
    bootstrap_python="$(_bootstrap_python)" || return 1
    args=(--sources-only)
    args+=("${SOURCE_ARGS[@]}")
    [[ "$SKIP_SOURCE_UPDATE" == true ]] && args+=(--skip-source-update)
    [[ "$DRY_RUN" == true ]] && args+=(--dry-run)
    print_info 'Step 1: Prepare Arena, IsaacLab, and Policy Space sources; install Newton from the X2Real Git pip reference.'
    _run_or_fail 'Publication source preparation failed.' \
        _cleaned_command "$bootstrap_python" "$MANAENV_ROOT/install.py" "${args[@]}"
}

preflight_hardware() {
    if [[ "$SKIP_HARDWARE_CHECK" == true ]]; then
        print_warning 'Skipping NVIDIA GPU/driver checks.'
        return 0
    fi
    command -v nvidia-smi >/dev/null 2>&1 || {
        print_error 'nvidia-smi is unavailable.'
        return 1
    }
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader >/dev/null 2>&1 || {
        print_error 'NVIDIA GPU/driver check failed.'
        return 1
    }
    print_success 'NVIDIA GPU/driver checks passed.'
}

_ask_recreate_environment() {
    [[ "$YES" == true ]] && return 0
    printf "Environment '%s' already exists. Reinstall from scratch? (y/N): " "$ENV_NAME"
    local answer
    IFS= read -r answer || return 1
    [[ "$answer" =~ ^[Yy]$ ]]
}

probe_environment() {
    print_info "Step 2: Check $ENV_BACKEND environment."
    case "$ENV_BACKEND" in
        conda)
            init_conda || return 1
            ;;
        uv)
            command -v uv >/dev/null 2>&1 || {
                print_error 'uv is required for --uv environments.'
                return 1
            }
            ;;
        python)
            target_env_exists || {
                print_error "Python interpreter is not executable: $PYTHON_BIN"
                return 1
            }
            SKIP_CORE=true
            return 0
            ;;
    esac

    if [[ "$SKIP_SOURCES" == true ]]; then
        target_env_exists || {
            print_error "Environment '$ENV_NAME' does not exist; --skip-sources cannot create the simulation stack."
            return 1
        }
        SKIP_CORE=true
        print_info "Reusing existing environment '$ENV_NAME' and installed simulation packages."
        return 0
    fi

    if [[ "$FORCE" == true ]]; then
        SKIP_CORE=false
        print_info "Keeping environment '$ENV_NAME' and force-reinstalling IsaacLab and Arena."
        return 0
    fi

    if ! target_env_exists; then
        SKIP_CORE=false
        print_info "Environment '$ENV_NAME' is missing and will be created by the Arena installer."
        return 0
    fi

    print_warning "$ENV_BACKEND environment '$ENV_NAME' already exists."
    if ! _ask_recreate_environment; then
        SKIP_CORE=true
        print_info "Keeping existing environment '$ENV_NAME'; skipping the core reinstall."
        return 0
    fi

    if [[ "$ENV_BACKEND" == uv ]]; then
        FORCE=true
        SKIP_CORE=false
        print_info "Keeping $(target_uv_dir) and force-reinstalling its core packages."
        return 0
    fi

    _strip_active_sim_paths
    local deactivate_guard=0
    while [[ "${CONDA_DEFAULT_ENV:-base}" == "$ENV_NAME" && "$deactivate_guard" -lt 5 ]]; do
        conda deactivate 2>/dev/null || break
        deactivate_guard=$((deactivate_guard + 1))
    done
    [[ "${CONDA_DEFAULT_ENV:-base}" != "$ENV_NAME" ]] || {
        print_error "Could not deactivate '$ENV_NAME'; it cannot be recreated while active."
        return 1
    }
    _run_or_fail "Failed to remove Conda environment '$ENV_NAME'." \
        conda env remove -n "$ENV_NAME" -y || return 1
    SKIP_CORE=false
}

install_core() {
    [[ -f "$MANAENV_ARENA_INSTALLER" ]] || {
        print_error "Arena installer is missing: $MANAENV_ARENA_INSTALLER"
        return 1
    }
    local -a args
    local uv_http_timeout
    args=("--$ENV_BACKEND" "$ENV_NAME")
    uv_http_timeout="${UV_HTTP_TIMEOUT:-600}"
    [[ "$FORCE" == true ]] && args+=(-f)
    print_info 'Step 3: Install IsaacLab and Arena through the Arena installer.'

    if [[ "$ENV_BACKEND" == conda ]]; then
        local conda_executable conda_base conda_hook
        conda_executable="$(_find_conda_executable)" || {
            print_error 'Conda was not found while preparing the Arena installer.'
            return 1
        }
        conda_base="$(_cleaned_command "$conda_executable" info --base)" || {
            print_error 'Unable to locate the Conda base environment.'
            return 1
        }
        conda_hook="$conda_base/etc/profile.d/conda.sh"
        [[ -f "$conda_hook" ]] || {
            print_error "Conda shell hook is missing: $conda_hook"
            return 1
        }
        _run_or_fail 'Arena core installation failed.' \
            env UV_HTTP_TIMEOUT="$uv_http_timeout" bash -c '
                source "$1"
                arena_installer="$2"
                shift 2
                source "$arena_installer" "$@"
            ' manaenv-arena-installer "$conda_hook" "$MANAENV_ARENA_INSTALLER" "${args[@]}"
        return $?
    fi

    _run_or_fail 'Arena core installation failed.' \
        env UV_HTTP_TIMEOUT="$uv_http_timeout" \
            bash "$MANAENV_ARENA_INSTALLER" "${args[@]}"
}

install_manaenv() {
    activate_target_env || return 1
    local python_bin
    local -a args
    python_bin="$(target_python)" || return 1
    args=(--python "$python_bin" --skip-verify)
    if [[ "$SKIP_SOURCES" == true ]]; then
        args+=(--skip-sources)
    else
        args+=("${SOURCE_ARGS[@]}" --skip-source-update)
    fi
    print_info 'Step 4: Install dependency sources and ManaEnv.'
    _run_or_fail 'ManaEnv installation failed.' \
        _cleaned_command "$python_bin" "$MANAENV_ROOT/install.py" "${args[@]}"
}

verify_install() {
    if [[ "$SKIP_VERIFY" == true ]]; then
        print_warning 'Skipping final import verification.'
        return 0
    fi
    activate_target_env || return 1
    local python_bin
    python_bin="$(target_python)" || return 1
    print_info 'Step 5: Verify publication imports.'
    "$python_bin" -c '
import importlib
import sys
from importlib.metadata import version

installed_isaacsim = version("isaacsim")
if installed_isaacsim != sys.argv[1]:
    raise SystemExit(
        f"Isaac Sim {sys.argv[1]} is required; found {installed_isaacsim}"
    )

modules = (
    "isaacsim",
    "newton",
    "newton_usd_schemas",
    "isaaclab",
    "isaaclab_arena",
    "isaaclab_visualizers.kit",
    "policy_space_protocol",
    "manaenv",
)
for name in modules:
    importlib.import_module(name)
print("ManaEnv publication install: OK")
' "$MANAENV_ISAACSIM_VERSION" || {
        print_error 'Publication import verification failed.'
        return 1
    }
}

print_plan() {
    local target
    if [[ "$ENV_BACKEND" == python ]]; then
        target="$PYTHON_BIN"
    else
        target="$ENV_BACKEND / $ENV_NAME"
    fi
    cat <<EOF
ManaEnv publication install plan
  Environment:          $target
  Default environment:  conda / $MANAENV_DEFAULT_ENV
  Newton install:       pip / X-Square-Robot/Newton@x2real
  Arena branch:         x2real
  IsaacLab branch:      x2real
  Isaac Sim gitlink:    x2real (provenance only)
  Isaac Sim install:    pip / NVIDIA 6.0.1.0
  Policy Space branch:  main (protocol package installed)
  Isaac Sim package:    $MANAENV_ISAACSIM_VERSION (NVIDIA pip index)
  Optional policies:    not installed
EOF
}

_manaenv_install() {
    local original_dir
    original_dir="$(pwd)"

    ENV_BACKEND=conda
    ENV_NAME="$MANAENV_DEFAULT_ENV"
    PYTHON_BIN=""
    FORCE=false
    YES=false
    SKIP_HARDWARE_CHECK=false
    SKIP_SOURCE_UPDATE=false
    SKIP_SOURCES=false
    SOURCES_ONLY=false
    SKIP_VERIFY=false
    DRY_RUN=false
    SKIP_CORE=false
    ENVIRONMENT_EXPLICIT=false
    local -a SOURCE_ARGS=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --conda)
                _require_arg --conda "${2-}" || return 2
                [[ "$ENVIRONMENT_EXPLICIT" == false ]] || {
                    print_error 'Select only one of --conda, --uv, or --python.'
                    return 2
                }
                ENV_BACKEND=conda
                ENV_NAME="$2"
                ENVIRONMENT_EXPLICIT=true
                shift 2
                ;;
            --uv)
                _require_arg --uv "${2-}" || return 2
                [[ "$ENVIRONMENT_EXPLICIT" == false ]] || {
                    print_error 'Select only one of --conda, --uv, or --python.'
                    return 2
                }
                ENV_BACKEND=uv
                ENV_NAME="$2"
                ENVIRONMENT_EXPLICIT=true
                shift 2
                ;;
            --python)
                _require_arg --python "${2-}" || return 2
                [[ "$ENVIRONMENT_EXPLICIT" == false ]] || {
                    print_error 'Select only one of --conda, --uv, or --python.'
                    return 2
                }
                ENV_BACKEND=python
                PYTHON_BIN="$2"
                ENVIRONMENT_EXPLICIT=true
                shift 2
                ;;
            -f | --force)
                FORCE=true
                shift
                ;;
            -y | --yes)
                YES=true
                shift
                ;;
            --arena-url | --isaaclab-url | --isaacsim-url | --policy-space-url)
                _require_arg "$1" "${2-}" || return 2
                SOURCE_ARGS+=("$1" "$2")
                shift 2
                ;;
            --skip-source-update)
                SKIP_SOURCE_UPDATE=true
                shift
                ;;
            --skip-sources)
                SKIP_SOURCES=true
                shift
                ;;
            --sources-only)
                SOURCES_ONLY=true
                shift
                ;;
            --skip-hardware-check)
                SKIP_HARDWARE_CHECK=true
                shift
                ;;
            --skip-verify)
                SKIP_VERIFY=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            -h | --help)
                show_help
                return 0
                ;;
            *)
                print_error "Unknown argument: $1"
                show_help
                return 2
                ;;
        esac
    done

    if [[ "$ENV_BACKEND" != python && ! "$ENV_NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
        print_error "Invalid environment name: $ENV_NAME"
        return 2
    fi
    if [[ "$SKIP_SOURCES" == true && "$SOURCES_ONLY" == true ]]; then
        print_error '--skip-sources and --sources-only cannot be used together.'
        return 2
    fi
    if [[ "$ENV_BACKEND" == python && "$FORCE" == true ]]; then
        print_error '--force cannot be combined with --python.'
        return 2
    fi

    printf '%s\n' '=========================================='
    printf '%s\n' '       ManaEnv Publication Installer'
    printf '%s\n\n' '=========================================='
    print_plan
    printf '\n'

    if [[ "$DRY_RUN" == true ]]; then
        [[ "$SKIP_SOURCES" == true ]] || prepare_sources || return 1
        return 0
    fi

    if [[ "$SKIP_SOURCES" != true ]]; then
        prepare_sources || return 1
    fi
    if [[ "$SOURCES_ONLY" == true ]]; then
        print_success 'Publication source preparation complete.'
        return 0
    fi

    preflight_hardware || return 1
    probe_environment || return 1
    if [[ "$SKIP_CORE" != true ]]; then
        install_core || return 1
    else
        print_info 'Step 3: Reuse the existing simulation core.'
    fi
    install_manaenv || return 1
    verify_install || return 1

    cd "$original_dir" || return 1
    printf '%s\n' '=========================================='
    print_success 'ManaEnv publication installation complete.'
    print_info "Activate later with: $(target_activation_hint)"
    printf '%s\n' '=========================================='
}

_manaenv_install "$@"
_MANAENV_INSTALL_STATUS=$?
_manaenv_install_exit() {
    local exit_code="$1"
    unset _MANAENV_INSTALL_STATUS
    return "$exit_code"
}
_manaenv_install_exit "$_MANAENV_INSTALL_STATUS"
