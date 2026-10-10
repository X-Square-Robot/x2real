import logging
import os
import re
import sys
import time
from typing import Literal

# Add a global switch for enabling/disabling debug printing
DEBUG_PRINT_ENABLED = True


_T0 = time.perf_counter()

_DIST_IMAGE_INFO = "/etc/manaenv-image-info"
_NOISY_THIRD_PARTY_LOGGERS = (
    "asyncio",
    "AutoNode",
    "h5py",
    "idl.connection.ws",
    "matplotlib",
    "omni.discovery",
    "trimesh",
    "urllib3",
    "websockets",
    "websockets.client",
)


def _runtime_image_variant() -> str:
    variant = os.environ.get("MANAENV_IMAGE_VARIANT", "").strip().lower()
    if variant:
        return variant
    try:
        with open(_DIST_IMAGE_INFO, encoding="utf-8") as f:
            for line in f:
                key, sep, value = line.partition("=")
                if sep and key.strip() == "variant":
                    return value.strip().lower()
    except OSError:
        return ""
    return ""


def _should_quiet_third_party_debug() -> bool:
    override = os.environ.get("MANAENV_QUIET_THIRD_PARTY_DEBUG", "").strip().lower()
    if override in {"1", "true", "yes", "on"}:
        return True
    if override in {"0", "false", "no", "off"}:
        return False
    if os.environ.get("MANAENV_ENABLE_THIRD_PARTY_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}:
        return False
    return _runtime_image_variant() == "dist"


def _quiet_third_party_debug_logs() -> None:
    """Keep dist benchmark stdout focused on ManaEnv/WallX evidence.

    IsaacSim and its Python dependencies can set the root logger to DEBUG,
    which floods Ray stdout with websocket, urllib3, h5py, and Omni tagging
    internals. Dist images are used for benchmark/PAI runs, so default them to
    INFO+ while leaving dev mode unchanged.
    """
    logging.getLogger().setLevel(logging.INFO)
    for name in _NOISY_THIRD_PARTY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


_THIRD_PARTY_NOISE_PATTERNS = (
    # Isaac Sim's warp re-uses USD primitives across parallel envs; one warning
    # per (env, prim) per run floods the log without flagging real issues.
    re.compile(r"Object already in warp cache, using existing instance"),
    # cuRobo emits this every time the IK solver flips between solve_single /
    # solve_batch / solve_goalset (i.e. every replan in batched skill runs);
    # the cuda graph reset itself is benign in our usage.
    re.compile(r"changing goal type, breaking previous cuda graph"),
    re.compile(r"Mesh faces .* are not matching faceVertexCounts"),
)

_NOISE_FILTER: logging.Filter | None = None


class _ThirdPartyNoiseFilter(logging.Filter):
    """Drop a curated list of high-frequency third-party log lines.

    Both upstreams emit these via the stdlib ``logging`` module under the
    ``logger.py`` filename, with no actionable signal for our pipeline. We
    keep them locally suppressed instead of touching the third-party packages
    so an upstream upgrade simply re-imports cleanly.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True
        return not any(p.search(message) for p in _THIRD_PARTY_NOISE_PATTERNS)


def install_third_party_noise_filter() -> None:
    """Attach the curated third-party noise filter to the root logger.

    Isaac Lab's :func:`isaaclab.utils.logger.configure_logging` replaces all
    root *handlers* when the env starts; handler-only filters are lost. A
    logger-level filter survives that reset and suppresses high-frequency cuRobo
    / warp lines regardless of which handler prints them.
    """
    global _NOISE_FILTER
    if _NOISE_FILTER is None:
        _NOISE_FILTER = _ThirdPartyNoiseFilter()
    root = logging.getLogger()
    if _NOISE_FILTER not in root.filters:
        root.addFilter(_NOISE_FILTER)
    for handler in root.handlers:
        if _NOISE_FILTER not in handler.filters:
            handler.addFilter(_NOISE_FILTER)


def _configure_stdlib_logging():
    """Set a timestamped format on the stdlib root logger so any
    ``logging.getLogger(__name__)`` calls across the codebase also carry
    wall-clock + elapsed timestamps."""
    fmt = logging.Formatter(
        fmt="%(asctime)s +%(elapsed).3fs [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    class _ElapsedFilter(logging.Filter):
        def filter(self, record):
            record.elapsed = time.perf_counter() - _T0
            return True

    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        root.addHandler(handler)
    install_third_party_noise_filter()
    for h in root.handlers:
        h.setFormatter(fmt)
        if not any(isinstance(f, _ElapsedFilter) for f in h.filters):
            h.addFilter(_ElapsedFilter())
    if _should_quiet_third_party_debug():
        _quiet_third_party_debug_logs()


_configure_stdlib_logging()


class DebugPrinter:
    COLORS = {
        "DEBUG": "\033[94m",  # Blue
        "INFO": "\033[97m",  # White
        "WARNING": "\033[93m",  # Yellow
        "ERROR": "\033[91m",  # Red
        "CRITICAL": "\033[95m",  # Magenta
        "SUCCESS": "\033[92m",  # Green for success messages
        "RESET": "\033[0m",  # Reset
    }

    DIM = "\033[2m"

    def __init__(self):
        self.levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "SUCCESS"]

    def emit(self, message: str, level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "SUCCESS"] = "INFO"):
        """
        Emit a message with a specific debug level and color.

        Args:
            message (str): The message to emit.
            level (str): The debug level (DEBUG, INFO, WARNING, ERROR, CRITICAL, SUCCESS).
        """
        if not DEBUG_PRINT_ENABLED:
            return

        if level not in self.levels:
            raise ValueError(f"Invalid debug level: {level}")

        color = self.COLORS.get(level, self.COLORS["RESET"])
        reset = self.COLORS["RESET"]
        elapsed = time.perf_counter() - _T0
        wall = time.strftime("%H:%M:%S")
        ts = f"{self.DIM}{wall} +{elapsed:9.3f}s{reset} "
        sys.stderr.write(f"{ts}{color}[{level}] {message}{reset}\n")
        sys.stderr.flush()

    def add_level(self, level: str, color_code: str):
        """
        Add a new debug level with a specific color.

        Args:
            level (str): The name of the new debug level.
            color_code (str): The ANSI color code for the level.
        """
        self.levels.append(level)
        self.COLORS[level] = color_code


debug_printer = DebugPrinter()


def manaprint(message: str):
    """
    Simplified function to emit debug messages directly. Automatically detects levels based on message format.

    Args:
        message (str): The message to emit. If no level prefix is provided, defaults to INFO.
    """
    if not DEBUG_PRINT_ENABLED:
        return

    # Dynamically build the regex pattern based on available levels
    available_levels = "|".join(debug_printer.levels)
    match = re.match(rf"^({available_levels}):", message, re.IGNORECASE)

    if match:
        level = match.group(1).upper()
        # Remove the level prefix from the message
        message = message[len(match.group(0)) :].strip()
    else:
        level = "INFO"  # Default level

    # Emit the message with the detected or default level
    debug_printer.emit(message, level)


# def main():
#     """
#     Main function to demonstrate and configure debug printing.
#     """
#     global DEBUG_PRINT_ENABLED

#     parser = argparse.ArgumentParser(description="Configure debug printing.")
#     parser.add_argument("--enable-debug", action="store_true", help="Enable debug printing.")
#     parser.add_argument("--disable-debug", action="store_true", help="Disable debug printing.")
#     args = parser.parse_args()
#     if args.enable_debug:
#         DEBUG_PRINT_ENABLED = True
#     elif args.disable_debug:
#         DEBUG_PRINT_ENABLED = False

#     # Example usage
#     manaprint("This is a message without a prefix, so it defaults to INFO.")
#     manaprint("DEBUG: This is a debug message")
#     manaprint("INFO: This is an info message")
#     manaprint("WARNING: This is a warning message")
#     manaprint("ERROR: This is an error message")
#     manaprint("CRITICAL: This is a critical message")
#     manaprint("SUCCESS: This is a success message")

#     # Adding a custom level
#     debug_printer.add_level("CUSTOM", "\033[96m")  # Cyan
#     manaprint("CUSTOM: This is a custom message")
#     manaprint("INFO: Debug printing is now enabled." if DEBUG_PRINT_ENABLED else "INFO: Debug printing is now disabled.")

# if __name__ == "__main__":
#     main()
