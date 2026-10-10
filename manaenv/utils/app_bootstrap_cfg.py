"""App bootstrap CFGs consumed before Isaac AppLauncher starts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AppBootstrapCFG:
    """Early process bootstrap options."""

    enable_pinocchio: bool = False
    headless: bool | None = None
    enable_livestream: bool | None = None
