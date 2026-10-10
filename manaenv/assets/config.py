"""Logical paths for the user-supplied local evaluation asset bundle."""

ASSET_ROOT_URI = "assets"


def join_asset_uri(root: str, suffix: str) -> str:
    if root.endswith("://"):
        return f"{root}{suffix.lstrip('/')}"
    return f"{root.rstrip('/')}/{suffix.lstrip('/')}"


__all__ = [
    "ASSET_ROOT_URI",
    "join_asset_uri",
]
