"""Registry package.

Import registry APIs from their defining modules instead of this package
entrypoint. Keeping this file lightweight prevents package import from
bootstrapping discovery or re-exporting legacy compatibility symbols.
"""
