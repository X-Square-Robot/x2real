"""Policy planner family.

Concrete subclasses are loaded lazily through
:mod:`manaenv.pipeline.planners.discovery`; this package intentionally
performs no eager imports so the planner family stays cheap to enumerate
in environments without ``pxr`` / IsaacSim.
"""
