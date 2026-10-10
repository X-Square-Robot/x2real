"""manaenv-side actuator implementations.

manaenv previously shipped a warp-based Tan-Liu-Turk 2011 Stable PD
actuator here. It has been removed in favour of Isaac Lab's official
:class:`isaaclab.actuators.StablePDActuatorCfg`, which — with
``SimulationCfg.use_newton_actuators=True`` — routes to Newton's implicit
``ControllerStablePD`` while the simulation physics backend remains PhysX.
See the EX001 embodiment configs for the actuator
wiring and :mod:`manaenv.actuators.newton_compat` for the required target-name
compatibility layer.

The package intentionally exports no convenience symbols.
"""

__all__: list[str] = []
