# Public capability scope

This checkout contains the X2Real public ID evaluator and manual demonstration
collection on the existing public ID scenes.

Manual collection is adapted from the local `dev-6.0` snapshot
`a98b6d93c8817a7139d006983a6691bedcfba4a3`. It includes keyboard input,
Vuer/WebXR controller input, WebSocket master-arm input, direct IK/joint
control, and local HDF5 recording. The public bimanual device contracts cover
ArtiXon Arm-6A and Quanta X1. The formal evaluator retains five boards and
73 ID cases, with its original budgets, scoring, and policy-service boundary.

Automatic planner collection, CuRobo planning, replay, Mimic, dataset cleaning,
LeRobot conversion, training, OOD definitions, general navigation/ROS stacks,
control-plane/web deployment, private device endpoints, and internal asset
locations are excluded. Runtime assets remain a separate local bundle.
No source-branch Git history or third-party gitlink changes are needed for this port.

The regular installer installs `h5py` from `pyproject.toml`. Vuer is selected
explicitly through `.[teleop-vr]`. This repository has no production container
image to update; the optional browser input dependency is installed in the
simulator environment only when selected.

The installer derives explicitly versioned local Isaac Sim core/kernel wheels
from the NVIDIA originals to encode the X2Real dependency profile. The builder
retains runtime payloads and licenses, records upstream hashes, and verifies the
result with raw `pip check`. See the
[installation skill](.agents/skills/x2real-install/SKILL.md#runtime-dependency-profile).

Validation must distinguish catalog/configuration checks, CPU recording tests,
package-content checks, and actual simulator/device execution. A static check
or dry-run cannot establish a working headset, camera, or master-arm session.
