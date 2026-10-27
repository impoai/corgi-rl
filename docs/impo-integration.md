# IMPO integration design

Corgi's product direction combines a local motion policy with the existing
IMPO agent and gadget platform. This integration is a planned part of the
architecture. The current delivery validates simulation, PPO, policy export
and native replay; it does not yet connect a robot or simulator to IMPO.

## Two layers

| Layer | Responsibility | Current status |
|---|---|---|
| Local motion policy | Turn a bounded velocity command and robot observations into joint targets at 50 Hz | Simulation and ONNX smoke-tested; RK3576 execution pending |
| IMPO integration | Expose robot capabilities to the user's agent, receive tasks and return device events | Design only; adapter and product manifest pending |

IMPO already has a [Linux Gadget SDK](https://github.com/impoai/impo-gadget-sdk/tree/main/linux)
and a [developer workflow](https://github.com/impoai/impo-developer-skills)
for products, manifests and development devices. A product manifest describes
commands, events and goal-specific scenarios. Corgi should use these existing
contracts instead of introducing a separate agent protocol. Validate Linux
SDK support on the selected RK3576 system before committing to its transport.

## Proposed boundary

The device adapter accepts high-level requests such as stand, request a bounded
velocity, stop and report status. These are proposed capabilities; their final
command names and schemas must follow the current IMPO product contract.
IMPO does not participate in the 50 Hz joint-control loop.

The local controller validates each request, enforces motion limits and command
expiry, and selects a policy that matches the shared robot specification.
Its observation and action order must match the exported policy manifest.
Network loss must not interrupt the control loop; expired commands must cause
a locally defined safe stop. The hardware runtime still needs to implement and
test this behavior.

Device events can report the active mode, task progress, faults, battery state
and policy/specification identity. Available sensors and measurements determine
which events the first hardware version can support.

The IMPO developer API key belongs on development infrastructure. Each device
uses its own pairing credentials. No API key, pairing token or SSH credential
belongs in the public repository or a policy artifact.

## First integration milestone

Connect the simulator as an IMPO development gadget before connecting motors.
Validate command delivery, status events, rejected commands, expiry and
disconnect behavior against the same adapter contract. This requires an
explicit integration implementation and test run; the current PPO smoke test
does not validate IMPO connectivity.

Source contracts were checked on October 27, 2026. The SDK and developer APIs
can evolve; re-check them before implementing the adapter.
