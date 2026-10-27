# Design decisions

## One specification for simulation and deployment

`src/corgi_rl/specs/corgi.json` is the canonical parameter source. The model,
joint mapping, action scaling, control timing and exported policy manifest all
derive from it. A specification hash binds each policy to its robot parameters.

Initial parameters are estimates. The model uses original primitive geometry
so that dimensions, mass and inertia can change as measured hardware data arrives.
The concept image communicates a possible appearance; it does not define geometry,
mass distribution, joint limits or a manufacturing package.

## One training backend

Use the released mjlab 1.6.0 package with MuJoCo/Warp 3.11.0 and rsl_rl 5.4.2.
Install the versions resolved in `uv.lock`, including transitive dependencies.
The source on upstream `main` can have different requirements even when its
version string has not changed.

Use mjlab as an upstream dependency and PPO from rsl_rl. Project code defines
the robot, tasks, observation/action contract, export and validation. It does
not fork the framework or introduce another optimizer.

Training uses Linux and NVIDIA GPUs. Native MuJoCo provides independent replay
and rendering on the remote machine. MJX remains a possible future backend;
there is no second training implementation in this version.

## Start with explicit actuator limitations

The initial actuator is a saturated PD controller with joint limits, armature
and friction. It is sufficient for checking model and software interfaces.
It does not represent a specific supplier's motor.

Add a supplier-specific actuator model only when measurements support it.
Torque-speed curves, voltage effects, latency, backlash, thermal behavior and
sensor noise require calibration before sim-to-real claims.

## Keep deployment separate from training

Export a normalized ONNX policy with a manifest and the exact robot specification.
Native replay checks the specification hash. The future RK3576 runtime must
implement the same observation order, action mapping, reset semantics and timing.

Begin board evaluation with ONNX Runtime CPU inference and measured control-loop
latency. Consider NPU conversion only after numerical parity and SDK operator
support have been checked on the selected board.

## Verify in stages

The repository is currently at the design-review stage. Basic native model tests
have passed; GPU environment and learning checks remain pending. A completed
smoke test will establish that the pipeline runs, not that walking has converged
or that the robot is ready for hardware operation.
