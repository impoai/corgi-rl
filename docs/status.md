# Validation status — October 27, 2026

The bounded smoke pipeline passed on one NVIDIA H200. Stand and walk each ran
two PPO iterations with 32 environments and 24 rollout steps: 1,536 transitions
per task. This is pipeline validation, not convergence or hardware readiness.

## Completed checks

- Python 3.11.14 inside the remote PyTorch 2.9.1 CUDA 12.8 container.
  NVIDIA driver 575.57.08; H200 memory reported as 143,771 MiB.
- Locked Python dependencies installed on the login node. The environment and
  source ran from compute-node local storage. No project Python ran on the Mac.
- `ruff check .` passed. Shell scripts passed syntax checks.
- Six tests passed in 16.30 seconds. Four cover model/calibration validation,
  invalid inputs, three seconds of fixed-PD standing and repeatable replay.
  Two cover GPU stand/walk stepping, finite rewards/observations, timeout reset,
  forced fall termination and initial observation parity.
- GPU and native model tests compare timestep, integrator, solver and iteration
  limits, contact cone, ground friction, actuator gains and torque limits.
- Model dimensions: 19 generalized positions, 18 velocities and 12 actuators.
- Both PPO runs produced finite losses and nonzero policy parameter updates.
  Actor and critic each use two hidden layers of 128 units. Learning rate is
  fixed at 0.0003. Each iteration runs two epochs with four minibatches.
- Both runs saved a PPO checkpoint, a `[1,45] -> [1,12]` ONNX policy,
  normalization, robot specification and manifest.
- Export comparisons used one actual final-rollout observation per task.
- Each three-second replay contains 150 H.264 frames at 50 fps and 640×480.
  Physics replay and OSMesa rendering used CPUs on the remote compute node.

| Measurement | Stand | Walk |
|---|---:|---:|
| Iteration 1 objective loss | 0.211479 | 0.189230 |
| Iteration 2 objective loss | 0.161039 | 0.134639 |
| Maximum policy parameter change | 0.00427974 | 0.00440379 |
| ONNX/PyTorch maximum absolute error | 1.00583e-7 | 8.94070e-8 |
| Training, save and export elapsed | 5.828 s | 5.616 s |
| Replay minimum body height | 0.213489 m | 0.212461 m |
| Replay maximum actuator torque | 1.032644 N·m | 1.292154 N·m |
| Replay mean XY velocity error | 0.008624 m/s | 0.205536 m/s |
| First fall in three seconds | None | None |

The walk command is 0.2 m/s forward. The error is about that magnitude, so this
run does not demonstrate learned walking. A three-second no-fall result also
does not establish robust standing.

A second remote compute job validated the explicit ONNX thread settings on
16 sequential observations per task: one-thread and two-thread outputs matched
exactly. Both exported policies rejected a changed robot specification. Repeated
three-second replay metrics matched the original run. All five main/postflight
videos were decoded and verified as 150 H.264 frames at 50 fps; each is below
20 MiB. Checkpoint files contain actor, critic, optimizer and iteration state,
but no saved environment or RNG state.

## Compatibility fixes

- Use mjlab 1.6.0's `write_root_link_pose_to_sim` in the forced-fall test.
- Keep scene physics settings in `MujocoCfg`; omit ignored scene options from
  the attached robot-only MJCF. Match exact joint names for XML actuators.
- Configure terrain friction from the shared robot specification.
- Respect the selected `MUJOCO_GL` backend. The tested container/node lacked
  EGL libraries and a usable NVIDIA EGL vendor library. OSMesa libraries are
  pinned by Debian package version and SHA-256 in `deploy/`.
- Set ONNX Runtime's CPU thread counts explicitly to avoid automatic affinity
  binding outside the Slurm allocation's CPU set.
- Use a compute-valid launch directory and local Warp cache. This cluster's
  batch launcher failed during user-environment retrieval; `srun` succeeded.
- Set `GIT_PYTHON_REFRESH=quiet` when the container lacks Git. The submission
  host records the source revision and file hashes.

See [remote runtime preparation](remote-runtime.md) for the reproducible setup.
The released packages are mjlab 1.6.0, MuJoCo/MuJoCo Warp 3.11.0, Warp 1.14.0,
rsl_rl 5.4.2 and PyTorch 2.9.1. Upstream development branches can differ.

## Pending checks

- Checkpoint recovery, including exact RNG/environment restoration semantics.
- Dynamic trajectory parity across native MuJoCo and MuJoCo Warp. The current
  contract test covers initial observations and selected physical parameters.
- Learned gait quality, longer horizons, disturbances, multiple seeds, and
  calibrated actuator latency, torque-speed behavior and mechanical variation.
- Actual hardware communication, RK3576 latency and sim-to-real trials.

No long training run was started. Future training requires a separate decision
after reviewing these results and the provisional hardware parameters.
