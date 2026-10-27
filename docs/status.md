# Validation status — October 27, 2026

Three independently initialized policies passed the fixed flat-ground command
sequence on ten held-out reset seeds each: **30/30 episodes, 900 seconds, no
falls**. Each episode includes forward motion, left and right turns, and stopping.
These results establish the documented simulation capability, not general
convergence, real-world reliability or hardware readiness.

## Locomotion results

| Training seed | Total PPO updates | Final episodes passed | Falls | Worst segment XY error (m/s) | Worst segment yaw error (rad/s) | Worst stopped RMS speed (m/s) | Worst four-second stop drift (cm) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | 1,200 | 10/10 | 0 | 0.0403 | 0.0836 | 0.0279 | 5.7 |
| 43 | 1,200 | 10/10 | 0 | 0.0577 | 0.0745 | 0.0372 | 9.4 |
| 44 | 1,800 | 10/10 | 0 | 0.0318 | 0.0590 | 0.0250 | 3.5 |

The final reset seeds are 101 through 110 for each policy. Development seeds
7, 11 and 23 were used before the final gate. The threshold definitions and
transition grace periods are in [the fixed protocol](walking-validation.md).
[Machine-readable results](walking-results.json) contain every segment and seed.

The equal-budget result was **2/3 training seeds qualified at 1,200 updates**.
Seed 44 failed the development stop-rotation check: RMS yaw rate was
0.119–0.121 rad/s against a 0.10 rad/s limit. Its final seeds remained unused.
It then resumed from the saved checkpoint for 600 additional updates, without
changing rewards, PPO settings or acceptance thresholds, and passed development
and final tests at 1,800 total updates. This extra training is included in the
table; the initial failed qualification is retained in the experiment records.

The default released policy is seed 42. It was selected after passing its
separate development gate, before final testing. Independent replicas are also
included. Stop commands still produce small residual motion; these tests do
not establish perfectly motionless holding.

[Seed 42 replay](assets/walking-train42.mp4) ·
[Seed 43 replay](assets/walking-train43.mp4) ·
[Seed 44 replay](assets/walking-train44.mp4) ·
[Portable policies](../policies/flat-ground-v1/)

## Training and verification

- All Python, simulation, training, tests and rendering ran on remote NVIDIA
  infrastructure. The Mac was used for editing and orchestration.
- One H200 was used at a time. Each run used 1,024 environments, 24 rollout
  steps, two PPO epochs, four minibatches and fixed learning rate 0.0003.
  Actor and critic each have two hidden layers of 128 units.
- Seeds 42 and 43 each used 29,491,200 transitions. Seed 44 used 44,236,800
  transitions including continuation. Reward and observation definitions were
  unchanged across these runs. The actor has no scripted joint trajectory or
  command-dependent controller switch.
- The initial attempt was cancelled after 17 updates because synchronous
  shared-storage output dominated runtime. Subsequent runs wrote locally and
  mirrored output separately. The final artifact copy was verified.
- Training/save/export took 327.49 s for seed 42, 344.57 s for seed 43, and
  145.27 s for seed 44's additional 600 updates. Scheduler allocation time,
  preparation and video rendering are separate.
- Seven tests passed in 15.89 s. The new benchmark regression test rejects a
  stationary policy even though it does not fall. The existing tests cover
  calibration, native replay, GPU stepping/reset and physics contracts.
- Ruff passed. Generated MJCF was refreshed. Portable policy checksums passed.
- Final training exports were compared with PyTorch on an actual rollout
  observation. The separate checkpoint exporter checked 32 synthetic
  observations per policy, with strict checkpoint loading and finite outputs.
- ONNX replay used native MuJoCo, outside the Warp training environment. Three
  complete videos were decoded and verified: H.264, 640×480, 50 fps, 1,500
  frames and 30 seconds each; each is below 20 MiB. Videos show commands and
  measured body velocity. Frames were inspected for readable overlays and
  visible robot geometry.
- Checkpoint continuation was verified on seed 44: saved iteration index moved
  from 1,199 to 1,799 and every recorded Adam step counter moved from 9,600 to
  14,400. The resume entry now advances past the saved zero-based iteration
  before continuing. [Measured recovery checks](resume-verification.json).
  Full physics state and RNG state are not restored.

## Remaining work

- Physical parameter calibration, measured motor torque/speed/latency and
  robustness to mechanical variation.
- Longer horizons, pushes, slopes, obstacles, expanded commands and deployment
  timing. The current pass rate applies only to this fixed flat-ground test.
- Detailed dynamic trajectory parity across native MuJoCo and MuJoCo Warp;
  current contract tests and ONNX replay provide narrower evidence.
- Smart-servo communication, RK3576 runtime and real-hardware trials.
- IMPO integration is designed but remains deferred.

## Earlier pipeline milestone

Before locomotion training, stand and walk each completed a two-update smoke
run with 32 environments. Those results below are historical pipeline evidence.
The smoke walk policy itself did not learn effective locomotion.

## Historical smoke checks

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
but no saved full physics state or RNG state.

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
