# Reference review — 2026-10-27

This is a source review, not a certification of a complete hardware design or
an assessment of whether one project copied another.

| Source | Inspected revision | Findings used here |
|---|---|---|
| [Jumper](https://github.com/KingKongRobotics/jumper) | `b2e58669848a3ba4f209597cd6bebdfe9d0533a4` | Currently a 22-DoF crab robot. Uses MuJoCo, a modified vendored mjlab, rsl_rl, native CPU and Warp backends, policy export and deploy bundles. Its hardware page identifies RK3576. |
| [Microduck runtime](https://github.com/pollen-robotics/microduck) | `5def2f3e66501f0f365d4fb1183020b8ee875443` | Biped runtime separated from training. The repository is software, not a complete production package for our quadruped. |
| [Microduck RL](https://github.com/pollen-robotics/microduck_rl) | `bae39f5900b318f7e87fc485c30a31213cfa14eb` | mjlab/PPO, ONNX policy contract, actuator modeling, calibration and domain randomization. Different robot, observation dimensions and actuation. |

Jumper's [LICENSE](https://github.com/KingKongRobotics/jumper/blob/b2e58669848a3ba4f209597cd6bebdfe9d0533a4/LICENSE)
is Apache-2.0 for maintainer-owned materials. Its
[NOTICE](https://github.com/KingKongRobotics/jumper/blob/b2e58669848a3ba4f209597cd6bebdfe9d0533a4/NOTICE)
lists third-party components with their own licenses; the Rockchip runtime SDK
is separate. A top-level license does not relicense those components.

Microduck [software](https://github.com/pollen-robotics/microduck/blob/5def2f3e66501f0f365d4fb1183020b8ee875443/LICENSE)
and [RL code](https://github.com/pollen-robotics/microduck_rl/blob/bae39f5900b318f7e87fc485c30a31213cfa14eb/LICENSE)
are Apache-2.0. Its [model license](https://github.com/pollen-robotics/microduck_rl/blob/bae39f5900b318f7e87fc485c30a31213cfa14eb/LICENSE-MODELS.md)
covers the named STL directories under CC BY-NC-SA-4.0. The file explicitly leaves
MJCF XML/configs/part metadata under Apache-2.0. This project imports none of them.

## Decisions

- Use the **released** mjlab 1.6.0 package with MuJoCo/Warp 3.11.0 and rsl_rl
  5.4.2. Upstream `main` has different dependency requirements even though its
  version string can still say 1.6.0. `uv.lock` is the resolution of record.
- Use upstream mjlab as a dependency. Do not inherit Jumper's framework fork or
  robot-specific control assumptions.
- Use PPO's upstream implementation. Our code defines the robot, task, contract
  and bounded validation; it does not introduce another optimizer.
- Train on Linux/NVIDIA. See the [mjlab installation guide](https://mujocolab.github.io/mjlab/v1.6.0/source/installation.html).
  Use native MuJoCo only for independent replay/rendering on the remote machine.
- Model our 12-DoF quadruped with original boxes, capsules and spheres.
  No source meshes, CAD, proprietary SDKs or third-party policy weights are copied.
- Defer BAM or a supplier-specific motor model until measured data supports the
  choice. An ideal saturated PD actuator is a declared starting approximation.
