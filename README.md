![Corgi robot concept render](docs/assets/corgi-concept.jpg)

# Corgi RL

**A small quadruped, built from simulation to hardware.**

Corgi RL is an open-source, low-cost **12-DoF quadruped** project. It brings a
calibratable robot model, stand and walk tasks, and a shared policy interface
into one repository. The planned hardware uses smart servos and a Rockchip
RK3576 board.

*The image above is a Gemini-generated concept render of the proposed appearance.
It is not a photograph, a validated mechanical design, or the current simulator model.*

**Status: initial repository, ready for design review.** PPO training has not
started. Walking and hardware operation have not been validated. Initial
dimensions, masses, and actuator parameters are estimates.

## Architecture

```mermaid
flowchart LR
    S[Shared robot specification] --> M[Generated MJCF]
    M --> E[MuJoCo Warp + mjlab: stand and walk]
    E --> P[PPO training]
    P --> C[Checkpoint + ONNX + manifest]
    C --> V[Native MuJoCo replay and video]
    S --> V
    C -. Planned .-> H[RK3576 + smart servos]
    S -. After calibration .-> H
```

Each leg has three actuated joints: hip abduction, hip pitch, and knee pitch.
The current simulation model uses original boxes, capsules, and spheres.
The proposed shell and corgi-like appearance are a separate design step.

| Path | Purpose |
|---|---|
| `src/corgi_rl/specs/corgi.json` | Canonical robot parameters in SI units |
| `src/corgi_rl/spec.py` | Validation, specification hash, action and hardware coordinate conversion |
| `src/corgi_rl/model.py`, `models/` | MJCF generator and generated model |
| `src/corgi_rl/tasks.py` | Standing and velocity-tracking tasks |
| `src/corgi_rl/train.py` | PPO, checkpoints, ONNX export, and export error checks |
| `src/corgi_rl/native.py` | Native MuJoCo replay, metrics, and H.264 video |
| `tests/` | Specification, model, fixed-PD standing, and pending GPU tests |
| `hardware/` | Calibration plan and supplier data requirements |
| `deploy/` | RK3576 interface, remote smoke test, and Slurm template |
| `docs/design.md` | Technical choices and implementation boundaries |
| `docs/status.md` | Completed checks, known issues, and pending work |
| `docs/concept-render.md` | Concept image provenance and generation prompt |
| `uv.lock` | Complete dependency lock |

## Verified so far

Checks ran on October 27, 2026, inside the remote cluster's PyTorch container:

- **Four basic tests passed:** model/specification consistency, invalid input
  rejection, three seconds of fixed-PD standing, and repeatable seeded replay.
- The model compiles to `nq=19`, `nv=18`, and `nu=12`: a floating base and
  12 actuated joints.
- The standing test checks fall status, body height, and the 2.5 N·m torque limit.
- Dependency locking and installation completed. The MJCF was generated.
- Static checks passed, and the stand configuration was constructed.

GPU stepping and resets, PPO updates, checkpoint recovery, ONNX numerical
checks, video rendering, and hardware tests remain pending. An implemented
entry point is not a passed test. Training is on hold for repository review.

## Stack

The project uses released **mjlab 1.6.0**, **MuJoCo/MuJoCo Warp 3.11.0**, and
**rsl_rl 5.4.2 PPO**, with PyTorch 2.9.1 and Warp 1.14.0. These are upstream
package dependencies. The project does not maintain a training-framework fork.

Native MuJoCo provides remote CPU replay and cross-checks. Parallel training
uses Linux and NVIDIA GPUs. `uv.lock` pins direct and transitive dependencies.
MJX is a possible future backend; the current implementation has one training path.

## Run on the remote machine

Python environments, package installation, simulation, training, and tests run
on the remote NVIDIA environment. The development Mac is for source editing.
Do not execute the following installation or runtime commands on that Mac.

The commands below are for use after review. **The training commands have not
been run.** Use Linux x86_64, Python 3.11, an NVIDIA GPU, and a working EGL setup.
Private cluster addresses, credentials, and access instructions are kept outside
this repository.

```bash
# Prepare dependencies on the remote login node before allocating a GPU.
uv sync --locked --python 3.11
uv run --locked corgi model
uv run --locked pytest tests/test_spec_and_native.py -q

# On an allocated NVIDIA GPU, check the environment before a bounded PPO run.
uv run --locked pytest tests/test_gpu.py -q
uv run --locked corgi train --task stand --output runs/stand-smoke \
  --num-envs 32 --steps 24 --iterations 2
uv run --locked corgi train --task walk --output runs/walk-smoke \
  --num-envs 32 --steps 24 --iterations 2

# Replay the exported policy and save metrics with an H.264 video.
MUJOCO_GL=egl uv run --locked corgi evaluate --task walk \
  --policy runs/walk-smoke/policy.onnx --seconds 5 --video runs/walk-smoke.mp4
```

Use a new output directory for each training run. To resume, pass
`--resume runs/stand-smoke/checkpoint.pt --output runs/stand-resume`.
The iteration count specifies additional iterations. Resume remains untested;
exact reproduction of random-number and environment state is not guaranteed.

If the container has no Git executable, set `GIT_PYTHON_REFRESH=quiet` before
importing the training stack. The Slurm smoke script sets it. Record the source
revision on the submission host. Other images need dependency and driver checks.

`deploy/slurm-smoke.sbatch` expects prepared source and environment archives.
The repository has no automation that submits jobs or starts training.

## Shared calibration contract

- Joint order: `FL, FR, RL, RR`, with `hip, thigh, knee` for each leg.
- Physics step: 4 ms. Policy action every five steps: **50 Hz**.
- Actor observation: 45 values — body angular velocity (3), gravity direction
  (3), velocity command (3), joint position relative to home (12), scaled joint
  velocity (12), and previous clipped action (12).
- Action: 12 values, clipped to `[-1,1]`, mapped to `home + scale × action`,
  then limited to each joint's angle range. Actuators use torque-limited PD control.
- The ONNX manifest records the specification SHA-256, joint order, units,
  and control rate. Replay checks the specification hash.
- Each joint supports calibration of direction, zero, limits, gains, torque,
  armature, and friction. Each body can receive measured mass, center of mass,
  and diagonal inertia.

Voltage sag, torque-speed curves, latency, backlash, thermal effects, and sensor
noise are not modeled yet. The initial walk task tracks commanded velocity.
Gait quality, rewards, and randomization need evaluation through actual training.
See the [calibration plan](hardware/README.md) and [deployment interface](deploy/README.md).

## Next steps

1. Review the robot specification, interfaces, and repository structure.
2. Run environment tests on one H200 and fix any issues they expose.
3. Run bounded PPO, checkpoint, ONNX, and rendering checks with recorded results.
4. Obtain and calibrate real actuator and mechanical parameters before larger runs.

## License

Project source and original primitive geometry use [Apache-2.0](LICENSE).
Runtime dependencies retain their respective licenses. See [NOTICE](NOTICE).
