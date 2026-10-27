# Validation status — October 27, 2026

Current delivery: a repository for design review. No GPU job has been submitted,
and PPO training has not started.

## Completed checks

- Installed an isolated environment on the remote login node inside
  `pytorch_291_cu128.sif`.
- Python 3.11.14; `uv lock` and `uv sync --locked` completed.
- `ruff check .` passed. The stand configuration object was constructed without
  starting its environment or training.
- `pytest tests/test_spec_and_native.py -q`: **4 passed in 3.85 seconds**.
- Generated and compiled the native MuJoCo model: 19 generalized positions,
  18 generalized velocities, and 12 actuators.
- Fixed-target PD control maintained standing for three seconds. This is a
  controller check, not a learned standing policy.

## Pending checks

- GPU stepping, resets, and fall termination for stand/walk in `tests/test_gpu.py`.
- Observation, action, and physics consistency between mjlab and native MuJoCo.
- PPO parameter updates, finite losses, checkpoint saving, and recovery.
- ONNX export, normalization, numerical error, and specification mismatch rejection.
- EGL rendering and H.264 video encoding.
- Hardware control, servo communication, RK3576 latency, and sim-to-real trials.

## Known compatibility issue

The container has no Git executable. GitPython raises an import error while
loading mjlab tasks with its default settings. The remote smoke script sets
`GIT_PYTHON_REFRESH=quiet`; the submission host must record the source revision.
Full GPU tests are still needed to validate the training path.

The released mjlab 1.6.0 package uses MuJoCo/Warp 3.11.0 and rsl_rl 5.4.2.
Upstream `main` has different dependency requirements. Install from this
repository's `uv.lock`.

## Next acceptance run

After repository review, run the GPU tests, then two PPO iterations for each
of stand and walk. Record the device, parallel environment count, actual
transition count, parameter changes, losses, export error, and replay metrics.
Expand training only after those checks pass. A short run does not establish
convergence or hardware readiness.
