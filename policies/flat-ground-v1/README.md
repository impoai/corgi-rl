# Flat-ground policy v1

This is the exact ONNX actor evaluated by the fixed `flat-ground-v1` protocol.
It was trained from random initialization with seed 42 for 1,200 PPO updates,
1,024 environments and 24 rollout steps: 29,491,200 simulated transitions.
Training used one H200 and the repository's unmodified baseline walk rewards.

The final checkpoint passed all checks for development seeds 7, 11 and 23 before
the final test. It then passed all ten final seeds, 101 through 110, with no
falls in 300 seconds. The test commands 0.2 m/s forward speed, left and right
turns at 0.4 rad/s, and stopping. Stopping is not perfectly motionless: measured
four-second drift ranged from 3.5 to 5.7 cm. These are simulation results.

| File | Contents |
|---|---|
| `policy.onnx` | Deterministic float32 actor with observation normalization |
| `manifest.json` | Robot hash, observation/action order, training config and export provenance |
| `robot-spec.json` | Exact estimated physical specification used for training |
| `checkpoint.pt` | Actor, critic, optimizer, iteration and environment step counter |
| `summary.json` | Measured training duration, update count and export check |
| `checksums.json` | SHA-256 of the portable policy files |
| `replicas/seed-43/`, `replicas/seed-44/` | Independently initialized actors, manifests and checksums |

Seed 43 used 1,200 updates. Seed 44 failed its development stop-rotation gate
at 1,200 updates, then resumed for 600 more before passing final testing.
Both replicas passed all ten final reset seeds. The replica source checkpoints
remain in the training archive; their ONNX actors are bundled here.

The checkpoint does not restore physical environment state or RNG state.
Checkpoint and optimizer continuation was validated on training seed 44, with
the optimizer step count increasing from 9,600 to 14,400. The ONNX input is `[1,45]`; its output
is `[1,12]`. Use the repository's observation construction and action conversion.
Do not apply raw ONNX output directly as joint angles.

Run on the remote NVIDIA infrastructure, with the locked dependencies and
rendering setup from `docs/remote-runtime.md`:

```bash
corgi benchmark --policy policies/flat-ground-v1/policy.onnx \
  --output runs/flat-ground-v1-test --seeds 101 102 103 104 105 106 107 108 109 110 \
  --video-seed 101
```

The robot's masses, geometry, actuator limits and friction are provisional.
This policy has not been tested on motors, RK3576, slopes, obstacles, pushes,
latency changes or hardware parameter randomization. IMPO integration remains
a separate future step. See the [validation protocol](../../docs/walking-validation.md)
and [current status](../../docs/status.md).
