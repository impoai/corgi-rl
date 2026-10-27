# Flat-ground walking validation

This protocol was fixed before evaluating locomotion candidates and was in
place before the completed training runs. It tests the
exported ONNX policy in native MuJoCo, outside the GPU training environment.
The actor receives the same 45-value observation and produces 12 joint targets.

Each episode lasts 30 seconds without resets: settle for 3 seconds, move forward
at 0.20 m/s for 6 seconds, turn left at 0.15 m/s and +0.40 rad/s for 6 seconds,
move forward for 3 seconds, turn right at 0.15 m/s and -0.40 rad/s for 6 seconds,
then stop for 6 seconds. Initial joint positions vary by +/-0.02 rad and heading
by +/-0.15 rad. Falls count throughout the episode. Tracking metrics exclude
the first second of each segment; stopping metrics use the final four seconds.

Each seed must pass all checks:

- No fall: body height stays above 0.08 m and tilt stays below 60 degrees.
- Mean XY velocity error at most 0.08 m/s in every moving segment.
- Mean yaw-rate error at most 0.15 rad/s in every moving segment.
- Mean forward speed at least 0.14 m/s in both straight segments.
- Mean turn rate at least 0.25 rad/s in the requested direction, both ways.
- Stopped RMS XY speed at most 0.04 m/s; RMS yaw rate at most 0.10 rad/s.
- Stopped displacement at most 0.12 m over the measured four seconds.

Development seeds are 7, 11 and 23. Final evaluation seeds are 101 through 110.
Training seeds and evaluation seeds are reported separately. Repeated initial
states test policy robustness; they do not establish independent training
reproducibility. Final seeds must not be used to tune rewards or select weights.

```bash
corgi benchmark --policy runs/walk/policy.onnx --output runs/walk-dev \
  --seeds 7 11 23
corgi benchmark --policy runs/walk/policy.onnx --output runs/walk-test \
  --seeds 101 102 103 104 105 106 107 108 109 110 --video-seed 101
```

This is a flat-ground simulation test. It does not validate physical actuator
models, uneven terrain, disturbance recovery, deployment timing or real motors.
