# RK3576 deployment boundary

Target: a future Linux ARM64 Rockchip RK3576 board. No board driver or motor
bus is implemented. This directory is a specification and staged plan.

The planned IMPO device adapter passes bounded motion goals to this local
runtime and returns device events. See the [IMPO integration design](../docs/impo-integration.md).
The network-facing adapter does not drive the 50 Hz joint-control loop.

Each training run produces `policy.onnx`, `manifest.json`, `robot-spec.json`
and a resumable PPO checkpoint. The ONNX graph includes observation
normalization. The actor accepts float32 `[1,45]` and emits `[1,12]`.
The manifest fixes joint order, observation layout, units, control rate and the
robot specification SHA-256. Native replay refuses a mismatched specification.

The policy uses body angular velocity, gravity direction, commanded velocity,
joint position relative to home, scaled joint velocity and previous clipped
action. It does not need privileged ground-truth translational velocity.

At 50 Hz, construct that observation, run inference, clip actions, convert to
joint position targets and apply the measured motor sign/zero conversion.
Match the simulator's action and reset semantics exactly.

First board target: ONNX Runtime CPU inference, measured for latency and jitter.
RKNN/NPU export comes only after numerical parity and operator support are
checked on the selected SDK. There is no assumed NPU speed-up or bundled SDK.

Before enabling a real bus implement and test: explicit arming, command
watchdog, emergency stop, voltage/current/temperature limits, position and
velocity guards, safe fault behavior and a calibrated IMU transform. Keep an
independent MCU or motor firmware safety path if the supplier supports one.
Benchmark the whole sensor-to-command loop under load; model-only latency is
not a control-loop timing test.
