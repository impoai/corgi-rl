"""Fixed, seeded forward/turn/stop evaluation of the exported ONNX policy."""

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

from .native import NativeRobot
from .spec import write_json


# Freeze this protocol before training. Do not tune against the final test seeds.
PROTOCOL = (
    ("settle", 3., (0., 0., 0.)),
    ("forward", 6., (.2, 0., 0.)),
    ("turn_left", 6., (.15, 0., .4)),
    ("forward_again", 3., (.2, 0., 0.)),
    ("turn_right", 6., (.15, 0., -.4)),
    ("stop", 6., (0., 0., 0.)),
)
THRESHOLDS = {
    "maximum_xy_error_m_s": .08,
    "maximum_yaw_error_rad_s": .15,
    "minimum_forward_speed_m_s": .14,
    "minimum_signed_turn_rate_rad_s": .25,
    "maximum_stop_rms_speed_m_s": .04,
    "maximum_stop_rms_yaw_rad_s": .10,
    "maximum_stop_drift_m": .12,
}


def benchmark(spec, *, policy, seeds=(101, 102, 103, 104, 105), output,
              video_seed=None):
    """One uninterrupted 30-second episode per seed; no reset after falls.

    Initial joints vary by +/-0.02 rad and heading by +/-0.15 rad. Each moving
    segment excludes its first second from tracking metrics; stop metrics use
    the final four seconds. All frames still count towards fall detection.
    """
    policy = Path(policy)
    manifest = json.loads(policy.with_name("manifest.json").read_text())
    if manifest["spec_sha256"] != spec.digest:
        raise ValueError("Policy specification differs from the evaluation robot")
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(policy), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    if session.get_inputs()[0].shape != [1, 45] or session.get_outputs()[0].shape != [1, 12]:
        raise ValueError("Expected ONNX [1,45] -> [1,12]")
    input_name = session.get_inputs()[0].name
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    seeds = tuple(seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Supply distinct evaluation seeds")
    if video_seed is not None and video_seed not in seeds:
        raise ValueError("Video seed must be one of the evaluated seeds")
    episodes = []
    for seed in seeds:
        robot = NativeRobot(spec)
        rng = np.random.default_rng(seed)
        robot.data.qpos[7:] += rng.uniform(-.02, .02, 12)
        yaw = rng.uniform(-.15, .15)
        robot.data.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        mujoco.mj_forward(robot.model, robot.data)
        renderer = writer = None
        segments, traces = [], []
        first_fall, step = None, 0
        try:
            if seed == video_seed:
                import imageio.v2 as imageio
                from PIL import Image, ImageDraw, ImageFont
                font = ImageFont.load_default(size=18)
                renderer = mujoco.Renderer(robot.model, height=480, width=640)
                writer = imageio.get_writer(str(out / f"seed-{seed}.mp4"),
                                           fps=round(1 / spec.control_dt),
                                           codec="libx264", pixelformat="yuv420p",
                                           ffmpeg_params=["-movflags", "+faststart", "-threads", "2"])
                camera = mujoco.MjvCamera()
                camera.distance, camera.azimuth, camera.elevation = 1.05, 135, -28
            for name, duration, command_tuple in PROTOCOL:
                command = np.array(command_tuple)
                rows = []
                for index in range(round(duration / spec.control_dt)):
                    obs = robot.observation(command)
                    action = session.run(None, {input_name: obs[None]})[0][0]
                    if not np.isfinite(action).all():
                        raise FloatingPointError("Nonfinite policy action")
                    robot.step(action)
                    step += 1
                    velocity = robot.body_velocity()
                    row = {"time_s": step * spec.control_dt, "segment": name,
                           "segment_time_s": (index + 1) * spec.control_dt,
                           "command": command.tolist(),
                           "velocity": velocity[[3, 4, 2]].tolist(),
                           "position": robot.data.qpos[:3].tolist(),
                           "upright_cosine": float(robot.data.xmat[robot.base_id, 8]),
                           "maximum_torque_nm": float(np.abs(robot.data.actuator_force).max())}
                    rows.append(row)
                    traces.append(row)
                    if robot.fallen() and first_fall is None:
                        first_fall = row["time_s"]
                    if renderer:
                        camera.lookat[:] = robot.data.qpos[:3]
                        renderer.update_scene(robot.data, camera=camera)
                        frame = Image.fromarray(renderer.render())
                        draw = ImageDraw.Draw(frame)
                        draw.rectangle((0, 0, 640, 80), fill=(15, 23, 32))
                        draw.text((12, 8), f"{name.replace('_', ' ').upper()} | {row['time_s']:.1f}s | train {manifest['seed']} / test {seed}", font=font, fill="white")
                        draw.text((12, 34), f"Command: forward {command[0]:.2f} m/s | turn {command[2]:+.2f} rad/s", font=font, fill="#9cd7ff")
                        draw.text((12, 57), f"Actual: {velocity[3]:.2f} m/s | {velocity[2]:+.2f} rad/s | SIMULATION", font=font, fill="white")
                        writer.append_data(np.asarray(frame))
                steady = [r for r in rows if r["segment_time_s"] > (2. if name == "stop" else 1.)]
                velocities = np.array([r["velocity"] for r in steady])
                positions = np.array([r["position"] for r in steady])
                segments.append({
                    "name": name, "command": command.tolist(), "samples": len(steady),
                    "mean_forward_speed_m_s": float(velocities[:, 0].mean()),
                    "mean_yaw_rate_rad_s": float(velocities[:, 2].mean()),
                    "mean_xy_error_m_s": float(np.linalg.norm(velocities[:, :2] - command[:2], axis=1).mean()),
                    "mean_yaw_error_rad_s": float(np.abs(velocities[:, 2] - command[2]).mean()),
                    "rms_xy_speed_m_s": float(np.sqrt(np.square(velocities[:, :2]).sum(1).mean())),
                    "rms_yaw_rate_rad_s": float(np.sqrt(np.square(velocities[:, 2]).mean())),
                    "drift_m": float(np.linalg.norm(positions[-1, :2] - positions[0, :2])),
                })
        finally:
            if writer:
                writer.close()
            if renderer:
                renderer.close()
        moving = [s for s in segments if s["name"] not in ("settle", "stop")]
        stop = segments[-1]
        checks = {
            "no_fall": first_fall is None,
            "xy_tracking": all(s["mean_xy_error_m_s"] <= THRESHOLDS["maximum_xy_error_m_s"] for s in moving),
            "yaw_tracking": all(s["mean_yaw_error_rad_s"] <= THRESHOLDS["maximum_yaw_error_rad_s"] for s in moving),
            "forward_motion": all(s["mean_forward_speed_m_s"] >= THRESHOLDS["minimum_forward_speed_m_s"] for s in moving if s["name"].startswith("forward")),
            "both_turn_directions": all(s["mean_yaw_rate_rad_s"] * np.sign(s["command"][2]) >= THRESHOLDS["minimum_signed_turn_rate_rad_s"] for s in moving if s["name"].startswith("turn")),
            "stop_speed": stop["rms_xy_speed_m_s"] <= THRESHOLDS["maximum_stop_rms_speed_m_s"],
            "stop_rotation": stop["rms_yaw_rate_rad_s"] <= THRESHOLDS["maximum_stop_rms_yaw_rad_s"],
            "stop_drift": stop["drift_m"] <= THRESHOLDS["maximum_stop_drift_m"],
        }
        checks = {k: bool(v) for k, v in checks.items()}
        episode = {"seed": seed, "passed": all(checks.values()), "checks": checks,
                   "first_fall_s": first_fall, "segments": segments,
                   "minimum_body_height_m": min(r["position"][2] for r in traces),
                   "maximum_torque_nm": max(r["maximum_torque_nm"] for r in traces)}
        episodes.append(episode)
        write_json(out / f"seed-{seed}.json", {**episode, "trace": traces})
        print(json.dumps({"seed": seed, "passed": episode["passed"], "checks": checks}), flush=True)
    result = {"protocol": "flat-ground-v1", "protocol_segments": PROTOCOL,
              "thresholds": THRESHOLDS, "policy": str(policy),
              "policy_sha256": hashlib.sha256(policy.read_bytes()).hexdigest(),
              "spec_sha256": spec.digest,
              "training_seed": manifest["seed"], "evaluation_seeds": seeds,
              "duration_per_seed_s": sum(s[1] for s in PROTOCOL),
              "backend": "native-mujoco-cpu-onnx", "episodes": episodes,
              "pass_count": sum(e["passed"] for e in episodes), "count": len(episodes),
              "fall_count": sum(e["first_fall_s"] is not None for e in episodes),
              "hardware_validated": False}
    write_json(out / "benchmark.json", result)
    return result
