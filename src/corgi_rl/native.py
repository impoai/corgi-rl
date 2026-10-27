"""Native MuJoCo replay of the exported policy contract (run remotely)."""

from pathlib import Path
import numpy as np
import mujoco

from .model import mjcf
from .spec import RobotSpec, write_json


class NativeRobot:
    def __init__(self, spec: RobotSpec):
        self.spec = spec
        self.model = mujoco.MjModel.from_xml_string(mjcf(spec))
        self.data = mujoco.MjData(self.model)
        self.base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.previous_action = np.zeros(12)
        self.reset()

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.previous_action[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def body_velocity(self):
        v = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
                                self.base_id, v, 1)
        return v

    def observation(self, command):
        rot = self.data.xmat[self.base_id].reshape(3, 3)
        return np.concatenate((
            self.body_velocity()[:3], rot.T @ np.array([0., 0., -1.]),
            command, self.data.qpos[7:] - self.spec.vector("home_rad"),
            self.data.qvel[6:] * self.spec.joint_velocity_observation_scale,
            self.previous_action,
        )).astype(np.float32)

    def step(self, action):
        self.data.ctrl[:] = self.spec.targets(action)
        self.previous_action[:] = np.clip(action, -1, 1)
        mujoco.mj_step(self.model, self.data, nstep=self.spec.control_decimation)
        mujoco.mj_forward(self.model, self.data)
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise FloatingPointError("Nonfinite MuJoCo state")
        if any(w.number for w in self.data.warning):
            raise RuntimeError("MuJoCo issued a simulation warning")

    def fallen(self):
        return bool(self.data.qpos[2] < .08 or self.data.xmat[self.base_id, 8] < .5)


def evaluate(spec, *, policy=None, task="stand", seconds=5., seed=42, video=None):
    if seconds <= 0:
        raise ValueError("seconds must be positive")
    robot = NativeRobot(spec)
    if policy:
        import json
        import onnxruntime as ort
        manifest = json.loads(Path(policy).with_name("manifest.json").read_text())
        if manifest["spec_sha256"] != spec.digest:
            raise ValueError("Policy specification hash differs; recalibrate/retrain explicitly")
        session = ort.InferenceSession(str(policy), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        if session.get_inputs()[0].shape != [1, 45] or session.get_outputs()[0].shape != [1, 12]:
            raise ValueError("Expected ONNX [1,45] -> [1,12]")
    else:
        session = None
    command = np.array([.2, 0, 0]) if task == "walk" else np.zeros(3)
    rng = np.random.default_rng(seed)
    robot.data.qpos[7:] += rng.uniform(-.01, .01, 12)
    mujoco.mj_forward(robot.model, robot.data)
    errors, heights, torque = [], [], []
    fell_at = None
    renderer = writer = None
    try:
        if video:
            import imageio.v2 as imageio
            Path(video).parent.mkdir(parents=True, exist_ok=True)
            renderer = mujoco.Renderer(robot.model, height=480, width=640)
            writer = imageio.get_writer(str(video), fps=round(1 / spec.control_dt),
                                       codec="libx264", pixelformat="yuv420p",
                                       ffmpeg_params=["-movflags", "+faststart"])
            camera = mujoco.MjvCamera()
            camera.distance, camera.azimuth, camera.elevation = .85, 135, -22
        for step in range(round(seconds / spec.control_dt)):
            obs = robot.observation(command)
            action = session.run(None, {input_name: obs[None]})[0][0] if session else np.zeros(12)
            robot.step(action)
            errors.append(float(np.linalg.norm(robot.body_velocity()[3:5] - command[:2])))
            heights.append(float(robot.data.qpos[2]))
            torque.append(float(np.abs(robot.data.actuator_force).max()))
            if robot.fallen() and fell_at is None:
                fell_at = (step + 1) * spec.control_dt
            if renderer:
                camera.lookat[:] = robot.data.qpos[:3]
                renderer.update_scene(robot.data, camera=camera)
                writer.append_data(renderer.render())
    finally:
        if writer:
            writer.close()
        if renderer:
            renderer.close()
    result = {
        "backend": "native-mujoco-cpu", "task": task, "seed": seed,
        "policy": str(policy) if policy else "fixed-home-PD (not trained)",
        "duration_s": seconds, "steps": len(errors), "first_fall_s": fell_at,
        "mean_xy_velocity_error_m_s": float(np.mean(errors)),
        "minimum_body_height_m": min(heights), "maximum_torque_nm": max(torque),
        "spec_sha256": spec.digest, "convergence_claim": False,
    }
    if video:
        write_json(Path(video).with_suffix(".json"), result)
    return result
