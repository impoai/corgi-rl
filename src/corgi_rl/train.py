"""Bounded PPO training, checkpoint, ONNX export and portable run metadata."""

from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.rl.config import RslRlOnPolicyRunnerCfg

from .spec import write_json
from .tasks import make_env_cfg


def train(spec, *, task="stand", output="runs/stand", iterations=1000,
          num_envs=1024, steps=24, seed=42, resume=None):
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires the remote NVIDIA machine; no CPU fallback")
    if min(iterations, num_envs, steps) <= 0 or num_envs * steps < 4:
        raise ValueError("Positive iterations/envs/steps and at least 4 samples required")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(4)
    cfg = RslRlOnPolicyRunnerCfg()
    cfg.seed, cfg.num_steps_per_env = seed, steps
    cfg.logger, cfg.upload_model = "tensorboard", False
    cfg.actor.hidden_dims = cfg.critic.hidden_dims = (128, 128)
    cfg.actor.obs_normalization = cfg.critic.obs_normalization = True
    cfg.actor.distribution_cfg["init_std"] = .3
    cfg.clip_actions = 1.
    cfg.algorithm.num_learning_epochs = 2
    cfg.algorithm.num_mini_batches = 4
    cfg.algorithm.learning_rate = 3e-4
    cfg.algorithm.schedule = "fixed"
    cfg.save_interval = max(1, min(100, iterations))
    env = ManagerBasedRlEnv(make_env_cfg(spec, task, num_envs, seed), device="cuda:0")
    wrapped = RslRlVecEnvWrapper(env, clip_actions=1.)
    runner = MjlabOnPolicyRunner(wrapped, asdict(cfg), log_dir=str(out), device="cuda:0")
    completed_before = 0
    if resume:
        manifest = json.loads(Path(resume).with_name("manifest.json").read_text())
        if manifest["spec_sha256"] != spec.digest or manifest["task"] != task:
            raise ValueError("Resume task/specification differs from the checkpoint")
        runner.load(str(resume))
        # rsl_rl saves the zero-based index of the last completed update.
        runner.current_learning_iteration += 1
        completed_before = runner.current_learning_iteration
    params_before = [p.detach().clone() for p in runner.alg.get_policy().parameters()]
    metadata = {**spec.manifest(), "task": task, "seed": seed, "device": "cuda:0",
                "gpu": torch.cuda.get_device_name(0), "num_envs": num_envs,
                "rollout_steps": steps, "requested_iterations": iterations,
                "completed_iterations_before_run": completed_before,
                "total_requested_iterations": completed_before + iterations,
                "resume": str(resume) if resume else None, "ppo": asdict(cfg),
                "versions": {p: version(p) for p in
                             ("mjlab", "mujoco", "mujoco-warp", "warp-lang", "torch", "rsl-rl-lib")},
                "claim": "training run; capability requires separate evaluation"}
    write_json(out / "manifest.json", metadata)
    write_json(out / "robot-spec.json", spec.model_dump())
    events = out / "events.jsonl"

    def emit(data):
        data["ts"] = datetime.now(timezone.utc).isoformat()
        data["t"] = time.time()
        line = json.dumps(data, allow_nan=False)
        with events.open("a") as f:
            f.write(line + "\n")
        print(line, flush=True)

    original_update = runner.alg.update
    updates = 0
    started = time.monotonic()

    def update_with_event(*args, **kwargs):
        nonlocal updates
        result = original_update(*args, **kwargs)
        updates += 1
        losses = {k: float(v) for k, v in result.items()}
        if not all(np.isfinite(v) for v in losses.values()):
            raise FloatingPointError("PPO returned nonfinite losses")
        objective = (losses["surrogate"] + cfg.algorithm.value_loss_coef * losses["value"]
                     - cfg.algorithm.entropy_coef * losses["entropy"])
        emit({"type": "update", "step": updates,
              "total_completed_iterations": completed_before + updates, "loss": objective,
              "loss_definition": "mean clipped surrogate + value_coef*value - entropy_coef*entropy",
              "losses": losses, "lr": float(runner.alg.learning_rate),
              "transitions": updates * num_envs * steps,
              "elapsed_s": time.monotonic() - started,
              "vram": torch.cuda.max_memory_allocated() / 1e9})
        return result

    runner.alg.update = update_with_event
    try:
        emit({"type": "start", "status": "training", "step": 0,
              "iteration_unit": "one rollout followed by 2 epochs and 4 minibatches"})
        runner.learn(iterations, init_at_random_ep_len=False)
        checkpoint = out / "checkpoint.pt"
        runner.save(str(checkpoint))
        after = list(runner.alg.get_policy().parameters())
        delta = max(float((a - b).abs().max()) for a, b in zip(after, params_before))
        if delta <= 0 or not all(torch.isfinite(p).all() for p in after):
            raise AssertionError("PPO did not make a finite, nonzero policy update")
        runner.export_policy_to_onnx(str(out), filename="policy.onnx")
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(out / "policy.onnx"),
                                       sess_options=options,
                                       providers=["CPUExecutionProvider"])
        policy = runner.get_inference_policy(device="cuda:0")
        obs = wrapped.get_observations()
        with torch.no_grad():
            expected = policy(obs).detach().cpu().numpy()[0:1]
        actual = session.run(None, {session.get_inputs()[0].name:
                                   obs["actor"][0:1].detach().cpu().numpy()})[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
        summary = {"iterations": updates, "transitions": updates * num_envs * steps,
                   "completed_iterations_before_run": completed_before,
                   "total_completed_iterations": completed_before + updates,
                   "max_policy_parameter_change": delta,
                   "onnx_max_abs_error": float(np.abs(actual - expected).max()),
                   "elapsed_s": time.monotonic() - started, "convergence_claim": False}
        write_json(out / "summary.json", summary)
        emit({"type": "done", "step": updates, "output": str(checkpoint), **summary})
        return summary
    finally:
        wrapped.close()
