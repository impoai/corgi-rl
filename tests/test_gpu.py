import numpy as np
import pytest
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper

from corgi_rl.native import NativeRobot
from corgi_rl.spec import load_spec
from corgi_rl.tasks import make_env_cfg

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("task", ["stand", "walk"])
def test_mjlab_steps_resets_and_observation_contract(task):
    assert torch.cuda.is_available(), "Run this suite on the remote NVIDIA host"
    spec = load_spec()
    cfg = make_env_cfg(spec, task, num_envs=4)
    cfg.episode_length_s = spec.control_dt * 4
    cfg.events["reset_joints"].params["position_range"] = (0., 0.)
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    wrapped = RslRlVecEnvWrapper(env, clip_actions=1.)
    try:
        obs = wrapped.get_observations()
        assert obs["actor"].shape == (4, 45)
        native = NativeRobot(spec)
        command = env.command_manager.get_command("twist")[0].cpu().numpy()
        np.testing.assert_allclose(obs["actor"][0].cpu().numpy(),
                                   native.observation(command), atol=1e-5)
        reset_seen = False
        for _ in range(12):
            obs, reward, done, info = wrapped.step(torch.zeros(4, 12, device="cuda:0"))
            assert torch.isfinite(obs["actor"]).all() and torch.isfinite(reward).all()
            reset_seen |= bool(done.any())
        assert reset_seen
        robot = env.scene["robot"]
        root = torch.cat((robot.data.root_link_pos_w, robot.data.root_link_quat_w), dim=-1)
        root[:, 2] = .03
        robot.write_root_pose_to_sim(root)
        env.sim.forward()
        _, _, done, _ = wrapped.step(torch.zeros(4, 12, device="cuda:0"))
        assert done.all()
    finally:
        wrapped.close()
