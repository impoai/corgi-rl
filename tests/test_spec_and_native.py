import numpy as np
import pytest
from pydantic import ValidationError

from corgi_rl.model import mjcf
from corgi_rl.native import NativeRobot, evaluate
from corgi_rl.spec import JOINT_ORDER, RobotSpec, load_spec


def test_model_contract_and_calibration():
    spec = load_spec()
    robot = NativeRobot(spec)
    assert (robot.model.nq, robot.model.nv, robot.model.nu) == (19, 18, 12)
    assert tuple(robot.model.joint(i).name for i in range(1, 13)) == JOINT_ORDER
    assert robot.observation(np.zeros(3)).shape == (45,)
    assert np.isfinite(robot.observation(np.zeros(3))).all()
    targets = spec.targets(np.full(12, 100.))
    assert np.all(targets <= spec.vector("upper_rad"))
    assert np.all(targets >= spec.vector("lower_rad"))
    changed = spec.model_copy(deep=True)
    changed.geometry.body_mass_kg *= 1.1
    assert changed.digest != spec.digest
    other = NativeRobot(changed)
    assert other.model.body_mass.sum() > robot.model.body_mass.sum()
    changed.joints[0].direction = -1
    changed.joints[0].zero_rad = .12
    q = np.full(12, 0.)
    q[:] = spec.vector("home_rad")
    q[0] = .1
    assert changed.hardware_positions(q)[0] == pytest.approx(.02)
    assert "mesh" not in mjcf(spec)


def test_bad_specs_rejected():
    spec = load_spec().model_dump()
    spec["joints"][0]["motor_id"] = spec["joints"][1]["motor_id"]
    with pytest.raises(ValidationError):
        RobotSpec.model_validate(spec)
    with pytest.raises(ValueError):
        load_spec().targets(np.full(12, np.nan))
    spec = load_spec().model_dump()
    spec["geometry"]["body_mass_kg"] = -1
    with pytest.raises(ValidationError):
        RobotSpec.model_validate(spec)


def test_home_pd_stands_for_three_seconds():
    result = evaluate(load_spec(), seconds=3.)
    assert result["first_fall_s"] is None
    assert result["minimum_body_height_m"] > .12
    assert result["maximum_torque_nm"] <= 2.50001


def test_seeded_replay_is_repeatable():
    a = evaluate(load_spec(), seconds=.2, seed=7)
    b = evaluate(load_spec(), seconds=.2, seed=7)
    assert a == b
