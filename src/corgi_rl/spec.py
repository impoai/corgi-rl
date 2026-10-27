"""Validated SI-unit robot specification; also consumed by deployment tooling."""

from hashlib import sha256
import json
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_SPEC = Path(__file__).parent / "specs" / "corgi.json"
LEGS = ("FL", "FR", "RL", "RR")
KINDS = ("hip", "thigh", "knee")
JOINT_ORDER = tuple(f"{leg}_{kind}" for leg in LEGS for kind in KINDS)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Joint(StrictModel):
    name: str
    motor_id: int = Field(ge=1, le=253)
    direction: Literal[-1, 1]
    zero_rad: float
    lower_rad: float
    upper_rad: float
    home_rad: float
    action_scale_rad: float = Field(gt=0)
    kp_nm_per_rad: float = Field(gt=0)
    kd_nm_s_per_rad: float = Field(ge=0)
    effort_limit_nm: float = Field(gt=0)
    armature_kg_m2: float = Field(ge=0)
    frictionloss_nm: float = Field(ge=0)

    @model_validator(mode="after")
    def ordered_limits(self):
        if not self.lower_rad < self.home_rad < self.upper_rad:
            raise ValueError(f"{self.name}: home must lie within joint limits")
        return self


class Geometry(StrictModel):
    body_length_m: float = Field(gt=0)
    body_width_m: float = Field(gt=0)
    body_height_m: float = Field(gt=0)
    body_mass_kg: float = Field(gt=0)
    hip_length_m: float = Field(gt=0)
    upper_length_m: float = Field(gt=0)
    lower_length_m: float = Field(gt=0)
    hip_mass_kg: float = Field(gt=0)
    upper_mass_kg: float = Field(gt=0)
    lower_mass_kg: float = Field(gt=0)
    foot_mass_kg: float = Field(gt=0)
    link_radius_m: float = Field(gt=0)
    foot_radius_m: float = Field(gt=0)


class Inertial(StrictModel):
    """Optional measured principal inertia, in body coordinates (no rotation)."""
    mass_kg: float = Field(gt=0)
    com_m: tuple[float, float, float]
    diagonal_kg_m2: tuple[float, float, float]

    @model_validator(mode="after")
    def physical_inertia(self):
        d = self.diagonal_kg_m2
        if min(d) <= 0 or 2 * max(d) > sum(d) + 1e-12:
            raise ValueError("Inertia must be positive and satisfy the triangle inequality")
        return self


class RobotSpec(StrictModel):
    schema_version: Literal[1]
    name: str
    calibration_status: Literal["estimated", "measured"]
    physics_dt_s: float = Field(gt=0)
    control_decimation: int = Field(gt=0)
    spawn_height_m: float = Field(gt=0)
    floor_friction: float = Field(gt=0)
    joint_velocity_observation_scale: float = Field(gt=0)
    geometry: Geometry
    joints: list[Joint]
    inertial_overrides: dict[str, Inertial] = Field(default_factory=dict)
    unmeasured: list[str]

    @model_validator(mode="after")
    def check_layout(self):
        if tuple(j.name for j in self.joints) != JOINT_ORDER:
            raise ValueError("Joint order must be FL, FR, RL, RR; hip, thigh, knee per leg")
        if len({j.motor_id for j in self.joints}) != 12:
            raise ValueError("Motor IDs must be unique")
        bodies = {"base"} | {f"{leg}_{part}" for leg in LEGS for part in KINDS}
        if set(self.inertial_overrides) - bodies:
            raise ValueError("Unknown body in inertial_overrides")
        return self

    @property
    def control_dt(self):
        return self.physics_dt_s * self.control_decimation

    @property
    def digest(self):
        return sha256(self.model_dump_json().encode()).hexdigest()

    def vector(self, key):
        return np.asarray([getattr(j, key) for j in self.joints], dtype=np.float64)

    def targets(self, action):
        a = np.asarray(action)
        if a.shape != (12,) or not np.isfinite(a).all():
            raise ValueError("Expected 12 finite actions")
        return np.clip(
            self.vector("home_rad") + self.vector("action_scale_rad") * np.clip(a, -1, 1),
            self.vector("lower_rad"), self.vector("upper_rad"),
        )

    def hardware_positions(self, model_positions):
        """Position-only conversion; no hardware I/O is implemented."""
        q = np.asarray(model_positions)
        if q.shape != (12,) or not np.isfinite(q).all():
            raise ValueError("Expected 12 finite joint positions")
        if np.any(q < self.vector("lower_rad")) or np.any(q > self.vector("upper_rad")):
            raise ValueError("Joint position exceeds calibrated limits")
        return self.vector("direction") * q + self.vector("zero_rad")

    def manifest(self):
        return {
            "contract_version": 1, "spec_sha256": self.digest,
            "joint_order": list(JOINT_ORDER), "control_hz": 1 / self.control_dt,
            "observation_dim": 45, "action_dim": 12,
            "observations": [
                ["angular_velocity_body_rad_s", 3], ["gravity_unit_body", 3],
                ["command_vx_vy_yaw_body", 3], ["joint_position_minus_home_rad", 12],
                ["joint_velocity_scaled", 12], ["previous_clipped_action", 12],
            ],
            "joint_velocity_scale": self.joint_velocity_observation_scale,
            "action": "clip(home + scale * clip(action, -1, 1), lower, upper)",
            "frames": "right handed: x forward, y left, z up; quaternion wxyz",
            "hardware_ready": False, "calibration_status": self.calibration_status,
        }


def load_spec(path=None):
    return RobotSpec.model_validate_json(Path(path or DEFAULT_SPEC).read_text())


def write_json(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
