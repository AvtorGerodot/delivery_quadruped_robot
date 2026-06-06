"""Shared configuration for the delivery-dog robot (B2 + Z1 + gripper).

The customer's ``delivery_dog_ws`` model differs from ``complex_urdf/b2_z1.urdf``:

* All B2 leg joints are prefixed ``b2_`` (e.g. ``b2_FL_hip_joint``).
* The Z1 arm carries a real **gripper** joint ``z1_jointGripper`` and a delivery
  box on the back, so the robot is heavier and more front-loaded → it stands a
  bit lower than the bare B2.
* The end-effector is the gripper finger TCP; with ``merge_fixed_links=True``
  it folds into ``z1_link06``, so we track ``z1_link06`` + a measured tool
  offset ``(0.208, 0, -0.042)`` (link06 frame) that lands on the finger tip.

This module produces ready-to-use config tuples for the two training regimes,
reusing the proven env classes:

* :func:`velocity_cfgs`  -> ``B2VelEnv``        (fixed manipulator, locomotion)
* :func:`wbc_cfgs`       -> ``B2Z1WholeBodyEnv``(whole-body EE reaching)

Both accept ``entrance=True`` to drop the entrance scene into the env (for eval)
and reposition the robot in front of the door.
"""

from __future__ import annotations

import os

import b2_vel_env
import b2_z1_wbc_env


REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DELIVERY_DOG_URDF = os.path.join(REPO, "complex_urdf", "delivery_dog_b2_z1.urdf")
ENTRANCE_MJCF = os.path.join(
    REPO, "delivery_dog_ws", "src", "mujoco_models", "mjcf", "entrance_group.xml"
)

# ---- Joint groups (names as they appear in the cleaned URDF) --------------
LEG_JOINTS = [
    "b2_FR_hip_joint", "b2_FR_thigh_joint", "b2_FR_calf_joint",
    "b2_FL_hip_joint", "b2_FL_thigh_joint", "b2_FL_calf_joint",
    "b2_RR_hip_joint", "b2_RR_thigh_joint", "b2_RR_calf_joint",
    "b2_RL_hip_joint", "b2_RL_thigh_joint", "b2_RL_calf_joint",
]
LEG_DEFAULTS = {
    "b2_FR_hip_joint": 0.0, "b2_FR_thigh_joint": 0.8, "b2_FR_calf_joint": -1.5,
    "b2_FL_hip_joint": 0.0, "b2_FL_thigh_joint": 0.8, "b2_FL_calf_joint": -1.5,
    "b2_RR_hip_joint": 0.0, "b2_RR_thigh_joint": 1.0, "b2_RR_calf_joint": -1.5,
    "b2_RL_hip_joint": 0.0, "b2_RL_thigh_joint": 1.0, "b2_RL_calf_joint": -1.5,
}
ARM_JOINTS = [f"z1_joint{i}" for i in range(1, 7)]
ARM_HOME = {
    "z1_joint1": 0.0, "z1_joint2": 1.5, "z1_joint3": -1.0,
    "z1_joint4": 0.0, "z1_joint5": 0.0, "z1_joint6": 0.0,
}
GRIPPER_JOINT = "z1_jointGripper"
GRIPPER_HOME = 0.0

# Measured end-effector (gripper finger tip) offset in the z1_link06 frame.
EE_LINK = "z1_link06"
EE_OFFSET = (0.208, 0.0, -0.042)

# This robot stands lower than the bare B2 (heavier, front-loaded).
BASE_INIT_POS = [0.0, 0.0, 0.55]
BASE_HEIGHT_TARGET = 0.45

# Eval-only: the dog spawns near the origin facing +x, and the entrance scene
# is pushed forward (+x) by ENTRANCE_OFFSET_X so the dog stands in front of the
# door rather than inside the doorway.
ENTRANCE_SPAWN_POS = [0.0, 0.0, 0.55]
ENTRANCE_OFFSET_X = 2.2


# =========================================================================
# Velocity locomotion with a fixed (static) manipulator + gripper
# =========================================================================
def velocity_cfgs(entrance: bool = False):
    env_cfg, obs_cfg, reward_cfg, command_cfg = b2_vel_env.default_cfgs("b2")

    env_cfg["urdf_path"] = DELIVERY_DOG_URDF
    env_cfg["joint_names"] = list(LEG_JOINTS)
    env_cfg["default_joint_angles"] = dict(LEG_DEFAULTS)
    # Arm (6) + gripper (1) are all held static during locomotion.
    env_cfg["arm_joint_names"] = ARM_JOINTS + [GRIPPER_JOINT]
    env_cfg["arm_default_angles"] = {**ARM_HOME, GRIPPER_JOINT: GRIPPER_HOME}
    env_cfg["arm_kp"] = 80.0
    env_cfg["arm_kd"] = 2.0
    env_cfg["base_init_pos"] = list(BASE_INIT_POS)
    reward_cfg["base_height_target"] = BASE_HEIGHT_TARGET

    if entrance:
        env_cfg["extra_mjcf"] = ENTRANCE_MJCF
        env_cfg["extra_mjcf_pos"] = [ENTRANCE_OFFSET_X, 0.0, 0.0]
        env_cfg["base_init_pos"] = list(ENTRANCE_SPAWN_POS)

    return env_cfg, obs_cfg, reward_cfg, command_cfg


# =========================================================================
# Whole-body control: legs + arm reach a world-frame ball, gripper static
# =========================================================================
def wbc_cfgs(entrance: bool = False):
    env_cfg, obs_cfg, reward_cfg, command_cfg, target_cfg, dynamic_cfg = (
        b2_z1_wbc_env.default_cfgs()
    )

    env_cfg["urdf_path"] = DELIVERY_DOG_URDF
    env_cfg["leg_joint_names"] = list(LEG_JOINTS)
    env_cfg["arm_joint_names"] = list(ARM_JOINTS)
    env_cfg["default_joint_angles"] = {**LEG_DEFAULTS, **ARM_HOME}
    # Gripper held static (not controlled by the policy).
    env_cfg["static_joint_names"] = [GRIPPER_JOINT]
    env_cfg["static_default_angles"] = {GRIPPER_JOINT: GRIPPER_HOME}
    env_cfg["static_kp"] = 40.0
    env_cfg["static_kd"] = 1.0
    env_cfg["base_init_pos"] = list(BASE_INIT_POS)
    reward_cfg["base_height_target"] = BASE_HEIGHT_TARGET

    # End-effector + reachable ball band for this taller mount.
    target_cfg["ee_link"] = EE_LINK
    target_cfg["ee_offset"] = EE_OFFSET
    target_cfg["z_min"] = 0.3
    target_cfg["z_max"] = 1.0

    if entrance:
        env_cfg["extra_mjcf"] = ENTRANCE_MJCF
        env_cfg["extra_mjcf_pos"] = [ENTRANCE_OFFSET_X, 0.0, 0.0]
        env_cfg["base_init_pos"] = list(ENTRANCE_SPAWN_POS)

    return env_cfg, obs_cfg, reward_cfg, command_cfg, target_cfg, dynamic_cfg
