"""
Actions: how the policy output maps onto the robot (e.g. joint position targets with a scale factor)
"""

from mjlab.asset_zoo.robots.crawler.actuators import ACTION_OFFSET, ACTION_SCALE
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg

actions: dict[str, ActionTermCfg] = {
  "joint_pos": JointPositionActionCfg(
    entity_name="robot",
    actuator_names=(".*",),
    scale=ACTION_SCALE,
    offset=ACTION_OFFSET,
    use_default_offset=False,
  ),
}
