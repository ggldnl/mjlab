"""
Actions: how the policy output maps onto the robo (e.g. joint position targets with a scale factor)
"""

from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg

from mjlab.asset_zoo.robots.crawler.actuators import CRAWLER_ACTION_SCALE, CRAWLER_ACTION_OFFSET

actions: dict[str, ActionTermCfg] = {
  "joint_pos": JointPositionActionCfg(
    entity_name="robot",
    actuator_names=(".*",),
    scale=CRAWLER_ACTION_SCALE,
    offset=CRAWLER_ACTION_OFFSET,
    use_default_offset=True,
  ),
}