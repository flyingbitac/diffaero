from quaddif.env.position_control import *
from quaddif.env.obstacle_avoidance import *

ENV_ALIAS = {
    "position_control": PositionControl,
    "obstacle_avoidance": ObstacleAvoidance,
    "obstacle_avoidance_yopo": ObstacleAvoidanceYOPO,
    "obstacle_avoidance_grid": ObstacleAvoidanceGrid,
}