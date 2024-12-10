from quaddif.algo.PPO import *
from quaddif.algo.APG import *
from quaddif.algo.SHAC import *
from quaddif.algo.dreamerv3 import *

AGENT_ALIAS = {
    "ppo": PPO,
    "ppo_rpl": PPO_RPL,
    "shac": SHAC,
    "shac_q": SHAC_Q,
    "shac_rpl": SHAC_RPL,
    "apg": APG,
    "apg_sto": APG_stochastic,
    "world": World_Agent
}