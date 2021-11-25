# from src.constants import GAMMA, MIN_EPSILON
# from src.trainers.dqn_trainer import run as DQN_run, play

from src.constants import EPISODES, PPO_GAMMA, PPO_LAMBDA
# from src.trainers.ppo_trainer import play
from src.trainers.ppo_trainer import run as PPO_Run

# DQN_run(
#     pretrained=False,
#     wandb_name=f"DDQN-run-v2-{EPISODES}-eps-epsilon-{MIN_EPSILON}-gamma-{GAMMA}-final-run",
# )
PPO_Run(
    pretrained=False,
    wandb_name=f"PPO-run-{EPISODES}-eps-gamma-{PPO_GAMMA}-lambda-{PPO_LAMBDA}-final-run",
)
