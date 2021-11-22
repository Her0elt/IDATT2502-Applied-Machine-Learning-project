from src.constants import GAMMA, MIN_EPSILON
from src.trainers.dqn_trainer import run as DQN_run

# from src.constants import  PPO_GAMMA, PPO_LAMBDA
# from src.trainers.ppo_trainer import run as PPO_Run

DQN_run(
    pretrained=False,
    double=False,
    wandb_name=f"DQN-run-v2-5000-eps-epsilon-{MIN_EPSILON}-gamma-{GAMMA}",
)
# PPO_Run(pretrained=False, wandb_name=f"PPO-run-5000-eps-gamma-{PPO_GAMMA}-lambda-{PPO_LAMBDA}")
