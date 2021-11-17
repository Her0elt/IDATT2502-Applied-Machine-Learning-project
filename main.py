# from src.trainers.dqn_trainer import run
from src.trainers.ppo_trainer import run

# run(pretrained=False, wandb_name="DDQN-run")
run(pretrained=False, wandb_name="PPO-run")
