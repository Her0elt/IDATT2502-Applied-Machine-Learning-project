# from trainers.dqn_trainer import play, run
from src.trainers.ppo_trainer import run

run(training_mode=True, pretrained=False)
# play()
