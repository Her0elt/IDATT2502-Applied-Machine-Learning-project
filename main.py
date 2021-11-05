from agent import DQNAgent
from constants import EPISODES, REPLAY_FRAME_COUNT
from environment import create_mario_env


def train(env, agent, episodes):
    scores = []
    num_episode_steps = env.spec.max_episode_steps
    max_reward = 0
    for episode in range(episodes):
        score = 0
        print(f"Episode: {episode}")
        done = False
        frame_count = 0
        state = agent.preprocess_state(env.reset())
        info = None
        for _ in range(num_episode_steps):
            action = agent.action(state)
            next_state, reward, done, info = env.step(action)
            env.render(mode="human")
            next_state = agent.preprocess_state(next_state)
            agent.remember(state, action, reward, next_state, done)
            score += reward
            state = next_state
            if done:
                break
            if frame_count % REPLAY_FRAME_COUNT == 0:
                agent.replay()
            frame_count += 1
        print(f"Score: {score}, max: {max_reward}")
        if score > max_reward or info["flag_get"]:
            max_reward = score
            agent.model.save_model(f"models/dqn_mario_agent_v0_{score}.h5")

        scores.append(reward)
    print("Finished training!")
    env.close()


def run(env, agent, model_name):
    agent.model.load_model(model_name)
    num_episode_steps = env.spec.max_episode_steps
    state = agent.preprocess_state(env.reset())
    for _ in range(num_episode_steps):
        action = agent.action(state)
        next_state, reward, done, _ = env.step(action)
        env.render(mode="human")
        next_state = agent.preprocess_state(next_state)
        state = next_state
    env.close()


env = create_mario_env()
state_space = env.observation_space.shape
action_space = env.action_space.n
agent = DQNAgent(env, state_space, action_space)
train(env, agent, episodes=EPISODES)
# run(env, agent, "dqn_mario_agent_v0.h5")
