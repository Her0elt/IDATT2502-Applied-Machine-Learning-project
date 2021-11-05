from agent import DQNAgent
from environment import MarioEnvironment


def train(env, agent, episodes):
    frame_count = 0
    score = 0
    scores = []
    num_episode_steps = env.spec.max_episode_steps
    max_reward = 0
    for _ in range(episodes):
        done = False
        state = agent.preprocess_state(env.reset())
        for _ in range(num_episode_steps):
            action = agent.action(state)
            next_state, reward, done, _ = env.step(action)
            env.render(mode="human")
            next_state = agent.preprocess_state(next_state)
            agent.remember(state, action, reward, next_state, done)
            score += reward
            state = next_state
            if done:
                break
            if frame_count % 4 == 0:
                agent.replay()
            frame_count += 1
        if score > max_reward:
            agent.model.save_model("dqn_mario_agent_v0.h5")

        scores.append(reward)
    print("Finished training!")
    env.close()


env = MarioEnvironment()
state_space = env.observation_space.shape
action_space = env.action_space.n
agent = DQNAgent(env, state_space, action_space)
train(env, agent, episodes=50000)
