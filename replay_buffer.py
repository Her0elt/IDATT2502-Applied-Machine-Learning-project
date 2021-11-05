from collections import deque


class ReplyBuffer:
    def __init__(self, memory_size=20000):
        self.state = deque(maxlen=memory_size)
        self.action = deque(maxlen=memory_size)
        self.reward = deque(maxlen=memory_size)
        self.next_state = deque(maxlen=memory_size)
        self.done = deque(maxlen=memory_size)

    def append(self, state, action, reward, next_state, done):
        self.state.append(state)
        self.action.append(action)
        self.reward.append(reward)
        self.next_state.append(next_state)
        self.done.append(done)

    def __len__(self) -> int:
        return len(self.done)
