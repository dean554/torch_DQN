import numpy as np

class envirn:
    def __init__(self):
        self.start_x = 10.0
        self.min_x = 5.0
        self.max_x = 15.0
        self.state = self.start_x

        self.current_step = 0
        self.max_step = 200

    def reset(self):
        self.current_step = 0
        self.state = self.start_x
        observation = np.array([self.state], np.float32)
        info = {}
        return observation, info

    def step(self, action):
        self.current_step += 1
        if action == 1:
            self.state += 0.5
        elif action == 0:
            self.state -= 0.5

        self.state += np.random.uniform(-0.8, 0.8)

        terminate = bool(self.state < self.min_x or self.state > self.max_x)
        truncated = bool(self.current_step > self.max_step)

        reward = 0.0 if terminate else 1.0

        observation = np.array([self.state], np.float32)
        info = {}

        return observation, reward, terminate, truncated, info