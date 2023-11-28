import numpy as np
import pygame
import gym
from gym import spaces
import time

class QLearningAgent:
    def __init__(self, action_space_size, observation_space_size, learning_rate=0.9, discount_factor=0.9, exploration_prob=0.3):
        self.action_space_size = action_space_size
        self.observation_space_size = observation_space_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob
        self.q_table = np.zeros((observation_space_size, action_space_size))

    def select_action(self, state):
        if np.random.rand() < self.exploration_prob:
            return np.random.randint(0, self.action_space_size)
        else:
            return np.argmax(self.q_table[state, :])

    def update_q_table(self, state, action, reward, next_state, done):
        if not done:
            best_next_action = np.argmax(self.q_table[next_state, :])
            self.q_table[state, action] += self.learning_rate * (
                reward + self.discount_factor * self.q_table[next_state, best_next_action] - self.q_table[state, action]
            )
        else:
            self.q_table[state, action] += self.learning_rate * (reward - self.q_table[state, action])

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=10, num_obstacles=15, action_delay=0.0000001):
        self.size = size
        self.window_size = 512
        self.num_obstacles = num_obstacles
        self.action_delay = action_delay
        self._target_location = np.random.randint(0, self.size, size=2, dtype=int)

        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        self.action_space = spaces.Discrete(8)

        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([1, 1]),
            2: np.array([0, 1]),
            3: np.array([-1, 1]),
            4: np.array([-1, 0]),
            5: np.array([-1, -1]),
            6: np.array([0, -1]),
            7: np.array([1, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None
        self.obstacles = set()
        self._generate_obstacles()
        self.q_learning_agent = QLearningAgent(
            action_space_size=self.action_space.n,
            observation_space_size=size * size
        )

    def _generate_obstacles(self):
        # �ֶ��趨һЩ�ϰ����λ��
        obstacle_positions = [
            (2, 3), (4, 7), (6, 2), (8, 5), (1, 9)
            # ���������ӻ��޸��ϰ����λ��
        ]

        self.obstacles = set(obstacle_positions)

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._agent_location = np.random.randint(0, self.size, size=2, dtype=int)
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        state = self._agent_location[0] * self.size + self._agent_location[1]
        return state, info

    def display_q_table(self):
        print("Q-Table:")
        for state in range(self.size * self.size):
            print(f"State {state}: {self.q_learning_agent.q_table[state]}")

    def step(self, action):
        direction = self._action_to_direction[action]
        new_location = np.clip(self._agent_location + direction, 0, self.size - 1)

        if tuple(new_location) not in self.obstacles:
            self._agent_location = new_location

        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if terminated else 0
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()
            time.sleep(0.01)

        next_state = self._agent_location[0] * self.size + self._agent_location[1]
        self.q_learning_agent.update_q_table(state, action, reward, next_state, terminated)

        return next_state, reward, terminated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = self.window_size / self.size

        for obstacle_location in self.obstacles:
            pygame.draw.rect(
                canvas,
                (169, 169, 169),  # �޸���ɫΪ��ɫ
                pygame.Rect(
                    pix_square_size * np.array(obstacle_location),
                    (pix_square_size, pix_square_size),
                ),
            )

        distance = np.linalg.norm(self._agent_location - self._target_location, ord=1)
        color = (int(255 - 15 * distance), 0, int(15 * distance))

        pygame.draw.rect(
            canvas,
            (255, 0, 0),  # �޸���ɫΪ��ɫ
            pygame.Rect(
                pix_square_size * self._agent_location,
                (pix_square_size, pix_square_size),
            ),
        )

        pygame.draw.rect(
            canvas,
            (255, 255, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )

        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def evaluate_agent(self, num_episodes=100):
        total_penalties = 0
        total_timesteps = 0
        total_rewards = 0

        for _ in range(num_episodes):
            state, _ = self.reset()
            episode_penalties = 0
            episode_timesteps = 0
            episode_rewards = 0

            while True:
                action = self.q_learning_agent.select_action(state)
                next_state, reward, done, _ = self.step(action)

                episode_penalties += reward
                episode_timesteps += 1
                episode_rewards += reward

                if done:
                    break

                state = next_state

            total_penalties += episode_penalties
            total_timesteps += episode_timesteps
            total_rewards += episode_rewards

        avg_penalties = total_penalties / num_episodes
        avg_timesteps = total_timesteps / num_episodes
        avg_rewards_per_move = total_rewards / total_timesteps

        print("Evaluation Metrics:")
        print(f"Average Penalties per Episode: {avg_penalties}")
        print(f"Average Timesteps per Trip: {avg_timesteps}")
        print(f"Average Rewards per Move: {avg_rewards_per_move}")

# ����GridWorldEnv����
env = GridWorldEnv(size=10, num_obstacles=15, render_mode="human", action_delay=0.1)
env.display_q_table()

# ѵ��Q-learning����
num_episodes_training = 10

for episode in range(num_episodes_training):
    state, _ = env.reset()
    total_reward = 0

    while True:
        action = env.q_learning_agent.select_action(state)
        next_state, reward, done, _ = env.step(action)

        env.q_learning_agent.update_q_table(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward

        if done:
            break

    print(f"Episode {episode + 1}/{num_episodes_training}, Total Reward: {total_reward}")

# ѵ������������
env.evaluate_agent(num_episodes=10)
env.display_q_table()

# ����ѵ����Ĵ���
state, _ = env.reset()
visited_states = [state]

while True:
    action = env.q_learning_agent.select_action(state)
    next_state, reward, done, _ = env.step(action)

    if done:
        break

env.display_q_table()
print("Optimal Path:")
for state in visited_states:
    position = (state // env.size, state % env.size)
    print(f"State: {position}")

env.close()
