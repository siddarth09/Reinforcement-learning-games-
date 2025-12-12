import pygame
import random
import numpy as np
import sys
import argparse
from collections import deque

# Optional: plotting after training
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# --------------------------
# Argument Parser (NEW)
# --------------------------
def get_args():
    parser = argparse.ArgumentParser(description="Flappy Bird Q-Learning")

    parser.add_argument("--episodes", type=int, default=3000,
                        help="Number of training episodes")

    parser.add_argument("--render-training", type=bool, default=False,
                        help="Render during training (slower)")

    parser.add_argument("--epsilon-start", type=float, default=1.0,
                        help="Initial epsilon")

    parser.add_argument("--epsilon-min", type=float, default=0.05,
                        help="Minimum epsilon")

    parser.add_argument("--epsilon-decay", type=float, default=0.995,
                        help="Epsilon decay rate")

    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Q-learning learning rate")

    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")

    return parser.parse_args()


# ----------------------
# Game / Env Settings
# ----------------------
WIDTH, HEIGHT = 400, 600
FPS = 60

BIRD_X = 80
BIRD_RADIUS = 15
GRAVITY = 0.8
FLAP_STRENGTH = -10
PIPE_SPEED = 3
PIPE_WIDTH = 60
GAP = 160

# Discretization config
DY_BINS = 20
DX_BINS = 20
VEL_BINS = 10

DY_LOW, DY_HIGH = -400, 400
DX_LOW, DX_HIGH = -150, 400
VEL_LOW, VEL_HIGH = -20, 20

ACTIONS = [0, 1]  # 0 = nothing, 1 = flap


# ----------------------
# Helper functions
# ----------------------
def discretize(value, bins, low, high):
    value = max(min(value, high), low)
    ratio = (value - low) / (high - low)
    bin_index = int(ratio * bins)
    return min(bins - 1, max(0, bin_index))


# ----------------------
# Flappy Bird Environment
# ----------------------
class FlappyEnv:
    def __init__(self, screen=None, clock=None):
        self.screen = screen
        self.clock = clock
        self.font = pygame.font.SysFont("Arial", 20)
        self.reset()

    def reset(self):
        self.bird_y = HEIGHT // 2
        self.bird_vel = 0

        self.pipe_x = WIDTH + 100
        self.pipe_gap_y = random.randint(150, HEIGHT - 150)

        self.score = 0
        self.done = False
        return self.get_state()

    def step(self, action):
        if action == 1:
            self.bird_vel = FLAP_STRENGTH

        self.bird_vel += GRAVITY
        self.bird_y += self.bird_vel

        self.pipe_x -= PIPE_SPEED

        reward = 0.5

        if self.pipe_x + PIPE_WIDTH < BIRD_X and not self.done:
            self.score += 1
            reward += 50.0
            self.pipe_x = WIDTH + random.randint(0, 80)
            self.pipe_gap_y = random.randint(150, HEIGHT - 150)

        if self.bird_y - BIRD_RADIUS < 0 or self.bird_y + BIRD_RADIUS > HEIGHT:
            reward -= 100.0
            self.done = True

        if not self.done:
            in_pipe_x = (BIRD_X + BIRD_RADIUS > self.pipe_x) and (BIRD_X - BIRD_RADIUS < self.pipe_x + PIPE_WIDTH)
            if in_pipe_x:
                if not (self.pipe_gap_y < self.bird_y < self.pipe_gap_y + GAP):
                    reward -= 100.0
                    self.done = True

        return self.get_state(), reward, self.done

    def get_state(self):
        dy = self.bird_y - (self.pipe_gap_y + GAP / 2.0)
        dx = self.pipe_x - BIRD_X
        dy_bin = discretize(dy, DY_BINS, DY_LOW, DY_HIGH)
        dx_bin = discretize(dx, DX_BINS, DX_LOW, DX_HIGH)
        vel_bin = discretize(self.bird_vel, VEL_BINS, VEL_LOW, VEL_HIGH)
        return (dy_bin, dx_bin, vel_bin)

    def render(self, episode=None, mode="train"):
        if self.screen is None:
            return

        self.screen.fill((135, 206, 235))

        pygame.draw.rect(self.screen, (34, 139, 34),
                         (self.pipe_x, 0, PIPE_WIDTH, self.pipe_gap_y))

        pygame.draw.rect(self.screen, (34, 139, 34),
                         (self.pipe_x, self.pipe_gap_y + GAP, PIPE_WIDTH, HEIGHT))

        pygame.draw.circle(self.screen, (255, 255, 0),
                           (int(BIRD_X), int(self.bird_y)), BIRD_RADIUS)

        score_text = self.font.render(f"Score: {self.score}", True, (0, 0, 0))
        self.screen.blit(score_text, (10, 10))

        mode_text = self.font.render(f"Mode: {mode}", True, (0, 0, 0))
        self.screen.blit(mode_text, (10, 35))

        if episode is not None:
            ep_text = self.font.render(f"Episode: {episode}", True, (0, 0, 0))
            self.screen.blit(ep_text, (10, 60))

        pygame.display.flip()

        if self.clock is not None:
            self.clock.tick(FPS)


# ----------------------
# Q-Learning Agent
# ----------------------
class QLearningAgent:
    def __init__(self, alpha, gamma, epsilon_start, epsilon_min, epsilon_decay):
        self.Q = np.zeros((DY_BINS, DX_BINS, VEL_BINS, len(ACTIONS)), dtype=np.float32)

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def select_action(self, state, greedy=False):
        if greedy or np.random.rand() > self.epsilon:
            return int(np.argmax(self.Q[state]))
        else:
            return random.choice(ACTIONS)

    def update(self, state, action, reward, next_state, done):
        q_sa = self.Q[state][action]
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q[next_state])

        self.Q[state][action] = q_sa + self.alpha * (target - q_sa)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ----------------------
# Training Loop
# ----------------------
def train(env, agent, episodes, render_training):
    reward_history = []
    score_history = []

    for episode in range(1, episodes + 1):
        state = env.reset()
        ep_reward = 0.0
        done = False

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)

            state = next_state
            ep_reward += reward

            if render_training:
                env.render(episode=episode, mode="train")

        agent.decay_epsilon()
        reward_history.append(ep_reward)
        score_history.append(env.score)

        if episode % 100 == 0:
            print(f"Episode {episode:4d} | Epsilon: {agent.epsilon:.3f} | "
                  f"AvgReward: {np.mean(reward_history[-100:]):.2f} | "
                  f"AvgScore: {np.mean(score_history[-100:]):.2f}")

    return reward_history, score_history


# ----------------------
# Plotting
# ----------------------
def plot_training(reward_history, score_history):
    if not HAS_MPL:
        print("matplotlib not installed, skipping plot.")
        return

    plt.figure(figsize=(10, 5))
    plt.plot(reward_history, label="Reward")
    plt.plot(score_history, label="Score")
    plt.title("Training Progress")
    plt.legend()
    plt.show()


# ----------------------
# Watch Agent
# ----------------------
def watch_agent(env, agent, episodes=5):
    agent.epsilon = 0.0  # fully greedy

    for ep in range(episodes):
        state = env.reset()
        done = False

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            action = agent.select_action(state, greedy=True)
            next_state, reward, done = env.step(action)
            state = next_state

            env.render(episode=ep, mode="play")

        print(f"Play Episode {ep}: Score = {env.score}")


# ----------------------
# Main
# ----------------------
def main():
    args = get_args()

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    env = FlappyEnv(screen, clock)

    agent = QLearningAgent(
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
    )

    print("Starting training...")
    reward_history, score_history = train(
        env,
        agent,
        episodes=args.episodes,
        render_training=args.render_training
    )
    print("Training complete!")

    np.save("flappy_q_table.npy", agent.Q)
    print("Saved Q-table to flappy_q_table.npy")

    plot_training(reward_history, score_history)

    print("Watching trained agent...")
    watch_agent(env, agent, episodes=5)

    pygame.quit()


if __name__ == "__main__":
    main()
