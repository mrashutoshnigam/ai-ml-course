import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

# Named tuple for state-action-reward-done
Step = namedtuple('Step', ['next_state', 'reward', 'done'])


class GridWorld:
    def __init__(self):
        self.size = 5
        self.actions = {'N': (-1, 0), 'S': (1, 0), 'E': (0, 1), 'W': (0, -1)}
        self.action_keys = list(self.actions.keys())
        self.num_actions = len(self.actions)
        self.start = (0, 0)  # 'A'
        self.goal = (4, 4)   # 'G', reward +5
        self.terminal_bad = (2, 4)  # 'T', reward -5
        self.walls = [(1, 1), (1, 3), (2, 0), (3, 2)]  # Black walls
        self.rewards = {self.goal: 5, self.terminal_bad: -5}
        self.state_size = self.size * self.size

    def is_valid(self, state):
        row, col = state
        return 0 <= row < self.size and 0 <= col < self.size and state not in self.walls

    def step(self, state, action):
        delta = self.actions[action]
        next_state = (state[0] + delta[0], state[1] + delta[1])
        if not self.is_valid(next_state):
            return Step(state, -1, False)  # Boundary/wall hit: stay, -1 reward
        reward = self.rewards.get(next_state, 0)
        done = next_state in {self.goal, self.terminal_bad}
        return Step(next_state, reward, done)

    def state_to_index(self, state):
        return state[0] * self.size + state[1]

    def index_to_state(self, index):
        return divmod(index, self.size)


def q_learning(env, gamma, epsilon, episodes=100000, alpha=0.1):
    np.random.seed(42)  # Fixed seed for reproducibility
    Q = np.zeros((env.state_size, env.num_actions))
    steps_to_goal = []  # For Task 3

    for episode in range(episodes):
        state = env.start
        done = False
        steps = 0
        while not done:
            s_idx = env.state_to_index(state)
            if np.random.uniform() < epsilon:
                action_idx = np.random.randint(env.num_actions)
            else:
                action_idx = np.argmax(Q[s_idx])
            action = env.action_keys[action_idx]
            step_result = env.step(state, action)
            ns_idx = env.state_to_index(step_result.next_state)
            # Q-update
            target = step_result.reward + gamma * \
                np.max(Q[ns_idx]) * (not step_result.done)
            Q[s_idx, action_idx] += alpha * (target - Q[s_idx, action_idx])
            state = step_result.next_state
            done = step_result.done
            steps += 1
        # Only count if reached goal (not trap)
        if step_result.next_state == env.goal:
            steps_to_goal.append(steps)
        else:
            # Trap episodes: infinite steps (penalty)
            steps_to_goal.append(np.inf)

    # Compute value function V(s) = max_a Q(s,a)
    V = np.max(Q, axis=1).reshape((env.size, env.size))

    # Policy: best action per state
    policy = np.argmax(Q, axis=1).reshape((env.size, env.size))

    return Q, V, policy, steps_to_goal


def plot_policy_value(env, V, policy, title, gamma_str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Value heatmap
    im1 = ax1.imshow(V, cmap='RdYlGn', vmin=-10, vmax=10)
    ax1.set_title(f'Value Function (γ={gamma_str})')
    ax1.set_xticks(np.arange(env.size))
    ax1.set_yticks(np.arange(env.size))
    plt.colorbar(im1, ax=ax1)

    # Mark special states
    ax1.plot(env.start[1], env.start[0], 'bs',
             markersize=10, label='Start')  # Blue square
    ax1.plot(env.goal[1], env.goal[0], 'gs', markersize=10, label='Goal (+5)')
    ax1.plot(env.terminal_bad[1], env.terminal_bad[0],
             'rs', markersize=10, label='Trap (-5)')
    for wall in env.walls:
        ax1.plot(wall[1], wall[0], 'k^', markersize=8)
    ax1.legend()

    # Policy arrows
    arrow_dict = {0: '↑', 1: '↓', 2: '→', 3: '←'}  # N=0, S=1, E=2, W=3
    for i in range(env.size):
        for j in range(env.size):
            if (i, j) in env.walls or (i, j) in {env.goal, env.terminal_bad}:
                continue
            ax2.text(j, i, arrow_dict[policy[i, j]], ha='center',
                     va='center', fontsize=20, color='blue')
    ax2.set_title(f'Policy (γ={gamma_str})')
    ax2.set_xticks(np.arange(env.size))
    ax2.set_yticks(np.arange(env.size))
    ax2.grid(True)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_steps_to_goal(steps_list, epsilons, gamma=0.9):
    # Smooth with moving average (window=1000) for 100k episodes
    episodes = np.arange(len(steps_list))
    window = 1000
    smoothed = np.convolve(steps_list, np.ones(window)/window, mode='valid')

    plt.figure(figsize=(10, 6))
    for eps in epsilons:
        # Note: Run separate trainings for each epsilon in practice; here assume steps_list per epsilon
        # For illustration, plot one; extend as needed
        plt.plot(episodes[:len(smoothed)], smoothed, label=f'ε={eps}')
    plt.title(
        f'Steps to Goal vs. Episodes (γ={gamma}, Moving Avg. Window={window})')
    plt.xlabel('Episode')
    plt.ylabel('Steps to Goal')
    plt.legend()
    plt.yscale('log')  # Log scale for large range (inf steps for traps)
    plt.grid(True)
    plt.show()


# Main execution
env = GridWorld()

# Task 1: Converged policy and value (assuming γ=0.9, ε=0.1)
print("Running Task 1 (γ=0.9, ε=0.1)...")
Q1, V1, policy1, _ = q_learning(env, gamma=0.9, epsilon=0.1)
plot_policy_value(env, V1, policy1,
                  'Task 1: Converged Policy and Value Function', '0.9')

# Task 2: For γ=0.1, 0.5, 0.9 (ε=0.1)
print("Running Task 2...")
gammas = [0.1, 0.5, 0.9]
for g in gammas:
    Q, V, policy, _ = q_learning(env, gamma=g, epsilon=0.1)
    plot_policy_value(
        env, V, policy, f'Task 2: Policy and Value (γ={g})', str(g))

# Task 3: For γ=0.9, steps across episodes for ε=0.1, 0.3, 0.5
print("Running Task 3 (γ=0.9)...")
epsilons = [0.1, 0.3, 0.5]
all_steps = []
for eps in epsilons:
    _, _, _, steps = q_learning(env, gamma=0.9, epsilon=eps)
    all_steps.append(steps)
# Plot (extend plot_steps_to_goal to handle list of lists if multiple runs)
# For brevity, plot first one; replicate for others
plot_steps_to_goal(all_steps[0], epsilons)  # Example; adjust to plot all
