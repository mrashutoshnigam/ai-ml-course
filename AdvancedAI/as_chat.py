# Q-learning implementation and plots
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time
np.random.seed(0)
random.seed(0)

# Grid setup inferred from image
rows, cols = 5, 5
start = (0, 0)
goal = (4, 4)  # G
ter_minus = (2, 4)  # T -5
obstacles = set([(1, 1), (1, 3), (2, 0), (3, 2)])

actions = ['N', 'S', 'E', 'W']
action_delta = {'N': (-1, 0), 'S': (1, 0), 'E': (0, 1), 'W': (0, -1)}


def step(state, action):
    r, c = state
    dr, dc = action_delta[action]
    nr, nc = r+dr, c+dc
    # outside boundary
    if not (0 <= nr < rows and 0 <= nc < cols):
        return state, -1, False  # stay, reward -1, not terminal
    # into obstacle: stay, reward 0
    if (nr, nc) in obstacles:
        return state, 0, False
    # moved to new cell
    if (nr, nc) == goal:
        return (nr, nc), 5, True
    if (nr, nc) == ter_minus:
        return (nr, nc), -5, True
    return (nr, nc), 0, False


def q_learning(gamma=0.9, epsilon=0.1, alpha=0.1, episodes=100000, track_steps=False):
    Q = {(r, c): {a: 0.0 for a in actions}
         for r in range(rows) for c in range(cols)}
    for obs in obstacles:
        # can keep Q but it's never visited if start avoids, but leave as zeros
        pass
    steps_to_goal = []
    for ep in range(episodes):
        state = start
        steps = 0
        done = False
        while not done and steps < 1000:
            # epsilon-greedy
            if random.random() < epsilon:
                act = random.choice(actions)
            else:
                vals = Q[state]
                maxv = max(vals.values())
                best = [a for a, v in vals.items() if v == maxv]
                act = random.choice(best)
            next_state, reward, done = step(state, act)
            # Q update (Q-learning)
            next_max = 0 if done else max(Q[next_state].values())
            Q[state][act] += alpha*(reward + gamma*next_max - Q[state][act])
            state = next_state
            steps += 1
        if track_steps:
            steps_to_goal.append(steps if done else None)
    return Q, steps_to_goal


def policy_and_value_from_Q(Q):
    policy = np.full((rows, cols), '', dtype=object)
    V = np.zeros((rows, cols))
    for r in range(rows):
        for c in range(cols):
            if (r, c) in obstacles:
                policy[r, c] = 'X'   # obstacle
                V[r, c] = np.nan
            elif (r, c) == goal:
                policy[r, c] = 'G'
                V[r, c] = 5.0
            elif (r, c) == ter_minus:
                policy[r, c] = 'T'
                V[r, c] = -5.0
            else:
                acts = Q[(r, c)]
                maxv = max(acts.values())
                best = [a for a, v in acts.items() if v == maxv]
                policy[r, c] = random.choice(best)
                V[r, c] = maxv
    return policy, V


# Run for gamma = 0.1, 0.5, 0.9 with epsilon=0.1
results = {}
for gamma in [0.1, 0.5, 0.9]:
    print("Running gamma", gamma)
    Q, _ = q_learning(gamma=gamma, epsilon=0.1, alpha=0.1,
                      episodes=100000, track_steps=False)
    policy, V = policy_and_value_from_Q(Q)
    results[gamma] = (policy, V)

# Plot the policy arrows and value heatmap
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, gamma in zip(axes, [0.1, 0.5, 0.9]):
    policy, V = results[gamma]
    ax.set_title(f'Policy (gamma={gamma})')
    # show grid
    ax.set_xlim(-0.5, cols-0.5)
    ax.set_ylim(rows-0.5, -0.5)
    for r in range(rows):
        for c in range(cols):
            if (r, c) in obstacles:
                rect = plt.Rectangle((c-0.5, r-0.5), 1, 1,
                                     fill=True, edgecolor='k')
                ax.add_patch(rect)
            else:
                ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1,
                             1, fill=False, edgecolor='gray'))
            cell = policy[r, c]
            if cell in ['N', 'S', 'E', 'W']:
                dx, dy = 0, 0
                if cell == 'N':
                    dx, dy = 0, -0.25
                if cell == 'S':
                    dx, dy = 0, 0.25
                if cell == 'E':
                    dx, dy = 0.25, 0
                if cell == 'W':
                    dx, dy = -0.25, 0
                ax.arrow(c, r, dx, dy, head_width=0.12, head_length=0.1)
            elif cell == 'G':
                ax.text(c, r, 'G', ha='center', va='center',
                        fontsize=12, fontweight='bold')
            elif cell == 'T':
                ax.text(c, r, 'T', ha='center', va='center',
                        fontsize=12, fontweight='bold')
            elif cell == 'X':
                ax.text(c, r, '■', ha='center', va='center', fontsize=18)
    # overlay value
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in obstacles:
                ax.text(c-0.35, r+0.25, f"{V[r,c]:.2f}", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
plt.suptitle('Converged policy and value for different gamma (epsilon=0.1)')
plt.show()

# For gamma=0.9, track steps across episodes for epsilons 0.1,0.3,0.5
epsilons = [0.1, 0.3, 0.5]
steps_data = {}
# to speed up, we'll use 20k episodes for tracking (still informative)
episodes = 20000
for eps in epsilons:
    print("Running gamma=0.9, epsilon=", eps)
    Q, steps = q_learning(gamma=0.9, epsilon=eps, alpha=0.1,
                          episodes=episodes, track_steps=True)
    steps_data[eps] = steps

# Compute moving average of steps (for episodes where goal reached)


def moving_avg(data, window=200):
    arr = np.array([x if x is not None else np.nan for x in data], dtype=float)
    # replace nans with large value for visualization? we'll keep nans
    out = np.convolve(np.nan_to_num(arr, nan=np.nanmean(arr)),
                      np.ones(window)/window, mode='valid')
    return out


plt.figure(figsize=(8, 5))
for eps in epsilons:
    ma = moving_avg(steps_data[eps], window=200)
    plt.plot(ma, label=f'eps={eps}')
plt.xlabel('Episode (smoothed)')
plt.ylabel('Steps to reach goal (moving average)')
plt.legend()
plt.title('Steps to reach goal across episodes (gamma=0.9)')
plt.show()

# Save key outputs to files
os.makedirs('/mnt/data/grid_outputs', exist_ok=True)
plt.savefig('/mnt/data/grid_outputs/steps_plot.png')  # last plt saved
# Also save policy/value arrays for gamma=0.9
with open('/mnt/data/grid_outputs/results.pkl', 'wb') as f:
    pickle.dump(results, f)

print("Saved outputs to /mnt/data/grid_outputs. Files: steps_plot.png, results.pkl")
