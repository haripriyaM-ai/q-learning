# EXP-07 : Q Learning Algorithm
### NAME : HARI PRIYA M
### REG NO : 212224240047

## AIM
To implement the Q-Learning algorithm for the FrozenLake environment and determine the optimal policy and state-value function.

## PROBLEM STATEMENT
The problem is to find an optimal policy for an agent navigating through the FrozenLake environment. The agent must learn which action to take in each state to reach the goal while maximizing the expected reward. Q-Learning is used to learn the action-value function through repeated interaction with the environment.

## Q LEARNING ALGORITHM
The steps involved in the Q-Learning algorithm are:

1. Initialize the Q-table with zeros for all states and actions.
2. Initialize the learning rate (alpha) and exploration rate (epsilon).
3. Select an action using the epsilon-greedy strategy.
4. Perform the selected action and observe the next state and reward.
5. Update the Q-value using the Q-Learning update equation.
6. Move to the next state and continue until the episode ends.
7. Gradually reduce alpha and epsilon during training.
8. Repeat the process for the specified number of episodes.
9. Derive the optimal policy by selecting the action with the highest Q-value for each state.
    
## Q LEARNING FUNCTION
### Name: HARI PRIYA M

### Register Number: 212224240047

```python
def q_learning(env,
               gamma=1.0,
               init_alpha=0.5,
               min_alpha=0.01,
               alpha_decay_ratio=0.5,
               init_epsilon=1.0,
               min_epsilon=0.1,
               epsilon_decay_ratio=0.9,
               n_episodes=3000):

    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)

    alphas = decay_schedule(
        init_alpha,
        min_alpha,
        alpha_decay_ratio,
        n_episodes
    )

    epsilons = decay_schedule(
        init_epsilon,
        min_epsilon,
        epsilon_decay_ratio,
        n_episodes
    )

    for e in tqdm(range(n_episodes), leave=False):

        state = env.reset()
        done = False

        while not done:

            if np.random.random() > epsilons[e]:
                action = np.argmax(Q[state])
            else:
                action = np.random.randint(nA)

            next_state, reward, done, _ = env.step(action)

            best_next_action = np.max(Q[next_state])

            Q[state][action] = Q[state][action] + alphas[e] * (
                reward
                + gamma * best_next_action * (not done)
                - Q[state][action]
            )

            state = next_state

        Q_track[e] = Q
        pi_track.append(np.argmax(Q, axis=1))

    V = np.max(Q, axis=1)

    pi = lambda s: {
        s: a for s, a in enumerate(np.argmax(Q, axis=1))
    }[s]

    return Q, V, pi, Q_track, pi_track

```



## OUTPUT

### Optimal Policy and Optimal Value Function
<img width="800" height="500" alt="image" src="https://github.com/user-attachments/assets/06d75e62-a193-4093-b775-0b0d15017aef" />


### Monte Carlo Method Results
<img width="800" height="500" alt="image" src="https://github.com/user-attachments/assets/6570b9c7-6379-420b-8f30-a7d56b840381" />


### Q-Learning Results
<img width="800" height="500" alt="image" src="https://github.com/user-attachments/assets/0957613c-f68c-4512-a338-992c9bea963e" />


### Comparison of State Value Functions of Monte Carlo Method and Q-Learning

#### Monte Carlo State-Value Function
<img width="700" height="300" alt="image" src="https://github.com/user-attachments/assets/03bc6d5f-00e0-4d64-b5ef-cc5520072a82" />

#### Q-Learning State-Value Function
<img width="700" height="300" alt="image" src="https://github.com/user-attachments/assets/af699cea-d519-4c33-933a-a3c187f3a4ef" />

## RESULT:

Thus, the Q-Learning algorithm was successfully implemented for the FrozenLake environment. The algorithm learned the action-value function through interaction with the environment and obtained an optimal policy for reaching the goal. The learned state-value function and policy were also compared with those obtained using the Monte Carlo method.
