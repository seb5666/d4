import gym
import numpy as np
from blackjack.policies.simple_policy import SimplePolicy


# generate episode in env iusing the current policy
def generate_episode(env, policy):
    states = []
    actions = []
    rewards = []

    current_state = env.reset()

    done = False
    while not(done):
        action = policy(current_state)
        next_state, reward, done, _ = env.step(action)

        states.append(current_state)
        actions.append(action)
        rewards.append(reward)

        current_state = next_state

    return states, actions, rewards


# Compute returns for a generated episode
# Assuming non-discounted rewards, the return at any point is the sum of all following rewards.
# Since all rewards are 0 except -1 /1 in the last step the return at any point is just the last reward of the episode
def compute_returns(rewards):
    return np.full(len(rewards), fill_value=rewards[-1])

# TODO: Update policy


env = gym.make('Blackjack-v0')

policy = SimplePolicy(sketch_path='policies/sketches/simple_policy.d4')

while True:
    (states, actions, rewards) = generate_episode(env, policy)
    for s, a, r in zip(states, actions, rewards):
        print(s,a,r)
    returns = compute_returns(rewards)
    break
    update_policy(policy, episode)
