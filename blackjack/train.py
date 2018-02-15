import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from blackjack.policies.simple_policy import SimplePolicy
from blackjack.policies.policy_utils import visualize_policy

# generate episode in env iusing the current policy
def generate_episode(env, policy):
    states = []
    actions = []
    rewards = []

    current_state = env.reset()

    done = False
    while not(done):
        action = policy(state=current_state)
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

env = gym.make('Blackjack-v0')


show_plots = True

with tf.Session() as sess:
    policy = SimplePolicy(sess=sess, sketch_path='policies/sketches/simple_policy.d4')

    print("Initializing global tf variables...")
    sess.run(tf.global_variables_initializer())

    tvars = tf.trainable_variables()
    tvars_ = sess.run(tvars)
    print("Initial parameters: {}".format(np.array(tvars_).squeeze()))


    print("Generating episodes...")
    iteration = 0
    max_iters = 10000

    if show_plots:
        visualize_policy(policy)

    R = []
    average_rewards = []
    while True and iteration < max_iters:
        iteration += 1
        (states, actions, rewards) = generate_episode(env, policy)
        returns = compute_returns(rewards)
        for state, action, total_return in zip(states, actions, returns):
            policy.update_policy(state, action, total_return)

        R.append(rewards[-1])
        if len(R) > 100:
            R = R[1:]

        if iteration % 1000 == 0:
            average_return = 0
            sim_length = 100
            for j in range(sim_length):
                (states, actions, rewards) = generate_episode(env, policy)
                returns = compute_returns(rewards)
                average_return += returns[0]
            average_return /= sim_length
            average_rewards.append(average_return)
            print("Average return after {} iterations over the last 100 episodes: {}".format(iteration, average_return))

    tvars = tf.trainable_variables()
    tvars_ = sess.run(tvars)
    print("Final parameters: {}".format(np.array(tvars_).squeeze()))

    print("Average rewards:\n{}".format(average_rewards))
    
    if show_plots:
        visualize_policy(policy)
        plt.figure()
        plt.plot(np.arange(len(average_rewards)), average_rewards)
        plt.show()