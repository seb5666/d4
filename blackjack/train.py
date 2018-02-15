import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from blackjack.policies.simple_policy import SimplePolicy

from blackjack.policies.policy_utils import visualize_policy


# generate an episode rollout in env using the current policy
def generate_episode(env, policy, argmax_stack = False):
    states = []
    actions = []
    rewards = []

    current_state = env.reset()

    done = False
    while not(done):
        action = policy(state=current_state, argmax_stack=argmax_stack)
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


if __name__ == "__main__":

    env = gym.make('Blackjack-v0')

    show_plots = True

    with tf.Session() as sess:
        policy = SimplePolicy(sess=sess, sketch_path='blackjack/policies/sketches/simple_policy.d4')

        print("Initializing global tf variables...")
        sess.run(tf.global_variables_initializer())

        tvars = tf.trainable_variables()
        tvars_ = sess.run(tvars)
        print("Initial parameters: {}".format(np.array(tvars_).squeeze()))

        iteration = 0
        max_iters = 10000
        eval_every = 100

        if show_plots:
            visualize_policy(policy, "Policy before training")

        average_rewards = []
        params = []

        while True and iteration < max_iters:
            iteration += 1

            (states, actions, rewards) = generate_episode(env, policy)
            returns = compute_returns(rewards)

            # Perform the REINFORCE update, i.e. once for each state/action pair visited
            for state, action, total_return in zip(states, actions, returns):
                policy.update_policy(state, action, total_return)

            if iteration % eval_every == 0:
                average_return = 0
                sim_length = 500
                for j in range(sim_length):
                    (states, actions, rewards) = generate_episode(env, policy, argmax_stack=True)
                    returns = compute_returns(rewards)
                    average_return += returns[0]

                average_return /= sim_length
                print("Average return after {} iterations over {} episodes: {}".format(iteration, sim_length, average_return))

                average_rewards.append(average_return)

                tvars_ = sess.run(tvars)
                tvars_ = np.array(tvars_).squeeze()
                params.append(tvars_)
                print("Parameters: {}".format(tvars_.squeeze()))

        tvars_ = sess.run(tvars)
        print("Final parameters: {}".format(np.array(tvars_).squeeze()))

        print("Average rewards:\n{}".format(average_rewards))

        if show_plots:
            params = np.array(params)
            plt.figure()
            plt.title("Weight params for the policy sketch")
            for weight in range(params.shape[1]):
                plt.plot(params[:, weight], label="weight {}".format(weight))
            plt.legend()

            visualize_policy(policy, "Policy after training")

            plt.figure()
            plt.title("Rewards over training steps")
            plt.plot(np.arange(1, max_iters, eval_every), average_rewards)
            plt.show()
