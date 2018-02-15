from matplotlib import pyplot as plt
import numpy as np


def visualize_policy(policy, title=""):

    usable_ace_actions = np.zeros(shape=[22, 10], dtype=np.int32)
    non_usable_ace_actions = np.zeros(shape=[22, 10])

    for sum in range(22):
        for dealer_card in range(1, 11):
            usable_ace_state = (sum, dealer_card, True)
            non_usable_ace_state = (sum, dealer_card, False)

            usable_ace_action = policy(usable_ace_state, argmax_stack=True)
            non_usable_ace_action = policy(non_usable_ace_state, argmax_stack=True)

            usable_ace_actions[sum, dealer_card-1] = usable_ace_action
            non_usable_ace_actions[sum, dealer_card-1] = non_usable_ace_action

    def plot_policy(ax, actions, title="policy"):
        ax.set_title(title)
        ax.set_xlabel("Dealer card")
        ax.set_ylabel("Player sum")
        # plt.xlim([0, 21])
        ax.set_xticks(np.arange(1, 11))
        ax.set_yticks(np.arange(21))
        ax.imshow(actions, cmap=plt.cm.Blues, origin='lower')

    print("Usable ace policy")
    print(usable_ace_actions)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(title)
    plot_policy(ax1, usable_ace_actions, title="Usable ace policy")
    plot_policy(ax2, non_usable_ace_actions, title="No usable ace policy")