from matplotlib import pyplot as plt
import numpy as np


def visualize_policy(policy):

    usable_ace_actions = np.zeros(shape=[22, 10], dtype=np.int32)
    non_usable_ace_actions = np.zeros(shape=[22, 10])

    for sum in range(22):
        for dealer_card in range(1, 11):
            usable_ace_state = (sum, dealer_card, True)
            non_usable_ace_state = (sum, dealer_card, False)

            usable_ace_action = policy(usable_ace_state)
            non_usable_ace_action = policy(non_usable_ace_state)

            usable_ace_actions[sum, dealer_card-1] = usable_ace_action
            non_usable_ace_actions[sum, dealer_card-1] = non_usable_ace_action

    def plot_policy(actions, title="policy"):
        plt.figure()
        plt.title = title
        plt.xlabel("Dealer card")
        plt.ylabel("Player sum")
        # plt.xlim([0, 21])
        plt.xticks(np.arange(1, 11))
        plt.yticks(np.arange(21))
        plt.imshow(actions, cmap='hot')

    plot_policy(usable_ace_actions, title="Usable ace policy")
    plot_policy(non_usable_ace_actions, title="No usable ace policy")
    plt.show()