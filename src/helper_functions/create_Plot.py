import matplotlib.pyplot as plt

def plot_results(reward_list: list, steps_list: list, q_list:list, loss_list:list, epsilon_list:list, save_path: str):
    """Plots the results of the training

    Args:
        reward_list (list): List of rewards
        steps_list (list): List of steps
        q_list (list): List of Q values
        loss_list (list): List of losses
        epsilon_list (list): List of epsilons
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(12, 12))

    plt.subplot(5, 1, 1) 
    plt.plot(reward_list, label="Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()

    plt.subplot(5, 1, 2)
    plt.plot(steps_list, label="Steps")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.legend()

    plt.subplot(5, 1, 3)
    plt.plot(q_list, label="Q Value")
    plt.xlabel("Episode")
    plt.ylabel("Q Value")
    plt.legend()

    plt.subplot(5, 1, 4)
    plt.plot(loss_list, label="Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(5, 1, 5)
    plt.plot(epsilon_list, label="Epsilon")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.legend()

    plt.tight_layout()
    
    # Save plot before showing
    plt.savefig(save_path)
    plt.show()