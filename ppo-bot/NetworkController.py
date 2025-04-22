import sys
import select
import torch
import os
#import pickle
import threading
import matplotlib.pyplot as plt

from Bot import Bot
from NetworkTrainer import NetworkTrainer
from SimpleNetwork import SimpleNet
from DualStreamNet import DualStreamNet
from BigNet import BigNet
from SimplerNet import SimplerNet
from TinyNet import TinyNet

MAX_BOTS = 1
DIR_PATH = "./models/"

def iteration(trainer, network, batch_size=10):
    # go through a batch of episodes and collect data
    print("Beginning iteration...")
    rollout_data = rollout(network, batch_size)

    # rollout complete, now train on the data
    return trainer.train(rollout_data)
    

def rollout(network, batch_size):
    network.eval()
    rollout_data = []
    for i in range(batch_size):
        print("Running episode ", i)
        episode_data = run_episode(network)

        rollout_data.extend(episode_data)
    return rollout_data


def run_episode(network):
    results = []
    threads = []

    def thread_target(i):
        bot_results = run_bot_instance(network)
        results.append(bot_results)
    
    for i in range(MAX_BOTS):
        thread = threading.Thread(target=thread_target, args=(i,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    print("Data collected")
    return results

def run_bot_instance(network):
    bot = Bot(network)
    bot.connect()
    return bot.run()

def download_network(file_path):
    # Check if model file exists
    if os.path.exists(file_path):
        print(f"Loading model from {file_path}...")
        network = DualStreamNet()  # Initialize the model
        network.load_state_dict(torch.load(file_path))  # Load the weights from file
        network.eval()  # Set to eval mode if you're not training
    else:
        print(f"{file_path} not found, creating a new model...")
        network = DualStreamNet()  # Initialize a new model
    return network

def plot_logs(logs):
    # Create a new figure
    plt.figure(figsize=(10, 5))

    # Loop through each log in the dictionary and plot it
    for key, values in logs.items():
        plt.plot(values, label=key)

    # Add labels and title
    plt.xlabel('Training Steps')
    plt.ylabel('Loss / Metric Value')
    plt.title('Training Loss and Metrics Over Time')

    # Add legend
    plt.legend()

    # Show grid for better readability
    plt.grid(True)

    # Ensure everything fits within the layout
    plt.tight_layout()

    # Display the plot
    plt.show()

def plot_rewards(rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='Average Reward', color='green')
    plt.xlabel('Rollouts')
    plt.ylabel('Average Reward')
    plt.title('Average Reward Per Rollout Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    # get the network from storage if available
    file_path = DIR_PATH + "none.pth"
    network = download_network(file_path)
    print("parameters: ", count_parameters(network))

    # while true iterate

    stop_event = threading.Event()

    def wait_for_input():
        while True:
            user_input = sys.stdin.readline().strip()
            if user_input.lower() == 'q':
                stop_event.set()
                break

    threading.Thread(target=wait_for_input, daemon=True).start()

    print("Running optimisation sequence, enter 'q' at any time to quit...")

    logs = {
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "total_loss": [],
    }
    rewards = []
    trainer = NetworkTrainer(network)
    while not stop_event.is_set():
        iter_logs, avg_reward = iteration(trainer, network)
        for log in iter_logs:
            logs[log].extend(iter_logs[log])
            rewards.append(avg_reward)
        # i, o, e = select.select([sys.stdin], [], [], 1)  # wait 1 second
        # if i:
        #     user_input = sys.stdin.readline().strip()
        #     if user_input.lower() == 'q':
        #         break
    
    # sequence complete, save then exit
    print("Completed optimisation sequence")
    plot_logs(logs)
    plot_rewards(rewards)
    save_input = input("Do you want to save the model? (y/n): ").strip().lower()
    if save_input == 'y':
        # Save the model parameters to the file
        torch.save(network.state_dict(), file_path)
        print(f"Model saved to {file_path}")
    print("Exiting...")

main()

