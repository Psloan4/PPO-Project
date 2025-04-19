import sys
import select
import torch
import os
#import pickle
import threading

from Bot import Bot
from NetworkTrainer import NetworkTrainer
from SimpleNetwork import SimpleNet

MAX_BOTS = 4
DIR_PATH = "./models/"

def iteration(network, batch_size=4):
    # go through a batch of episodes and collect data
    print("Beginning iteration...")
    rollout_data = rollout(network, batch_size)

    # rollout complete, now train on the data
    trainer = NetworkTrainer(network)
    trainer.train(rollout_data)
    

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
        network = SimpleNet()  # Initialize the model
        network.load_state_dict(torch.load(file_path))  # Load the weights from file
        network.eval()  # Set to eval mode if you're not training
    else:
        print(f"{file_path} not found, creating a new model...")
        network = SimpleNet()  # Initialize a new model
    return network

def main():
    # get the network from storage if available
    file_path = DIR_PATH + "test_file.pth"
    network = download_network(file_path)

    # while true iterate
    print("Running optimisation sequence, enter 'q' at any time to quit...")
    while True:
        iteration(network)
        i, o, e = select.select([sys.stdin], [], [], 1)  # wait 1 second
        if i:
            user_input = sys.stdin.readline().strip()
            if user_input.lower() == 'q':
                break
    
    # sequence complete, save then exit
    print("Completed optimisation sequence")
    save_input = input("Do you want to save the model? (y/n): ").strip().lower()
    if save_input == 'y':
        # Save the model parameters to the file
        torch.save(network.state_dict(), file_path)
        print(f"Model saved to {file_path}")
    print("Exiting...")

main()