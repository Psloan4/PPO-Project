import sys
import select
import time
import os
import json

from Bot import Bot

MAX_BOTS = 4

def iteration(network, batch_size=8):
    # go through n batches of rollouts, and save their data, and return it
    data = rollout(network, batch_size)

def rollout(network, batch_size):
    for i in range(batch_size):
        run_episode(network)

def run_episode(network):
    processes = []
    pipes = []
    results = []

    for i in range(MAX_BOTS):
        read_fd, write_fd = os.pipe()
        pid = os.fork()

        if pid == 0: # child process
            os.close(read_fd)
            results = run_bot_instance(network)

            os.write(write_fd, json.dumps(results).encode())
            os.close(write_fd)
            os._exit(0)

        else: # parent process
            os.close(write_fd)
            processes.append(pid)
            pipes.append(read_fd)
    
    for i in range(MAX_BOTS):
        os.waitpid(processes[i], 0)
        with os.fdopen(pipes[i]) as pipe:
            data = pipe.read()
            result = json.loads(data)
            results.append(result)

def run_bot_instance(network):
    bot = Bot(network)
    bot.connect()
    bot.run()

def main():
    # get the network from storage if available

    #while true iterate
    print("Running optimisation sequence, enter 'q' at any time to quit...")
    while True:
        iteration(None)
        i, o, e = select.select([sys.stdin], [], [], 1)  # wait 1 second
        if i:
            user_input = sys.stdin.readline().strip()
            if user_input.lower() == 'q':
                break
    print("Completed optimisation sequence")

main()