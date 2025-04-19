import socket
import json
import time
import numpy as np
import torch
import math
import torch.distributions as D

class Bot:
    def __init__(self, network, server_ip='127.0.0.1', server_port=3000):
        self.network = network
        self.server_ip = server_ip
        self.server_port = server_port

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.is_connected = False

        self.max_x = 1152
        self.max_y = 648
        self.action_interval = .2

    def connect(self):
        """ Attempt to send a message to the server to establish a connection """
        try:
            message = "connection_request".encode()
            self.sock.sendto(message, (self.server_ip, self.server_port))
            # print(f"Bot: Sent message to {self.server_ip}:{self.server_port}")

            # Wait for a response from the server
            data, addr = self.sock.recvfrom(1024)  # buffer size is 1024 bytes
            # print(f"Bot: Received response from server: {data.decode()}")

            # If we received a valid response, consider it connected
            self.is_connected = True
            self.sock.setblocking(False)

        except socket.timeout:
            print("Bot: Server did not respond in time. Connection failed.")
        except Exception as e:
            print(f"Bot: An error occurred while connecting to the server: {e}")

    def send_data(self, data):
        """ Send data to the UDP server """
        if self.is_connected:
            try:
                message = data.encode()
                self.sock.sendto(message, (self.server_ip, self.server_port))
                #print(f"Bot: Sent data: {data}")
            except Exception as e:
                print(f"Bot: Failed to send data: {e}")
        else:
            print("Bot: Not connected to the server. Cannot send data.")

    def receive_data(self):
        """ Receive data from the UDP server """
        try:
            data, addr = self.sock.recvfrom(1024)  # buffer size is 1024 bytes
            #print(f"Bot: Received data: {data.decode()}")
            return data.decode()
        except Exception as e:
            # print(f"Bot: Failed to receive data: {e}")
            return {}

    def close(self):
        """ Close the UDP connection gracefully """
        self.sock.close()
        self.is_connected = False
        # print("Bot: Connection closed.")


    def build_input_tensor(self, game_state):
        bot_pos = np.array(game_state['bot_data'], dtype=np.float32)
        goal_pos = np.array(game_state['goal_data'], dtype=np.float32)
        
        normalized_bot = bot_pos / np.array([self.max_x, self.max_y], dtype=np.float32)
        normalized_goal = goal_pos / np.array([self.max_x, self.max_y], dtype=np.float32)

        scan = np.array(game_state['scan_data'], dtype=np.float32) / 3.0
        flat_scan = scan.flatten()

        full_obs = np.concatenate([normalized_bot, normalized_goal, flat_scan])

        return torch.tensor(full_obs, dtype=torch.float32)

    def run(self):
        """ Simulate bot's main loop for interaction """
        episode_started = False
        while not episode_started:
            received = self.receive_data()
            if received:
                try:
                    data = json.loads(received)

                    if "start_episode" in data:
                        print("Episode started")
                        episode_started = True

                except json.JSONDecodeError:
                    print("Invalid JSON received")
        
        # replace this with running a timestep, which includes reading data until it is time to take an action, get that action, send it, and then loop again
        self.episode_data = []
        self.last_action_time = time.time()
        finished = False
        while not finished:
            finished = self.run_timestep()
        return self.episode_data
        

    def run_timestep(self) -> bool:
        data = {}

        while time.time() - self.last_action_time < self.action_interval:
            received = self.receive_data()
            if received:
                try:
                    data = json.loads(received)

                    if "end_episode" in data:
                        print("Episode result: ", data["end_episode"])
                        self.episode_data[-1]["done"] = True
                        self.episode_data[-1]["reward"] += 10
                        self.close()
                        return True
                    elif "game_state" in data:
                        #print("received game state: ", data["game_state"])
                        self.latest_game_state = data["game_state"]
                except json.JSONDecodeError:
                    print("Invalid JSON received")
        
        input_tensor = self.build_input_tensor(self.latest_game_state)

        self.last_action_time = time.time()
        action_info = self.get_action(input_tensor)
        self.send_data(json.dumps(action_info["action_packet"]))

        # save timestep
        self.episode_data.append({
            "state": input_tensor.detach().cpu().numpy().tolist(),
            "action": action_info["action"],
            "log_prob": action_info["log_prob"],
            "reward": self.get_distance_reward(self.latest_game_state["bot_data"], self.latest_game_state["goal_data"]),
            "value": action_info["value"],
            "done": False,
        })
        # print("reward: ", self.episode_data)

        return False
        
        
    def get_action(self, input_tensor):
        with torch.no_grad():
            
            action_logits, state_value = self.network(input_tensor)
            action_dist = D.Categorical(logits=action_logits)
            action_idx = action_dist.sample()

            # Define the 6 possible actions
            actions = [
                ("left", True),   # left_jump
                ("left", False),  # left_stay
                ("left", False),  # left_no_jump
                ("right", True),  # right_jump
                ("right", False), # right_stay
                ("right", False)  # right_no_jump
            ]

            # Get the direction and jump for the sampled action
            direction, jump = actions[action_idx.item()]

            # Return the action packet, log probabilities, and the raw action index
            return {
                "action_packet": {
                    "direction": direction,
                    "jump": jump
                },
                "action": action_idx.item(),
                "log_prob": action_dist.log_prob(action_idx).item(),  # Log probability of the sampled action
                "value": state_value.item(),
            }

    def get_distance_reward(self, bot_data, goal_data) -> float:
        distance = math.sqrt((goal_data[0] - bot_data[0])**2 + (goal_data[1] - bot_data[1])**2)
        reward = -distance / 700
        #print("reward: ", reward)
        return reward 