import socket
import threading
import time

class Bot:
    def __init__(self, network, server_ip='127.0.0.1', server_port=3000):
        self.network = network
        self.server_ip = server_ip
        self.server_port = server_port

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.is_connected = False

    def connect(self):
            """ Attempt to send a message to the server to establish a connection """
            try:
                message = "connection_request".encode()
                self.sock.sendto(message, (self.server_ip, self.server_port))
                print(f"Bot: Sent message to {self.server_ip}:{self.server_port}")

                # Wait for a response from the server
                data, addr = self.sock.recvfrom(1024)  # buffer size is 1024 bytes
                print(f"Bot: Received response from server: {data.decode()}")

                # If we received a valid response, consider it connected
                self.is_connected = True

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
                print(f"Bot: Sent data: {data}")
            except Exception as e:
                print(f"Bot: Failed to send data: {e}")
        else:
            print("Bot: Not connected to the server. Cannot send data.")

    def receive_data(self):
        """ Receive data from the UDP server """
        try:
            data, addr = self.sock.recvfrom(1024)  # buffer size is 1024 bytes
            print(f"Bot: Received data: {data.decode()}")
            return data.decode()
        except Exception as e:
            print(f"Bot: Failed to receive data: {e}")
            return None

    def close(self):
        """ Close the UDP connection gracefully """
        self.sock.close()
        print("Bot: Connection closed.")

    def run(self):
        """ Simulate bot's main loop for interaction """
        while self.is_connected:
            # Example bot activity: receive and process data, then send a response
            received = self.receive_data()
            if received:
                # Perform some action, for example, sending a response back to the server
                self.send_data(f"Bot responding to: {received}")
            time.sleep(1)  # Simulate work (e.g., bot logic)