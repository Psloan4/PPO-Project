import socket
import random
import string

# Generate a random message
message = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
print(f"Sending: {message}")

# Create UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Send the message to localhost on port 8000
sock.sendto(message.encode(), ('127.0.0.1', 3000))

# Close socket
sock.close()
