class_name BotConnector
extends Node

const PORT = 3000

var server := UDPServer.new()
var bots_connected := 0
const MAX_BOTS = 4

signal bot_connected
signal episode_ready

func _ready():
	print("Starting server...")
	server.listen(PORT)

func _process(_delta):
	server.poll()
	if bots_connected >= MAX_BOTS: return
	if server.is_connection_available():
		var peer: PacketPeerUDP = server.take_connection()
		if peer.get_available_packet_count() > 0:
			var packet = peer.get_packet().get_string_from_utf8()
			if packet == "connection_request":
				print("Received connection from " + peer.get_packet_ip())
				bot_connected.emit(peer)
				bots_connected += 1
				if bots_connected == MAX_BOTS:
					get_tree().create_timer(.5).timeout.connect(episode_ready.emit)
				peer.put_packet("connection_response".to_utf8_buffer())

func reset_server():
	bots_connected = 0
