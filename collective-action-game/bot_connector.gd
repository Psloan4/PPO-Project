class_name BotConnector
extends Node

const PORT = 3000

var server := UDPServer.new()
var bots_connected := 0
const MAX_BOTS = 4

signal bot_connected

func _ready():
	print("Starting server...")
	server.listen(PORT)

func _process(_delta):
	if bots_connected >= MAX_BOTS: return
	server.poll()
	if server.is_connection_available():
		var peer: PacketPeerUDP = server.take_connection()
		print("Received connection from " + peer.get_packet_ip())
		bot_connected.emit(peer)
		peer.put_packet("connection_response".to_utf8_buffer())
