class_name BotManager
extends Node

const BOT = preload("res://bot.tscn")

const SPAWN_Y := 569
var spawn_x := 150
const SPAWN_X_INC := 80

@export var bot_container: Node2D

var colors = [
	Color.RED,
	Color.SKY_BLUE,
	Color.ORANGE,
	Color.GREEN,
]

func _ready():
	colors.shuffle()

func create_bot(peer: PacketPeerUDP):
	var bot = BOT.instantiate()
	bot.peer = peer
	bot.color = colors.pop_front()
	bot.global_position = Vector2(spawn_x, SPAWN_Y)
	spawn_x += SPAWN_X_INC
	bot_container.add_child(bot)
