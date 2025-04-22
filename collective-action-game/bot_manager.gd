class_name BotManager
extends Node

const BOT = preload("res://bot.tscn")

var spawn_points: Array[Node2D]
var round_spawn_points: Array[Node2D]

var total_bots := 0
var goal_position: Vector2
#var bot_data= {}

@export var bot_container: Node2D

const COLORS = [
	Color.RED,
	Color.SKY_BLUE,
	Color.ORANGE,
	Color.GREEN,
]
var color_i := 0

func create_bot(peer: PacketPeerUDP):
	var bot = BOT.instantiate()
	bot.peer = peer
	bot.color = COLORS[color_i]
	color_i += 1
	var spawn_point = round_spawn_points.pop_front()
	bot.global_position = spawn_point.global_position
	total_bots += 1
	bot.bot_id = total_bots
	bot_container.add_child(bot)
	#bot_data[str(bot.bot_id)] = [0, 0]

#func administer_bot_data():
	#for bot: Bot in bot_container.get_children():
		#

func start_bots(goal_position: Vector2):
	for bot: Bot in bot_container.get_children():
		bot.goal_position = goal_position
		bot.in_session = true
		bot.send_packet({"start_episode": null})

func stop_bots(end_state: String):
	color_i = 0
	round_spawn_points = spawn_points.duplicate()
	for bot: Bot in bot_container.get_children():
		bot.send_packet({"end_episode": end_state})
		bot_container.remove_child(bot)
		bot.queue_free()
