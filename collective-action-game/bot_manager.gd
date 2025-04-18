class_name BotManager
extends Node

const BOT = preload("res://bot.tscn")

const SPAWN_Y := 569
const SPAWN_X_DEF := 120
var spawn_x := SPAWN_X_DEF
const SPAWN_X_INC := 60

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
	bot.global_position = Vector2(spawn_x, SPAWN_Y)
	spawn_x += SPAWN_X_INC
	total_bots += 1
	bot.bot_id = total_bots
	bot.goal_position = goal_position
	bot_container.add_child(bot)
	#bot_data[str(bot.bot_id)] = [0, 0]

#func administer_bot_data():
	#for bot: Bot in bot_container.get_children():
		#

func start_bots():
	for bot: Bot in bot_container.get_children():
		bot.in_session = true
		bot.send_packet({"start_episode": null})

func stop_bots(end_state: String):
	color_i = 0
	spawn_x = SPAWN_X_DEF
	for bot: Bot in bot_container.get_children():
		bot.send_packet({"end_episode": end_state})
		bot_container.remove_child(bot)
		bot.queue_free()
