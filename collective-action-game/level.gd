class_name Level
extends Node2D

const EPISODE_TIME = 15

@export var goal: Area2D

@onready var bot_connector = $BotConnector as BotConnector
@onready var bot_manager = $BotManager as BotManager
@onready var episode_timer = $EpisodeTimer


func _ready() -> void:
	goal.body_entered.connect(end_episode.bind("win"))
	bot_connector.bot_connected.connect(bot_manager.create_bot)
	bot_manager.goal_position = goal.global_position
	bot_connector.episode_ready.connect(begin_episode)
	episode_timer.timeout.connect(end_episode.bind(null, "lose"))

func begin_episode():
	print("Episode started")
	bot_manager.start_bots()
	episode_timer.start(EPISODE_TIME)

func end_episode(_winner: Node2D, end_state: String):
	print("Episode ended")
	print("bots " + end_state)
	if not episode_timer.is_stopped():
		episode_timer.stop()
	bot_manager.stop_bots(end_state)
	bot_connector.reset_server()
