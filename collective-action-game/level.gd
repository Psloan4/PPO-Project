class_name Level
extends Node2D

const EPISODE_TIME = 30

@export var goal: Area2D

@onready var bot_connector = $BotConnector as BotConnector
@onready var bot_manager = $BotManager as BotManager
@onready var episode_timer = $EpisodeTimer

@export var spawn1: Node2D
@export var spawn2: Node2D
@export var spawn3: Node2D
@export var spawn4: Node2D

func _ready() -> void:
	var spawn_points: Array[Node2D] = [spawn1, spawn2, spawn3, spawn4]
	bot_manager.spawn_points = spawn_points
	bot_manager.round_spawn_points = spawn_points.duplicate()
	
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
	reposition_goal_and_spawn()
	if not episode_timer.is_stopped():
		episode_timer.stop()
	bot_manager.stop_bots(end_state)
	bot_connector.reset_server()

func reposition_goal_and_spawn():
	var min_x = 100
	var max_x = 1052
	var min_distance = 100

	# Move goal to a new position
	var new_goal_x = randf_range(min_x, max_x)
	var goal_y = goal.global_position.y
	goal.global_position.x = new_goal_x

	# Try random positions for spawn1 until it's far enough from the goal
	var spawn_x = 0
	while true:
		spawn_x = randf_range(min_x, max_x)
		if abs(spawn_x - new_goal_x) >= min_distance:
			break

	var spawn_y = spawn1.global_position.y
	spawn1.global_position.x = spawn_x
