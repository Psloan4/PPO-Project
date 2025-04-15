class_name Level
extends Node2D

@export var goal: Area2D

@onready var bot_connector = $BotConnector as BotConnector
@onready var bot_manager = $BotManager as BotManager

func _ready() -> void:
	goal.body_entered.connect(win)
	bot_connector.bot_connected.connect(bot_manager.create_bot)
	

func win(_body):
	print("players win")
