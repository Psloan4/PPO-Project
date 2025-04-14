class_name Level
extends Node2D

@export var goal: Area2D

func _ready() -> void:
	goal.body_entered.connect(win)

func win(_body):
	print("players win")
