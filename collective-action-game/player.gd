extends CharacterBody2D


const SPEED = 300.0
const JUMP_VELOCITY = -450.0

var frame_counter = 0
@onready var sprite_2d: Sprite2D = $Sprite2D

@export var disabled: bool

func _physics_process(delta: float) -> void:
	if disabled: return
	# Add the gravity.
	if not is_on_floor():
		velocity += get_gravity() * delta

	# Handle jump.
	if Input.is_action_just_pressed("ui_accept") and is_on_floor():
		velocity.y = JUMP_VELOCITY

	# Get the input direction and handle the movement/deceleration.
	# As good practice, you should replace UI actions with custom gameplay actions.
	var direction := Input.get_axis("ui_left", "ui_right")
	if direction:
		velocity.x = direction * SPEED
	else:
		velocity.x = move_toward(velocity.x, 0, SPEED)
	
	if velocity.x != 0:
		frame_counter = (frame_counter + 1)
		if frame_counter > 6:
			frame_counter = 0
			sprite_2d.frame = (sprite_2d.frame + 1) % 2
	
	move_and_slide()
