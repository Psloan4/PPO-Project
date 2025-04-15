class_name Bot
extends CharacterBody2D

const SPEED = 300.0
const JUMP_VELOCITY = -450.0

var movement_frame = 0
@onready var sprite_2d: Sprite2D = $Sprite2D

var color: Color = Color.WHITE
var peer: PacketPeerUDP

func _ready():
	var shader_mat = sprite_2d.material as ShaderMaterial
	shader_mat.set_shader_parameter("replacement_color", color)

func _physics_process(delta: float) -> void:
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
		movement_frame = (movement_frame + 1) % 6
		var jump_val := 0
		if not is_on_floor():
			jump_val = 2
		print("FRAME:")
		print(movement_frame)
		print(jump_val)
		print("")
		sprite_2d.frame = floor(movement_frame / 3.0) + jump_val
	
	move_and_slide()
