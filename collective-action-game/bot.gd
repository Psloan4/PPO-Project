class_name Bot
extends CharacterBody2D

const SPEED = 300.0
const JUMP_VELOCITY = -450.0
const VISION_DISTANCE = 4

var movement_frame = 0
@onready var sprite_2d: Sprite2D = $Sprite2D

var color: Color = Color.WHITE
var peer: PacketPeerUDP
var bot_id: int = 1

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
		#print("FRAME:")
		#print(movement_frame)
		#print(jump_val)
		#print("")
		sprite_2d.frame = floor(movement_frame / 3.0) + jump_val
	
	move_and_slide()
	var scan_data = scan_area()
	print_matrix(scan_data)

func get_body_data(body: Node2D) -> int:
	if body is StaticBody2D:
		return 5
	elif body is CharacterBody2D:
		return body.bot_id
	elif body is Area2D:
		return 6
	return -1

func get_point_data(point: Vector2) -> int:
	var space_state = get_world_2d().direct_space_state
	var params = PhysicsPointQueryParameters2D.new()
	params.position = point
	params.collide_with_bodies = true
	params.collide_with_areas = true
	var results = space_state.intersect_point(params, 3)
	
	if results.size() == 0:
		return 0
	var result_data = []
	for result in results:
		result_data.append(get_body_data(result["collider"]))
	return result_data.max()


func scan_area() -> Array:
	const POINT_OFFSET = 64
	var data = []
	for dy in range(-VISION_DISTANCE, VISION_DISTANCE + 1):
		var row_data = []
		for dx in range(-VISION_DISTANCE, VISION_DISTANCE + 1):
			var point := global_position + Vector2(dx, dy) * POINT_OFFSET
			row_data.append(get_point_data(point))
		data.append(row_data)
	return data

func print_matrix(matrix: Array) -> void:
	for row in matrix:
		var row_string = ""
		for value in row:
			row_string += str(value) + " "
		print(row_string.strip_edges())
	print("")
