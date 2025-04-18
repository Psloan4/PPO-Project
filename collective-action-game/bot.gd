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
var goal_position: Vector2
var in_session := false
var movement_direction := 0
var jump := false


func _ready():
	var shader_mat = sprite_2d.material as ShaderMaterial
	shader_mat.set_shader_parameter("replacement_color", color)

func _physics_process(delta: float) -> void:
	if not is_on_floor():
		velocity += get_gravity() * delta
	
	
	if in_session:
		update_movement_input()
		apply_movement_input()
	animate_movement()
	move_and_slide()
	
	if in_session:
		distribute_game_state()

func update_player_movement_input():
	movement_direction = int(Input.get_axis("ui_left", "ui_right"))
	jump = Input.is_action_just_pressed("ui_accept")


func update_movement_input():
	if peer.get_available_packet_count() > 0:
		var packet := peer.get_packet()
		var json_string := packet.get_string_from_utf8()
		var action_data: Dictionary = JSON.parse_string(json_string)
		match action_data["direction"]:
			"right":
				movement_direction = 1
			"left":
				movement_direction = -1
			_:
				movement_direction = 0
		jump = action_data["jump"]

func apply_movement_input():
	if jump and is_on_floor():
		velocity.y = JUMP_VELOCITY
	if movement_direction:
		velocity.x = movement_direction * SPEED
	else:
		velocity.x = move_toward(velocity.x, 0, SPEED)

func animate_movement():
	const ANIMATION_FRAME_THRESHOLD = 8
	if velocity.x != 0:
		movement_frame = (movement_frame + 1) % ANIMATION_FRAME_THRESHOLD
	var jump_val := 0
	if not is_on_floor():
		jump_val = 2
	sprite_2d.frame = floor(movement_frame / (ANIMATION_FRAME_THRESHOLD / 2.0)) + jump_val

func distribute_game_state():
	var bot_data = [round(self.global_position.x), round(self.global_position.y)]
	var goal_data = [goal_position.x, goal_position.y]
	var scan_data = scan_area()
	var packet := {
		"bot_data": bot_data,
		"goal_data": goal_data,
		"scan_data": scan_data,
	}
	send_packet({"game_state": packet})
	#print_matrix(scan_data)

func send_packet(packet: Dictionary):
	peer.put_packet(JSON.stringify(packet).to_utf8_buffer())

func get_body_data(body: Node2D) -> int:
	if body is StaticBody2D: # Wall
		return 1
	elif body is CharacterBody2D: # Bot
		return 2#body.bot_id
	elif body is Area2D: # Goal
		return 3
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
