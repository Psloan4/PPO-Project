shader_type canvas_item;

uniform vec4 replacement_color;

void fragment() {
    vec4 tex_color = texture(TEXTURE, UV);

    // Check for "white" with a tiny tolerance to account for float rounding
    if (tex_color.rgb == vec3(1.0)) {
        COLOR = vec4(replacement_color.rgb, tex_color.a); // preserve original alpha
    } else {
        COLOR = tex_color;
    }
}
