class RenderData:
    fps: int

    max_iterations: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    zoom_factor: float
    zoom_speed: float
    zoom_sign: int
    zoom_position_x: float
    zoom_position_y: float
    zoom_duration: int
    rotation_angle_degrees: float
    rotation_speed: float

    kick: int
    kick_max: int

    constant: complex
    color_palette: list
    color_palette_list: list
    color_step: int
    current_color: int
    color_maps: list
