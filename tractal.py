import pygame
import numpy as np
from numba import jit, prange
import math
import random
from concurrent.futures import ThreadPoolExecutor
import threading


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

    constant: complex
    color_palette: list
    color_palette_list: list
    color_step: int
    current_color: int

@jit(nopython=True, parallel=True, fastmath=True)
def mandelbrot(c, max_iterations, constant = None):
    height, width = c.shape
    iterations = np.zeros((height, width), dtype=np.int64)

    for y in prange(height):
        for x in prange(width):
            zy, zx = 0, 0
            c_value = c[y, x]
            for i in range(max_iterations):
                zy, zx = zy * zx * 2 + c_value.imag, zx * zx - zy * zy + c_value.real
                if zx * zx + zy * zy > 4:
                    break
                iterations[y, x] = i

    return iterations

@jit(nopython=True, parallel=True, fastmath=True)
def julia(c, max_iterations, julia_constant):
    height, width = c.shape
    iterations = np.zeros((height, width), dtype=np.int64)

    for y in prange(height):
        for x in prange(width):
            zx, zy = c[y, x].real, c[y, x].imag
            for i in range(max_iterations):
                new_zx = zx * zx - zy * zy + julia_constant.real
                new_zy = 2 * zx * zy + julia_constant.imag
                zx, zy = new_zx, new_zy
                
                # Escape time optimization
                if abs(zx) > 2.0 or abs(zy) > 2.0:
                    break
                
                iterations[y, x] = i # Update iteration count
                
                # Early escape condition check
                if i > 0 and (zx, zy) == (c[y, x].real, c[y, x].imag):
                    iterations[y, x] = max_iterations
                    break

    return iterations

def calculate_fractal(data: RenderData, width, height, fractal_set):
    real, imag = np.meshgrid(
        np.linspace(data.x_min, data.x_max, width), np.linspace(data.y_min, data.y_max, height)
    )
    c = real + 1j * imag
    iterations = fractal_set(c, data.max_iterations, data.constant)

    # Precompute color mapping
    color_mapping = np.array(data.color_palette[:data.max_iterations], dtype=np.uint8)

    # Use direct mapping for colors
    colors = color_mapping[iterations]

    return colors

def display_fractal(screen, colors, clock):
    pygame.surfarray.blit_array(screen, colors.swapaxes(0, 1))

    fps = clock.get_fps()
    fps_font = pygame.font.Font(None, 20)
    fps_text = fps_font.render(str(int(fps)), True, (200, 200, 200))
    screen.blit(fps_text, (10, 10))

    pygame.display.flip()


def init_mandelbrot():
    data = RenderData()

    # Set up the clock for controlling the frame rate
    data.fps = 60

    # Mandelbrot parameters
    data.max_iterations = 200
    data.x_min = -2.0
    data.x_max = 1.0
    data.y_min = -1.5
    data.y_max = 1.5
    data.zoom_factor = 0.995  # Zoom factor (0 to 1)
    data.zoom_sign = 1
    data.zoom_position_x = -0.7462
    data.zoom_position_y = -0.1495  # Seahorse Valley zoom position
    data.zoom_duration = 900  # Number of zoom iterations

    # Define color variables
    data.red_multiplier = 1.5
    data.green_multiplier = 0.3
    data.blue_multiplier = 1.7

    data.cx = 0
    data.cy = 0
    data.constant = 0
    return data, mandelbrot

def init_julia():
    data = RenderData()

    # Set up the clock for controlling the frame rate
    data.fps = 60

    # Julia set parameters
    data.max_iterations = 3000
    data.x_min = -2.0
    data.x_max = 2.0
    data.y_min = -2.0
    data.y_max = 2.0
    data.zoom_factor = 0.985  # Zoom factor (0 to 1)
    data.zoom_speed = 1 - data.zoom_factor
    data.zoom_sign = 1
    # data.zoom_position_x = -0.527504221
    # data.zoom_position_y = 0.075911712  # Set center for Julia set
    data.zoom_position_x = 0
    data.zoom_position_y = 0  # Set center for Julia set
    data.zoom_duration = 500  # Number of zoom iterations

    data.cx = -0.8
    data.cy = 0.156

    data.constant = complex(-0.8, 0.156)

    data.color_palette = []
    data.color_palette_list = []
    data.color_step = 100
    data.current_color = 0

    return data, julia


def edit_var(data, zoom_iteration, i):
    if data.color_step == 0:
        data.current_color = (data.current_color + 1) % 20
        data.color_step = 100
    data.color_palette = transform_palette_iterative(data.color_palette, data.color_palette_list[(data.current_color + 1) % 20], data.color_step)
    data.color_step -= 1

    if data.zoom_sign == 1:
        zoom_iteration = data.zoom_duration - zoom_iteration
    data.zoom_factor = 1 - data.zoom_sign * data.zoom_speed * zoom_iteration / data.zoom_duration
    data.cx = -0.8 + 0.00003 * zoom_iteration
    data.cy = 0.156 - 0.00001 * zoom_iteration
    data.constant = complex(data.cx, data.cy)
    return data

def luminance(color):
    return np.dot(color, [0.2126, 0.7152, 0.0722])

def generate_palette(points, num_colors):
    sorted_points = np.array(sorted(points, key=lambda x: (luminance(x), points.index(x))))
    num_points = len(sorted_points)

    colors_between_points = num_colors // (num_points - 1)
    remainder = num_colors % (num_points - 1)

    palette = []
    for i in range(num_points - 1):
        color1 = sorted_points[i]
        color2 = sorted_points[i + 1]

        colors_for_pair = colors_between_points + (1 if i < remainder else 0)

        ratios = np.linspace(0, 1, colors_for_pair)
        colors = color1 * (1 - ratios[:, np.newaxis]) + color2 * ratios[:, np.newaxis]
        palette.extend(colors.astype(np.int32).tolist())

    return palette


def transform_palette_iterative(color_palette, target_palette, steps):
    for i in range(len(color_palette)):
        for channel in range(3):
            color_palette[i][channel] -= (color_palette[i][channel] - target_palette[i][channel]) / steps
    return color_palette

def calculate_zoom(data, current_zoom, zoom_iteration):
    if zoom_iteration < data.zoom_duration:
        t = zoom_iteration / data.zoom_duration
        current_zoom += (data.zoom_factor - current_zoom) * t
        zoom_iteration += 1
    else:
        if data.zoom_factor > 1:
            data.zoom_position_x = -0.527504221
            data.zoom_position_y = 0.075911712
            data.zoom_duration = 1200
        data.zoom_sign *= -1
        zoom_iteration = 0

    # Update the boundaries with the new zoom
    if current_zoom != 1.0:
        # Calculate the new center based on the zoom position
        zoom_x = (data.zoom_position_x - data.x_min) / (data.x_max - data.x_min)
        zoom_y = (data.zoom_position_y - data.y_min) / (data.y_max - data.y_min)

        # Calculate the new range based on the zoom factor
        range_x = (data.x_max - data.x_min) * current_zoom
        range_y = (data.y_max - data.y_min) * current_zoom

        # Update the boundaries with the new zoom
        data.x_min = data.zoom_position_x - range_x * zoom_x
        data.x_max = data.zoom_position_x + range_x * (1 - zoom_x)
        data.y_min = data.zoom_position_y - range_y * zoom_y
        data.y_max = data.zoom_position_y + range_y * (1 - zoom_y)
    return data, current_zoom, zoom_iteration

def display(data: RenderData, fractal_set):
    # Pygame initialization
    pygame.init()
    width, height = 800, 800
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Fractal")
    clock = pygame.time.Clock()

    for i in range(20):
        nb_color = 10
        rng_colors = [[0, 0, 0]] * nb_color
        rng_colors[-1] = [255,255,255]
        for i in range(1,nb_color-1):
            rng_colors[i] = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
        data.color_palette_list.append(generate_palette(rng_colors, data.max_iterations))
    data.color_palette = data.color_palette_list[0]

    zoom_iteration = 0
    running = True
    current_zoom = 1.0
    i=0
    terminate_flag = False
    lock = threading.Lock()

    def render_frame():
        nonlocal i, zoom_iteration, current_zoom, data, terminate_flag, lock
        while not terminate_flag:
            with lock:  # Acquire lock before accessing shared data
                colors = calculate_fractal(data, width, height, fractal_set)
                display_fractal(screen, colors, clock)
                data = edit_var(data, zoom_iteration, i)
                data, current_zoom, zoom_iteration = calculate_zoom(data, current_zoom, zoom_iteration)
                i += 1

    with ThreadPoolExecutor() as executor:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        terminate_flag = True

            # i+=1
            # colors = calculate_fractal(data, width, height, fractal_set)
            # display_fractal(screen, colors, clock)
            # data = edit_var(data, zoom_iteration, i)
            executor.submit(render_frame)
            # data, current_zoom, zoom_iteration = calculate_zoom(data, current_zoom, zoom_iteration)

            # Add a delay to control the frame rate
            clock.tick(data.fps)

    pygame.quit()


if __name__ == "__main__":
    # data, fractal_set = init_mandelbrot()
    data, fractal_set = init_julia()
    display(data, fractal_set)
