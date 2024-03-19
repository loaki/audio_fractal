import pygame
import numpy as np
from numba import jit, prange
import cmath
import math
import random
from concurrent.futures import ThreadPoolExecutor
import threading
import matplotlib.pyplot as plt

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

@jit(nopython=True, parallel=True, fastmath=True)
def mandelbrot(c, max_iterations, constant=None):
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


# @jit(nopython=True, parallel=True, fastmath=True)
# def julia(c, max_iterations, julia_constant, rotation_angle_degrees):
#     height, width = c.shape
#     iterations = np.zeros((height, width), dtype=np.int64)

#     rotation_angle_radians = np.radians(rotation_angle_degrees)
#     rotation_factor = cmath.exp(1j * rotation_angle_radians)

#     for y in prange(height):
#         for x in prange(width):
#             z = c[y, x]

#             for i in range(max_iterations):
#                 z *= z
#                 z += julia_constant

#                 if abs(z) > 2.0:
#                     break

#                 iterations[y, x] = i

#                 if i > 0 and z == c[y, x]:
#                     iterations[y, x] = max_iterations
#                     break

#     return iterations

@jit(nopython=True, parallel=True, fastmath=True)
def julia(c, max_iterations, julia_constant, rotation_angle_degrees):
    height, width = c.shape
    iterations = np.zeros((height, width), dtype=np.int64)

    rotation_angle_radians = np.radians(rotation_angle_degrees)
    rotation_factor = cmath.exp(1j * rotation_angle_radians)

    for y in prange(height):
        for x in prange(width):
            z = c[y, x] * rotation_factor  # Apply rotation
            for i in range(max_iterations):
                z = z * z + julia_constant

                # Escape time optimization
                if abs(z.real) > 2.0 or abs(z.imag) > 2.0:
                    break

                iterations[y, x] = i  # Update iteration count

                # Early escape condition check
                if i > 0 and z == c[y, x]:
                    iterations[y, x] = max_iterations
                    break

    return iterations


def calculate_fractal(data: RenderData, width, height, fractal_set, color_mapping):
    real, imag = np.meshgrid(
        np.linspace(data.x_min, data.x_max, width),
        np.linspace(data.y_min, data.y_max, height),
    )
    c = real + 1j * imag
    iterations = fractal_set(c, data.max_iterations, data.constant, data.rotation_angle_degrees)

    # Precompute color mapping

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
    data.max_iterations = 1000
    data.x_min = -2.0
    data.x_max = 1.0
    data.y_min = -1.5
    data.y_max = 1.5
    data.zoom_factor = 0.95  # Zoom factor (0 to 1)
    data.zoom_speed = 1 - data.zoom_factor

    data.zoom_sign = 1
    data.zoom_position_x = -0.7462
    data.zoom_position_y = -0.1495  # Seahorse Valley zoom position
    data.zoom_duration = 900  # Number of zoom iterations

    data.rotation_angle_degrees = 0

    data.cx = 0
    data.cy = 0
    data.constant = 0

    data.color_palette = []
    data.color_palette_list = []
    data.color_step = 50
    data.current_color = 0
    data.color_maps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
    
    return data, mandelbrot


def init_julia():
    data = RenderData()

    # Set up the clock for controlling the frame rate
    data.fps = 60

    # Julia set parameters
    data.max_iterations = 500
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
    data.zoom_duration = 600  # Number of zoom iterations
    data.rotation_angle_degrees = 0
    data.rotation_speed = 0.3

    data.cx = -0.8
    data.cy = 0.156

    data.constant = complex(-0.8, 0.156)

    data.kick = 0
    data.kick_max = 3
    data.color_palette = []
    data.color_palette_list = []
    data.color_step = 50
    data.current_color = 0
    # data.color_maps = [
    #         'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
    #         'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
    #         'gist_rainbow', 'rainbow', 'jet', 'turbo', 'nipy_spectral',
    #         'gist_ncar']
    # data.color_maps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
    # data.color_maps = [
    #         'Pastel1', 'Pastel2', 'Paired', 'Accent',
    #         'Dark2', 'Set1', 'Set2', 'Set3',
    #         'tab10', 'tab20', 'tab20b', 'tab20c']
    # data.color_maps = ['twilight', 'twilight_shifted', 'hsv']
    # data.color_maps = [
    #         'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
    #         'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']
    # data.color_maps = [
    #         'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
    #         'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
    #         'hot', 'afmhot', 'gist_heat', 'copper']
    data.color_maps = [
            'viridis', 'plasma', 'inferno', 'magma', 'cividis',
            'Dark2', 'Set1', 'Set2', 'Set3',
            'tab10', 'tab20', 'tab20b', 'tab20c',
            'twilight', 'twilight_shifted', 'hsv',
            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic',
            'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
            'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
            'hot', 'afmhot', 'gist_heat', 'copper']
    return data, julia


def edit_var(data, zoom_iteration, i):
    if data.color_step == 0:
        print(data.color_maps[(data.current_color + 1) % len(data.color_maps)])
        data.current_color = (data.current_color + 1) % len(data.color_maps)
        data.color_step = 100
    data.color_palette = transform_palette_iterative(
        data.color_palette,
        data.color_palette_list[(data.current_color + 1) % len(data.color_maps)],
        data.color_step,
    )
    data.color_step -= 1

    if data.zoom_sign == 1:
        zoom_iteration = data.zoom_duration - zoom_iteration
    data.zoom_factor = 1 - data.zoom_sign * data.zoom_speed * zoom_iteration / data.zoom_duration
    data.cx = -0.8 + 0.00003 * zoom_iteration
    data.cy = 0.156 - 0.00001 * zoom_iteration
    data.constant = complex(data.cx, data.cy)
    data.rotation_angle_degrees = (data.rotation_angle_degrees + data.rotation_speed) % 360
    return data


def luminance(color):
    return np.dot(color, [0.2126, 0.7152, 0.0722])


def generate_palette(points, num_colors):
    sorted_points = np.array(sorted(points, key=lambda x: (luminance(x), points.index(x))))
    # sorted_points = np.array(points)
    # sorted_points = points
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
            color_palette[i][channel] -= (
                color_palette[i][channel] - target_palette[i][channel]
            ) / steps
    return color_palette


def calculate_zoom(data, current_zoom, zoom_iteration):
    if zoom_iteration < data.zoom_duration:
        t = zoom_iteration / data.zoom_duration
        current_zoom += (data.zoom_factor - current_zoom) * t
        zoom_iteration += 1
    else:
        # if data.zoom_factor > 1:
        #     data.zoom_position_x = -0.527504221
        #     data.zoom_position_y = 0.075911712
        #     data.zoom_duration = 1200
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
    width, height = 960, 540
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Fractal")
    clock = pygame.time.Clock()

    num_colors = 8
    for color_map in data.color_maps:
        cmap = plt.get_cmap(color_map, num_colors)
        colors = cmap(np.linspace(0, 1, num_colors))

        # Convert the RGB values to the desired format
        rgb_colors = [[int(r*255), int(g*255), int(b*255)] for r, g, b, _ in colors]

        # Add black and white to the beginning and end of the palette
        rgb_colors =  [[0,0,0]]+ rgb_colors + [[255,255,255]]
        data.color_palette_list.append(generate_palette(rgb_colors, data.max_iterations))

    # for i in range(20):
    #     nb_color = 15
    #     rng_colors = [[0, 0, 0]] * nb_color
    #     rng_colors[-1] = [205, 205, 205]
    #     for i in range(1, nb_color - 1):
    #         rng_colors[i] = [
    #             random.randint(i, 255),
    #             random.randint(i, 255),
    #             random.randint(i, 255),
    #         ]
    #     data.color_palette_list.append(generate_palette(rng_colors, data.max_iterations))

    data.color_palette = data.color_palette_list[0]

    zoom_iteration = 0
    running = True
    current_zoom = 1.0
    i = 0
    lock = threading.Lock()

    def render_frame():
        nonlocal i, zoom_iteration, current_zoom, data, running, lock, clock
        while running:
            with lock:  # Acquire lock before accessing shared data
                if data.kick > 0:
                    kick = -(data.kick - data.kick_max / 2)**2 / data.kick_max ** 2 + 1
                    color_kick = [
                        tuple(np.clip(c * math.exp(0.5*kick * c / 255), 0, 255) for c in color)
                        for color in color_mapping
                    ]
                    color_mapping = np.array(color_kick[: data.max_iterations], dtype=np.uint8)
                    data.kick -= 1
                else:
                    color_mapping = np.array(data.color_palette[: data.max_iterations], dtype=np.uint8)
                colors = calculate_fractal(data, width, height, fractal_set, color_mapping)
                display_fractal(screen, colors, clock)
                data = edit_var(data, zoom_iteration, i)
                data, current_zoom, zoom_iteration = calculate_zoom(
                    data, current_zoom, zoom_iteration
                )
                i += 1

    with ThreadPoolExecutor() as executor:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_a:
                    data.x_min -= 0.01 * (data.x_max - data.x_min) 
                    data.x_max -= 0.01 * (data.x_max - data.x_min) 
                if event.type == pygame.KEYDOWN and event.key == pygame.K_d:
                    data.x_min += 0.01 * (data.x_max - data.x_min) 
                    data.x_max += 0.01 * (data.x_max - data.x_min) 
                if event.type == pygame.KEYDOWN and event.key == pygame.K_w:
                    data.y_min -= 0.01 * (data.y_max - data.y_min) 
                    data.y_max -= 0.01 * (data.y_max - data.y_min) 
                if event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                    data.y_min += 0.01 * (data.y_max - data.y_min) 
                    data.y_max += 0.01 * (data.y_max - data.y_min) 
                if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    data.rotation_speed += 0.5
                if event.type == pygame.KEYDOWN and event.key == pygame.K_e:
                    data.rotation_speed -= 0.5
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    data.kick = data.kick_max
                    

            executor.submit(render_frame)
            clock.tick(data.fps)

    pygame.quit()


if __name__ == "__main__":
    # data, fractal_set = init_mandelbrot()
    data, fractal_set = init_julia()
    display(data, fractal_set)
