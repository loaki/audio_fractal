import pygame
import numpy as np
from numba import jit, prange
import math
import random

class RenderData:
    fps: int

    max_iterations: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    zoom_factor: float
    zoom_sign: int
    zoom_position_x: float
    zoom_position_y: float
    zoom_duration: int

    red_multiplier: float
    green_multiplier: float
    blue_multiplier: float

    constant: complex
    color_palette: list

@jit(nopython=True, parallel=True)
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

@jit(nopython=True, parallel=True)
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
                if zx * zx + zy * zy > 4:
                    break
                iterations[y, x] = i
                if i > 0 and (zx, zy) == (c[y, x].real, c[y, x].imag):
                    iterations[y, x] = max_iterations
                    break
    return iterations


def calculate_fractal(data: RenderData, width, height, fractal_set, screen, clock):
    real, imag = np.meshgrid(
        np.linspace(data.x_min, data.x_max, width), np.linspace(data.y_min, data.y_max, height)
    )
    c = real + 1j * imag
    iterations = fractal_set(c, data.max_iterations, data.constant)

    # Map the number of iterations to colors
    color_mapping = np.zeros((data.max_iterations, 3), dtype=np.uint8)
    for i, color in enumerate(data.color_palette):
        color_mapping[i] = color

    # Use direct mapping for colors
    colors = color_mapping[iterations]

    pygame.surfarray.blit_array(screen, colors.swapaxes(0, 1))

    fps = clock.get_fps()
    fps_font = pygame.font.Font(None, 20)
    fps_text = fps_font.render(str(int(fps)), True, (200, 200, 200))
    screen.blit(fps_text, (10, 10))

    pygame.display.flip()


def smooth_zoom(current_zoom, target_zoom, zoom_iteration, zoom_duration):
    t = zoom_iteration / zoom_duration
    return current_zoom + (target_zoom - current_zoom) * t


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
    data.fps = 30

    # Julia set parameters
    data.max_iterations = 100
    data.x_min = -2.0
    data.x_max = 2.0
    data.y_min = -2.0
    data.y_max = 2.0
    data.zoom_factor = 0.985  # Zoom factor (0 to 1)
    data.zoom_sign = 1
    data.zoom_position_x = -0.527504221
    data.zoom_position_y = 0.075911712  # Set center for Julia set
    data.zoom_duration = 1500  # Number of zoom iterations

    # Define color variables
    data.red_multiplier = 0.9
    data.green_multiplier = 0.05
    data.blue_multiplier = 0.7

    data.cx = -0.8
    data.cy = 0.156

    data.constant = complex(-0.8, 0.156)

    data.color_palette = generate_palette([[0, 0, 0], [255, 255, 255]], data.max_iterations)

    return data, julia


def edit_var(data, i):
    # data.zoom_factor = data.zoom_factor + 0.000007 * data.zoom_sign
    data.cx += (2*math.sin(i%200)-1) * 0.00001
    data.cy -= (2*math.sin(i%200)-1) * 0.00002
    data.constant = complex(data.cx, data.cy)

def luminance(color):
    # Calculate luminance based on RGB values
    r, g, b = color
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def generate_palette(points, num_colors):
    # Sort points based on luminance and index to maintain relative order
    sorted_points = sorted(points, key=lambda x: (luminance(x), points.index(x)))

    num_points = len(sorted_points)
    palette = []

    # Calculate the number of colors between each pair of points
    colors_between_points = num_colors // (num_points - 1)
    remainder = num_colors % (num_points - 1)  # Distribute remaining colors evenly

    # Iterate over each pair of adjacent points
    for i in range(num_points - 1):
        color1 = sorted_points[i]
        color2 = sorted_points[i + 1]

        # Calculate the number of colors for the current pair of points
        colors_for_pair = colors_between_points + (1 if i < remainder else 0)

        # Interpolate between the colors of the current pair of points
        for j in range(colors_for_pair):
            ratio = j / (colors_for_pair - 1)
            color = np.array(color1) * (1 - ratio) + np.array(color2) * ratio
            palette.append([int(c) for c in color])

    return palette

def display(data: RenderData, fractal_set):
    # Pygame initialization
    pygame.init()
    width, height = 600, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Fractal")
    clock = pygame.time.Clock()

    zoom_iteration = 0
    running = True
    current_zoom = 1.0
    i = 0
    while running:
        calculate_fractal(data, width, height, fractal_set, screen, clock)

        i+=1
        # edit_var(data, i)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    data.max_iterations = 500
                    nb_color = 8
                    rng_colors = [[0, 0, 0]] * nb_color
                    for i in range(1, nb_color):
                        rng_colors[i] = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
                    data.color_palette = generate_palette(rng_colors, 500)
                if event.key == pygame.K_ESCAPE:
                    running = False

        if zoom_iteration < data.zoom_duration:
            current_zoom = smooth_zoom(
                current_zoom, data.zoom_factor, zoom_iteration, data.zoom_duration
            )
            zoom_iteration += 1
        else:
            # data.x_min, data.x_max, data.y_min, data.y_max= (
            #     original_x_min,
            #     original_x_max,
            #     original_y_min,
            #     original_y_max,
            # )
            data.zoom_factor = data.zoom_sign * abs(1 - data.zoom_factor) + 1
            data.zoom_sign *= -1
            # current_zoom = 1.0
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

        # Add a delay to control the frame rate
        clock.tick(data.fps)

    pygame.quit()


if __name__ == "__main__":
    # data, fractal_set = init_mandelbrot()
    data, fractal_set = init_julia()
    display(data, fractal_set)
