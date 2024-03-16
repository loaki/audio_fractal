import pygame
import numpy as np
from numba import jit, prange
import threading

# Pygame initialization
pygame.init()
width, height = 600, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Mandelbrot Fractal")

# Set up the clock for controlling the frame rate
fps = 100
frame_duration = 1000 // fps
clock = pygame.time.Clock()

# Mandelbrot parameters
max_iterations = 200
x_min, x_max = -2.0, 1.0
y_min, y_max = -1.5, 1.5
zoom_factor = 0.99  # Zoom factor (0 to 1)
zoom_position_x, zoom_position_y = -0.7462, -0.1495  # Seahorse Valley zoom position
zoom_duration = 5000  # Number of zoom iterations

# Define color variables
red_multiplier = 0.7
green_multiplier = 0.0
blue_multiplier = 0.9


@jit(nopython=True, parallel=True)
def mandelbrot(c, max_iterations):
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

def draw_fractal(screen, colors):
    pygame.surfarray.blit_array(screen, colors.swapaxes(0, 1))
    pygame.display.flip()

def calculate_fractal(x_min, x_max, y_min, y_max, max_iterations):
    real, imag = np.meshgrid(np.linspace(x_min, x_max, width), np.linspace(y_min, y_max, height))
    c = real + 1j * imag
    iterations = mandelbrot(c, max_iterations)

    # Map the number of iterations to colors
    colors = np.zeros((height, width, 3), dtype=np.uint8)
    colors[:, :, 0] = np.clip(255 * red_multiplier * np.sin(iterations / max_iterations * np.pi) ** 2, 0, 255)  # Red component
    colors[:, :, 1] = np.clip(255 * green_multiplier * np.cos(iterations / max_iterations * np.pi) ** 2, 0, 255)  # Green component
    colors[:, :, 2] = np.clip(255 * blue_multiplier * np.sin(iterations / max_iterations * np.pi) * np.cos(iterations / max_iterations * np.pi), 0, 255)  # Blue component

    return colors

def calculate_zoom(x_min, x_max, y_min, y_max, zoom_factor):
    # Calculate the new center
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2

    # Calculate the new range
    range_x = (x_max - x_min) * zoom_factor
    range_y = (y_max - y_min) * zoom_factor

    # Update the boundaries with the new zoom
    x_min = center_x - range_x / 2
    x_max = center_x + range_x / 2
    y_min = center_y - range_y / 2
    y_max = center_y + range_y / 2

    return x_min, x_max, y_min, y_max

def smooth_zoom(current_zoom, target_zoom, zoom_iteration, zoom_duration):
    t = zoom_iteration / zoom_duration
    return current_zoom + (target_zoom - current_zoom) * t

def render_thread():
    global x_min, x_max, y_min, y_max
    while True:
        colors = calculate_fractal(x_min, x_max, y_min, y_max, max_iterations)
        draw_fractal(screen, colors)

render_thread = threading.Thread(target=render_thread)
render_thread.daemon = True
render_thread.start()

zoom_iteration = 0
running = True
current_zoom = 1.0
target_zoom = 1.0
original_x_min, original_x_max, original_y_min, original_y_max = x_min, x_max, y_min, y_max

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE: # Press 'f' to toggle fullscreen
                running = False

    # if zoom_iteration % 50 == 0:
    #     green_multiplier = 0.1
    #     red_multiplier = 2
    #     blue_multiplier = 3
    # else:
    #     red_multiplier = 0.7
    #     green_multiplier = 0.0
    #     blue_multiplier = 0.9

    if zoom_iteration < zoom_duration:
        current_zoom = smooth_zoom(current_zoom, zoom_factor, zoom_iteration, zoom_duration)
        zoom_iteration += 1
    else:
        x_min, x_max, y_min, y_max = original_x_min, original_x_max, original_y_min, original_y_max
        current_zoom = 1.0
        zoom_iteration = 0

    # Update the boundaries with the new zoom
    if current_zoom != 1.0:
        # Calculate the new center based on the zoom position
        zoom_x = (zoom_position_x - x_min) / (x_max - x_min)
        zoom_y = (zoom_position_y - y_min) / (y_max - y_min)

        # Calculate the new range based on the zoom factor
        range_x = (x_max - x_min) * current_zoom
        range_y = (y_max - y_min) * current_zoom

        # Update the boundaries with the new zoom
        x_min = zoom_position_x - range_x * zoom_x
        x_max = zoom_position_x + range_x * (1 - zoom_x)
        y_min = zoom_position_y - range_y * zoom_y
        y_max = zoom_position_y + range_y * (1 - zoom_y)

    # Add a delay to control the frame rate
    pygame.time.wait(frame_duration)

pygame.quit()
