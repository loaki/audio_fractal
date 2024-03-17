import pygame
import numpy as np
from numba import jit, prange
import threading


class RenderData:
    fps: int
    frame_duration: int

    max_iterations: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    zoom_factor: float
    zoom_position_x: float
    zoom_position_y: float
    zoom_duration: int

    red_multiplier: float
    green_multiplier: float
    blue_multiplier: float


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


def calculate_fractal(data: RenderData, height, width):
    real, imag = np.meshgrid(
        np.linspace(data.x_min, data.x_max, width), np.linspace(data.y_min, data.y_max, height)
    )
    c = real + 1j * imag
    iterations = mandelbrot(c, data.max_iterations)

    # Map the number of iterations to colors
    colors = np.zeros((height, width, 3), dtype=np.uint8)
    colors[:, :, 0] = np.clip(
        255 * data.red_multiplier * np.sin(iterations / data.max_iterations * np.pi) ** 2, 0, 255
    )  # Red component
    colors[:, :, 1] = np.clip(
        255 * data.green_multiplier * np.cos(iterations / data.max_iterations * np.pi) ** 2, 0, 255
    )  # Green component
    colors[:, :, 2] = np.clip(
        255
        * data.blue_multiplier
        * np.sin(iterations / data.max_iterations * np.pi)
        * np.cos(iterations / data.max_iterations * np.pi),
        0,
        255,
    )  # Blue component

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


def render_fractal(screen, width, height, data: RenderData):
    while True:
        colors = calculate_fractal(data, width, height)
        draw_fractal(screen, colors)


def init_render():
    data = RenderData()

    # Set up the clock for controlling the frame rate
    data.fps = 60
    data.frame_duration = 1000 // data.fps

    # Mandelbrot parameters
    data.max_iterations = 200
    data.x_min = -2.0
    data.x_max = 1.0
    data.y_min = -1.5
    data.y_max = 1.5
    data.zoom_factor = 0.97  # Zoom factor (0 to 1)
    data.zoom_position_x = -0.7462
    data.zoom_position_y = -0.1495  # Seahorse Valley zoom position
    data.zoom_duration = 900  # Number of zoom iterations

    # Define color variables
    data.red_multiplier = 0.7
    data.green_multiplier = 0.1
    data.blue_multiplier = 0.9

    return data


def display(data: RenderData):
    # Pygame initialization
    pygame.init()
    width, height = 600, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Mandelbrot Fractal")

    zoom_iteration = 0
    running = True
    current_zoom = 1.0
    original_x_min, original_x_max, original_y_min, original_y_max = (
        data.x_min,
        data.x_max,
        data.y_min,
        data.y_max,
    )

    render_thread = threading.Thread(target=render_fractal, args=[screen, width, height, data])
    render_thread.daemon = True
    render_thread.start()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:  # Press 'f' to toggle fullscreen
                    running = False

        if zoom_iteration < data.zoom_duration:
            current_zoom = smooth_zoom(
                current_zoom, data.zoom_factor, zoom_iteration, data.zoom_duration
            )
            zoom_iteration += 1
        else:
            data.x_min, data.x_max, data.y_min, data.y_max = (
                original_x_min,
                original_x_max,
                original_y_min,
                original_y_max,
            )
            current_zoom = 1.0
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
        pygame.time.wait(data.frame_duration)

    pygame.quit()


if __name__ == "__main__":
    data = init_render()
    display(data)
