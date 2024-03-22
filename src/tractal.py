import pygame
import numpy as np
import math
from multiprocessing.pool import ThreadPool
import multiprocessing

from julia import init_julia
from models import RenderData


def calculate_fractal(x_min, x_max, y_min, y_max, max_iterations, constant, rotation_angle_degrees, width, height, fractal_set, color_mapping):
    real, imag = np.meshgrid(
        np.linspace(x_min, x_max, width),
        np.linspace(y_min, y_max, height),
    )
    c = real + 1j * imag
    iterations = fractal_set(c, max_iterations, constant, rotation_angle_degrees)

    colors = color_mapping[iterations]

    return colors


def display_fractal(screen, colors, clock):
    pygame.surfarray.blit_array(screen, colors.swapaxes(0, 1))

    fps = clock.get_fps()
    fps_font = pygame.font.Font(None, 20)
    fps_text = fps_font.render(str(int(fps)), True, (200, 200, 200))
    screen.blit(fps_text, (10, 10))

    pygame.display.flip()


def calculate_zoom(data, current_zoom, zoom_iteration):
    if zoom_iteration < data.zoom_duration:
        t = zoom_iteration / data.zoom_duration
        current_zoom += (data.zoom_factor - current_zoom) * t
        zoom_iteration += 1
    else:
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


def display(data: RenderData, fractal_func, edit_func):
    # Pygame initialization
    pygame.init()
    width, height = 800, 800
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Fractal")
    clock = pygame.time.Clock()

    zoom_iteration = 0
    running = True
    current_zoom = 1.0
    i = 0
    # lock = threading.Lock()

    def render_frame(running, data):
        # nonlocal i, zoom_iteration, current_zoom, data, running, lock, clock
        while running:
            # with lock:
            if data.kick > 0:
                kick = -((data.kick - data.kick_max / 2) ** 2) / data.kick_max**2 + 1
                color_kick = [
                    tuple(np.clip(c * math.exp(0.5 * kick * c / 255), 0, 255) for c in color)
                    for color in data.color_palette
                ]
                color_mapping = np.array(color_kick[: data.max_iterations], dtype=np.uint8)
                data.kick -= 1
            else:
                color_mapping = np.array(
                    data.color_palette[: data.max_iterations], dtype=np.uint8
                )

            return calculate_fractal(data.x_min, data.x_max, data.y_min, data.y_max, data.max_iterations, data.constant, data.rotation_angle_degrees, width, height, fractal_func, color_mapping)
            
    pool = ThreadPool(multiprocessing.cpu_count())

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

        colors = pool.apply(render_frame, (running, data))
        display_fractal(screen, colors, clock)
        data = edit_func(data, zoom_iteration, i)
        data, current_zoom, zoom_iteration = calculate_zoom(
            data, current_zoom, zoom_iteration
        )
        i += 1
        clock.tick(data.fps)

pygame.quit()


if __name__ == "__main__":
    data, fractal_func, edit_func = init_julia()
    display(data, fractal_func, edit_func)
