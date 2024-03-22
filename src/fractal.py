import pygame
import numpy as np
import math
import configparser
from multiprocessing.pool import ThreadPool
import multiprocessing
import threading
import queue

from julia import init_julia
from models import RenderData
from record import record_sound

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


def calculate_zoom(data):
    data.x_min = data.zoom_position_x - ((data.x_max - data.x_min) / data.zoom_factor / 2)
    data.x_max = data.zoom_position_x + ((data.x_max - data.x_min) / data.zoom_factor / 2)
    data.y_min = data.zoom_position_y - ((data.y_max - data.y_min) / data.zoom_factor / 2)
    data.y_max = data.zoom_position_y + ((data.y_max - data.y_min) / data.zoom_factor / 2)
    data.zoom_iteration += 1
    return data

def render_frame(data, width, height, fractal_func):
    if data.kick > 0:
        kick = -((data.kick - data.kick_max / 2) ** 2) / data.kick_max**2 + 1
        color_kick = [
            tuple(np.clip(c * math.exp(0.7 * kick * c / 255), 0, 255) for c in color)
            for color in data.color_palette
        ]
        color_mapping = np.array(color_kick[: data.max_iterations], dtype=np.uint8)
        data.kick -= 1
    else:
        color_mapping = np.array(
            data.color_palette[: data.max_iterations], dtype=np.uint8
        )

    return calculate_fractal(data.x_min, data.x_max, data.y_min, data.y_max, data.max_iterations, data.constant, data.rotation_angle_degrees, width, height, fractal_func, color_mapping)


def record_thread(queue, speaker, fps):
    for rec in record_sound(speaker_name=speaker, record_sec=1/fps):
        queue.put(rec)


def display(data: RenderData, config, fractal_func, edit_func):
    pygame.init()
    width, height = 960, 540
    # screen = pygame.display.set_mode((width, height))
    screen = pygame.display.set_mode((width, height), pygame.FULLSCREEN|pygame.SCALED)
    pygame.display.set_caption("Fractal")

    data_queue = queue.LifoQueue()
    thread = threading.Thread(target=record_thread, args=[data_queue, config["DEFAULT"]["speaker"], data.fps])
    thread.daemon = True 
    thread.start()
    clock = pygame.time.Clock()
    running = True
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
                data.rotation_speed = 300
            if event.type == pygame.KEYDOWN and event.key == pygame.K_e:
                data.rotation_speed = 0
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                data.kick = data.kick_max

        colors = pool.apply(render_frame, (data, width, height, fractal_func))
        display_fractal(screen, colors, clock)
        try:
            record = data_queue.get_nowait()
        except:
            record = None
        data = edit_func(data, record)
        data = calculate_zoom(data)
        clock.tick(data.fps)

    pygame.quit()


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("config.ini")
    data, fractal_func, edit_func = init_julia()
    display(data, config, fractal_func, edit_func)
