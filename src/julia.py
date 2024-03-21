from numba import jit, prange
import cmath
import numpy as np

from models import RenderData
from colors import transform_palette_iterative, create_palette


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

                # Early escape condition check
                if i > 0 and z == c[y, x]:
                    iterations[y, x] = max_iterations
                    break
                iterations[y, x] = i

    return iterations


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


def init_julia():
    data = RenderData()

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
        "viridis",
        "plasma",
        "inferno",
        "magma",
        "cividis",
        "Dark2",
        "Set1",
        "Set2",
        "Set3",
        "tab10",
        "tab20",
        "tab20b",
        "tab20c",
        "twilight",
        "twilight_shifted",
        "hsv",
        "PiYG",
        "PRGn",
        "BrBG",
        "PuOr",
        "RdGy",
        "RdBu",
        "RdYlBu",
        "RdYlGn",
        "Spectral",
        "coolwarm",
        "bwr",
        "seismic",
        "binary",
        "gist_yarg",
        "gist_gray",
        "gray",
        "bone",
        "pink",
        "spring",
        "summer",
        "autumn",
        "winter",
        "cool",
        "Wistia",
        "hot",
        "afmhot",
        "gist_heat",
        "copper",
    ]
    data.color_palette_list = create_palette(data.color_maps, data.max_iterations, 8)
    data.color_palette = data.color_palette_list[0]

    return data, julia, edit_var
