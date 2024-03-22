import numpy as np
import matplotlib.pyplot as plt


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
            color_palette[i][channel] -= (
                color_palette[i][channel] - target_palette[i][channel]
            ) / steps
    return color_palette


def create_palette(color_maps: list, max_iterations: int, num_colors: int):
    palette = []
    for color_map in color_maps:
        cmap = plt.get_cmap(color_map, num_colors)
        colors = cmap(np.linspace(0, 1, num_colors))
        rgb_colors = [[int(r * 255), int(g * 255), int(b * 255)] for r, g, b, _ in colors]
        rgb_colors = [[0, 0, 0]] + rgb_colors + [[255, 255, 255]]
        palette.append(generate_palette(rgb_colors, max_iterations))
    return palette
