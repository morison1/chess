from constants import *
from line import Line
import numpy as np


def get_color(color):
    if color:
        return color
    return [255, 0, 0]

def get_random_rgb_color():
    return np.random.randint(256, size=3)

def draw_line(line, image, color=None, width=3):
    color = get_color(color)
    for point in line:
        for i in range(-width // 2 + 1, width // 2 + 1):
            if line.vertical:
                image[point.y, point.x + i] = color
            else:
                image[point.y + i, point.x] = color


def draw_axis(axis, image, vertical, color=None, width=3):
    color = get_color(color)
    h, w, c = image.shape
    if vertical:
        bound = h
    else:
        bound = w
    for i in range(bound):
        for j in range(-width // 2 + 1, width // 2 + 1):
            if vertical:
                image[i, axis + j] = color
            else:
                image[axis + j, i] = color

def draw_colorfull_lines(lines, image):
    for line in lines:
        random_color = get_random_rgb_color()
        draw_line(line, image, random_color)


def draw_board_detection(left, top, square_size, image, color, width=3):
    board_length = square_size * CHESS_BOARD_SIZE

    up_line = Line(axis=top, small=left, large=left + board_length, vertical=False)
    down_line = Line(axis=top + board_length, small=left, large=left + board_length, vertical=False)
    left_line = Line(axis=left, small=top, large=top + board_length, vertical=True)
    right_line = Line(axis=left + board_length, small=top, large=top + board_length, vertical=True)

    lines = [up_line, down_line, left_line, right_line]
    for line in lines:
        draw_line(line, image, color, width)
