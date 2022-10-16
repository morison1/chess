import cv2
import numpy as np
import matplotlib.pyplot as plt
from constants import *
from draw import *
from line import Line
import itertools
from collections import Counter


# from tqdm import tqdm

def edge_detection_per_array_slow(array):
    increase = []
    decrease = []
    for i in range(EDGE_DETECTION_PIXEL_SAMPLE, len(array) - EDGE_DETECTION_PIXEL_JUMP - EDGE_DETECTION_PIXEL_SAMPLE):
        left_min = i - EDGE_DETECTION_PIXEL_SAMPLE
        left_max = i
        right_min = i + EDGE_DETECTION_PIXEL_JUMP
        right_max = i + EDGE_DETECTION_PIXEL_JUMP + EDGE_DETECTION_PIXEL_SAMPLE

        left = np.mean(array[left_min: left_max])
        right = np.mean(array[right_min: right_max])

        if abs(right - left) > PIXEL_DIFFERENCE:
            if right > left:
                increase.append(i)
            else:
                decrease.append(i)
    return increase, decrease


def edge_detection_per_array(array):
    increase = []
    decrease = []
    for i in range(len(array) - EDGE_DETECTION_PIXEL_JUMP):
        left = array[i]
        right = array[i + EDGE_DETECTION_PIXEL_JUMP]

        if abs(int(right) - int(left)) > PIXEL_DIFFERENCE:
            if right > left:
                increase.append(i)
            else:
                decrease.append(i)
    return increase, decrease


def edge_detection(image, vertical: bool):
    increase = []
    decrease = []
    if not vertical:
        image = image.transpose()
    for row in image:
        vertical_increase_per_row, vertical_decrease_per_row = edge_detection_per_array(row)
        increase.append(vertical_increase_per_row)
        decrease.append(vertical_decrease_per_row)

    return increase, decrease






def sort_lines_by_length(lines):
    return sorted(lines, key=lambda line: len(line), reverse=True)


def get_lines(image, vertical: bool = True):
    increase, decrease = edge_detection(image, vertical)
    axis_to_increase_map = map_axis_to_edges(increase)
    axis_to_decrease_map = map_axis_to_edges(decrease)
    increase_lines = _get_lines(axis_to_increase_map, vertical)
    decrease_lines = _get_lines(axis_to_decrease_map, vertical)
    lines = increase_lines + decrease_lines
    return lines


def map_axis_to_edges(diffs):
    result = {}
    for i, row in enumerate(diffs):
        for j in row:
            current = result.get(j, [])
            current.append(i)
            result[j] = current
    return result


def _get_lines(axis_to_edge_map, vertical):
    all_lines = []
    for axis, edges in axis_to_edge_map.items():
        lines = get_lines_per_array(axis=axis, array=edges, vertical=vertical)
        all_lines.extend(lines)
    return all_lines


def get_lines_per_array(axis, array, vertical):
    continuous_segments = np.split(array, np.where(np.diff(array) != 1)[0] + 1)
    lines = [Line(axis=axis,
                  small=segment[0],
                  large=segment[-1],
                  vertical=vertical) for segment in continuous_segments]
    return lines


def filter_lines_by_min_length(lines):
    filtered_lines = []
    for line in lines:
        if len(line) >= MIN_SQUARE_LENGTH:
            filtered_lines.append(line)
    return filtered_lines


def get_map_axis_to_lines(lines):
    mapping = {}
    for line in lines:
        axis = line.axis
        current_lines = mapping.get(axis, [])
        current_lines.append(line)
        mapping[axis] = current_lines
    return mapping


def filter_similar_lines(lines):
    """
    Assuming lines is either vertical or horizontal but not both
    """
    map_axis_to_line = get_map_axis_to_lines(lines)
    filtered_lines = []
    for axis in map_axis_to_line:
        previous_lines = []
        for i in (1, SIMILAR_LINES + 1):
            previous_lines.extend(map_axis_to_line.get(axis - i, []))
        for line in map_axis_to_line[axis]:
            pass_filter = True
            for previous_line in previous_lines:
                if are_similar_lines(previous_line, line):
                    pass_filter = False
                    break
            if pass_filter:
                filtered_lines.append(line)
    return filtered_lines


def are_similar_lines(first_line, second_line):
    if first_line.vertical != second_line.vertical:
        return False
    if abs(first_line.axis - second_line.axis) > SIMILAR_LINES:
        return False
    if abs(first_line.small - second_line.small) > SIMILAR_LINES:
        return False
    if abs(first_line.large - second_line.large) > SIMILAR_LINES:
        return False
    return True


def filter_lines(lines):
    lines = filter_lines_by_min_length(lines)
    lines = filter_similar_lines(lines)
    return lines


# def sort_vertical_line_old_solution(x_to_vertical_lines_map):
#     sorted_xs = sort_x_by_line_score(x_to_vertical_lines_map)
#     sorted_xs_filtered = filter_x_values(sorted_xs)
#
#     sorted_xs_filtered = sorted_xs_filtered[:NUM_OF_AXIS]
#     sorted_xs_filtered = sorted(sorted_xs_filtered)
#     return sorted_xs_filtered, x_to_vertical_lines_map



def resize_image(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return image


# def count_pairs_with_given_diff(arr, target_diff):
#     n = len(arr)
#     count = 0
#     for i in range(n):
#         for j in range(i + 1, n):
#             diff = arr[j] - arr[i]
#             if 1 + DIFFERENCE_EPSILON > diff / target_diff > 1 - DIFFERENCE_EPSILON:
#                 count += 1
#     return count


# def get_most_likely_diff(vertical_line_candidates, horizontal_line_candidates):
#     vertical_diff = set(np.diff(vertical_line_candidates))
#     horizontal_diff = set(np.diff(horizontal_line_candidates))
#
#     all_diff = list(vertical_diff.union(horizontal_diff))
#     all_counts = []
#     for target_diff in all_diff:
#         counts = count_pairs_with_given_diff(all_diff, target_diff)
#         all_counts.append(counts)
#
#     most_likely_diff_index = np.argmax(all_counts)
#     most_likely_diff = all_diff[most_likely_diff_index]
#     return most_likely_diff


# def get_points_with_given_diff(points, target_diff):
#     n = len(points)
#     all_points = set()
#     for i in range(n):
#         for j in range(i + 1, n):
#             diff = points[j] - points[i]
#             if 1 + DIFFERENCE_EPSILON > diff / target_diff > 1 - DIFFERENCE_EPSILON:
#                 all_points.add(points[i])
#                 all_points.add(points[j])
#     all_points = sorted(all_points)
#     return all_points


def transpose_image(image):
    if len(image.shape) == 3:
        return np.transpose(image, [1, 0, 2])
    else:
        return np.transpose(image)


def get_square_size(lines):
    lines = sort_lines_by_length(lines)
    length_counter = Counter()
    for line in lines:
        length_counter[len(line)] += 1
    length_counter_pairs = length_counter.most_common()
    valid_lengths = [length for length, count in length_counter_pairs if count >= MIN_NUMBER_OF_SQUARE_LINES]
    square_size = max(valid_lengths)
    return square_size


def filter_lines_by_square_size(lines, square_size):
    filtered_lines = []
    for line in lines:
        # if abs(len(line) - square_size) <= SQUARE_SIZE_EPSILON:
        if len(line) >= square_size - SQUARE_SIZE_EPSILON:
            filtered_lines.append(line)
    return filtered_lines


def get_vertical_horizontal_modulo(lines, square_size):
    vertical_modulos = []
    horizontal_modulos = []
    for line in lines:
        if line.vertical:
            vertical_modulos.append(line.axis % square_size)
        else:
            horizontal_modulos.append(line.axis % square_size)

    vertical_modulo = arg_max_with_width(vertical_modulos, width=ARG_MAX_WIDTH, mod=square_size)
    horizontal_modulo = arg_max_with_width(horizontal_modulos, width=ARG_MAX_WIDTH, mod=square_size)
    return vertical_modulo, horizontal_modulo


def filter_lines_by_modulo(lines, vertical_modulo, horizontal_modulo, square_size, width):
    filtered_lines = []
    for line in lines:
        if line.vertical:
            x_grid_coordinate = map_x_to_grid_coordinate(line.axis, vertical_modulo, square_size)
            new_axis = map_grid_coordinate_to_x(x_grid_coordinate, vertical_modulo, square_size)
        else:
            y_grid_coordinate = map_y_to_grid_coordinate(line.axis, horizontal_modulo, square_size)
            new_axis = map_grid_coordinate_to_y(y_grid_coordinate, horizontal_modulo, square_size)

        diff = abs(line.axis - new_axis)
        if diff <= width:
            filtered_lines.append(line)
    return filtered_lines


def map_x_to_grid_coordinate(x, vertical_modulo, square_size):
    return round((x - vertical_modulo) / square_size)


def map_y_to_grid_coordinate(y, horizontal_modulo, square_size):
    return round((y - horizontal_modulo) / square_size)


def map_grid_coordinate_to_x(x_grid_coordinate, vertical_modulo, square_size):
    return x_grid_coordinate * square_size + vertical_modulo


def map_grid_coordinate_to_y(y_grid_coordinate, horizontal_modulo, square_size):
    return y_grid_coordinate * square_size + horizontal_modulo


def arg_max_with_width(array, width, mod):
    """
    array - list of integers
    width - int
    return x in array s.t. [x - width <= y <= x + width (mod) for y in array] is maximal
    """
    array_counts = Counter(array)
    max_value = 0
    max_arg = -1
    for x in array_counts:
        x_val = 0
        for y in range(x - width, x + width + 1):
            y = y % mod
            x_val += array_counts[y]
        if x_val > max_value:
            max_value = x_val
            max_arg = x
    return max_arg


def max_ones_at_sub_matrix_slow(matrix, k=CHESS_BOARD_SIZE):
    """
    ### Naive solution. The running time can be improved of course from (m*n)^2 to m*n ###
    Given a matrix (m x n) with binary values and a positive integer k, 
    returns the k x k sub matrix with the maximal number of ones. 
    More precisely returns the index of the upper left corner of the optimal sub matrix.
    """
    m, n = matrix.shape
    if m < k or n < k:
        return None
    max_number_of_ones = 0
    i_sol, j_sol = 0, 0
    for i in range(m - k + 1):
        for j in range(n - k + 1):
            sub_matrix = matrix[i:i + k, j:j + k]
            sub_value = np.sum(sub_matrix)
            if sub_value > max_number_of_ones:
                max_number_of_ones = sub_value
                i_sol, j_sol = i, j
    return i_sol, j_sol


def max_ones_at_sub_matrix(matrix, k=CHESS_BOARD_SIZE):
    """
    ### Better solution than the naive one. The running time is m*n*k and can be improved to m*n ###
    Given a matrix (m x n) with binary values and a positive integer k,
    returns the k x k sub matrix with the maximal number of ones.
    More precisely returns the index of the upper left corner of the optimal sub matrix.
    """
    m, n = matrix.shape
    if m < k or n < k:
        return None
    max_sol = 0
    i_sol, j_sol = 0, 0

    left_most_value = np.sum(matrix[:k, :k])
    current_value = 0
    for i in range(m - k + 1):
        if i > 0:
            left_most_value = left_most_value - np.sum(matrix[i - 1, :k]) + np.sum(matrix[i + k - 1, :k])
        for j in range(n - k + 1):
            if j == 0:
                current_value = left_most_value
            else:
                current_value = current_value - np.sum(matrix[i:i + k, j - 1]) + np.sum(matrix[i:i + k, j + k - 1])

            if current_value > max_sol:
                max_sol = current_value
                i_sol, j_sol = i, j
    return i_sol, j_sol


def get_special_points(direction_modulo, square_size, max_length):
    special_points_val = list(enumerate(range(direction_modulo + square_size // 2, max_length, square_size)))
    special_points_map = {}
    for i, p in special_points_val:
        special_points_map[p] = i
    return special_points_map


def get_matrix_representation(lines, square_size, vertical_modulo, horizontal_modulo, image_shape):
    height, width, c = image_shape
    """
    In general we can say 0 <= x < width
    LOWER BOUND: 
    But the only x's got filtered in satisfy that they are close to the vertical_modulo (up to square_size),
    therefore x >= vertical_modulo - width -->  round((x - vertical_modulo) / square_size) >= 
    round (-width / square_size) = 0
    In other words it might happen that round((x - vertical_modulo) / square_size) would be equal to -1 but we don't
    care for this option (the same for y)
    ---> LOWER BOUND = 0
    UPPER BOUND:
    round((x - vertical_modulo) / square_size) <= round((width -1 - vertical_modulo) / square_size) <=
    round(width / square_size)
    The upper row of the chess board  == the most upper line of the chess board = 0 
    The left column of the chess board == the most left line of the chess board = 0
    In general we think of the i horizontal line together with the row beneath it,
    and we think of the j vertical line together with the column to the right of it.
    """
    h, w = round(height / square_size), round(width / square_size)
    matrix = np.zeros((h, w, 4))  # the 4 values are for up, right, down, left

    special_x_points_map = get_special_points(direction_modulo=vertical_modulo, square_size=square_size,
                                              max_length=width)
    special_y_points_map = get_special_points(direction_modulo=horizontal_modulo, square_size=square_size,
                                              max_length=height)

    for line in lines:
        if line.vertical:
            x = line.axis
            x_grid_coordinate = map_x_to_grid_coordinate(x, vertical_modulo, square_size)
            for point in line:
                y = point.y
                if y in special_y_points_map:
                    y_grid_coordinate = special_y_points_map[y]
                    try:
                        matrix[y_grid_coordinate, x_grid_coordinate][3] = 1
                    except:
                        pass
                    try:
                        matrix[y_grid_coordinate, x_grid_coordinate - 1][1] = 1
                    except:
                        pass

        else:
            y = line.axis
            y_grid_coordinate = map_y_to_grid_coordinate(y, horizontal_modulo, square_size)
            for point in line:
                x = point.x
                if x in special_x_points_map:
                    x_grid_coordinate = special_x_points_map[x]
                    try:
                        matrix[y_grid_coordinate, x_grid_coordinate][0] = 1
                    except:
                        pass
                    try:
                        matrix[y_grid_coordinate - 1, x_grid_coordinate][2] = 1
                    except:
                        pass
    matrix = np.sum(matrix, axis=2)
    return matrix


if __name__ == '__main__':
    image_path = 'train_set/hansen.png'
    # image_path = '/Users/moran/Desktop/Screenshot 2022-10-15 at 12.30.36.png'
    # image_path = '/Users/moran/Desktop/Screenshot 2022-10-15 at 12.39.00.png'
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # original_image = resize_image(original_image, 30)
    original_image_copy = original_image.copy()
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    # gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    blank_image = np.zeros_like(gray_image)  # H,W,C
    vertical_lines = get_lines(gray_image, vertical=True)
    vertical_lines = filter_lines(vertical_lines)

    horizontal_lines = get_lines(gray_image, vertical=False)
    horizontal_lines = filter_lines(horizontal_lines)

    lines = horizontal_lines + vertical_lines

    draw_colorfull_lines(lines, original_image_copy)
    plt.imshow(original_image_copy)
    plt.show()

    square_size = get_square_size(lines)
    lines = filter_lines_by_square_size(lines, square_size)


    vertical_modulo, horizontal_modulo = get_vertical_horizontal_modulo(lines, square_size)

    lines = filter_lines_by_modulo(lines, vertical_modulo, horizontal_modulo, square_size, ARG_MAX_WIDTH)
    # draw_colorfull_lines(lines, original_image)
    # plt.imshow(original_image)
    # plt.show()

    matrix = get_matrix_representation(lines, square_size, vertical_modulo, horizontal_modulo, original_image.shape)
    y_grid_coordinate, x_grid_coordinate = max_ones_at_sub_matrix(matrix)
    x = map_grid_coordinate_to_x(x_grid_coordinate, vertical_modulo, square_size)
    y = map_grid_coordinate_to_y(y_grid_coordinate, horizontal_modulo, square_size)

    draw_board_detection(x, y, square_size, original_image, color=[255,0,0], width=5)

    # color = [255,0,0]

    # for x in xs:
    #     draw_x(x, original_image, color)
    # for y in ys:
    #     draw_y(y, original_image, color)
    #
    plt.imshow(original_image)
    plt.show()


# right now 530 lines
