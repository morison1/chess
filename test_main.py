from main import max_ones_at_sub_matrix, max_ones_at_sub_matrix_slow

import numpy as np


def test_max_ones_at_sub_matrix():
    for i in range(10):
        matrix = np.random.randint(2, size = 100).reshape(10,10)
        assert max_ones_at_sub_matrix(matrix, k=5) == max_ones_at_sub_matrix_slow(matrix, k=5)

if __name__ == '__main__':
    test_max_ones_at_sub_matrix()