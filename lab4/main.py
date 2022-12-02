import matplotlib.pyplot as plt
import numpy as np


# 1. Deterministic automaton
def next_row(prev_row: str,
             mappings: dict):
    n = len(prev_row)
    new_row = []
    for middle_idx in range(n):
        right_idx = middle_idx + 1 if middle_idx + 1 < n else 0
        window = prev_row[middle_idx - 1] + prev_row[middle_idx] + prev_row[right_idx]
        next_value = mappings[window]
        new_row.append(next_value)

    return ''.join(new_row)


def algo(first_row: str,
         mappings: dict,
         iterations: int = 100):
    rows = [first_row]
    for _ in range(iterations):
        new_row = next_row(rows[-1], mappings)
        rows.append(new_row)

    mapped_rows = [
        list(map(int, row)) for row in rows
    ]

    image = np.array(mapped_rows)
    return image


def evolution(mappings, n=100, p=0.5, iterations=100):
    ones_num = int(p * n)
    zeros_num = n - ones_num
    xs = [1] * ones_num + [0] * zeros_num
    np.random.shuffle(xs)

    first_row = ''.join(map(str, xs))
    return algo(first_row, mappings, iterations)


if __name__ == '__main__':
    first_row = '111110101100011010001000'
    mappings = {
        '111': '1',
        '110': '0',
        '101': '1',
        '100': '1',
        '011': '1',
        '010': '0',
        '001': '0',
        '000': '0'
    }

    image = evolution(mappings, n=100, p=0.7, iterations=100)
    plt.imshow(image, cmap='binary')
    plt.show()

# Input:
# 111 110 101 100 011 010 001 000
#  1   0   1   1   1   0   0   0
#
#
# Generating first row:
# 11110...
#
# first window: 011, 111, 110
#
