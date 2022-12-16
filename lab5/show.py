import os

import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint


def show_M(name: str, M):
    pprint(name)
    pprint(M)
    plt.title(name)
    plt.imshow(M, cmap='binary')
    plt.show()
    pprint('-' * 82)


def show_S(name: str, S):
    pprint(name)
    pprint(M)
    plt.title(name)

    xs = np.arange(0, len(S))
    ys = S

    plt.plot(xs, ys)
    plt.show()
    pprint('-' * 82)


if __name__ == '__main__':
    dir = 'results'
    for filename in os.listdir('results'):
        path = os.path.join(dir, filename)
        if os.path.isfile(path):
            if 'stats' in filename:
                S = np.loadtxt(path, delimiter=' ', dtype=np.float)
                show_S(filename, S)
            else:
                M = np.loadtxt(path, delimiter=',', dtype=np.int32)
                show_M(filename, M)
