from __future__ import annotations

import random
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt

SIZE = 2048


def main():
    img = generate_L_shaped_image()
    show_img(img)

    # first part
    ifs = IFS(first_params_list())
    ifs = IFS(second_params_list())
    ifs = IFS(snakes())
    ifs = IFS(triangle())
    ifs = IFS(chaos())

    # second part
    ifs = IFS(first_params_list(), mode='roulette')
    ifs = IFS(second_params_list(), mode='roulette')
    ifs = IFS(snakes(), mode='roulette')
    ifs = IFS(triangle(), mode='roulette')
    ifs = IFS(chaos(), mode='roulette')

    # third part
    ifs = IFS(first_params_list(), mode='roulette', p=[0.8, 0.1, 0.1])
    ifs = IFS(first_params_list(), mode='roulette', p=[0.2, 0.3, 0.5])
    ifs = IFS(second_params_list(), mode='roulette', p=[0.3, 0.2, 0.3, 0.2])
    ifs = IFS(snakes(), mode='roulette', p=[0.8, 0.1, 0.1])
    ifs = IFS(snakes(), mode='roulette', p=[0.2, 0.2, 0.6])
    ifs = IFS(triangle(), mode='roulette', p=[0.2, 0.8])
    ifs = IFS(chaos(), mode='roulette', p=[0.05, 0.05, 0.3, 0.1, 0.1, 0.1, 0.3])

    for i in range(8):
        img = ifs.iteration(img)
        show_img(img)


def first_params_list() -> List[IFSParams]:
    ifs_0_params = IFSParams(a=0.0, b=-0.5, c=0.5, d=0.0, e=0.5, f=0.0)

    ifs_1_params = IFSParams(a=0.0, b=0.5, c=-0.5, d=0.0, e=0.5, f=0.5)

    ifs_2_params = IFSParams(a=0.5, b=0.0, c=0.0, d=0.5, e=0.25, f=0.5)

    return [ifs_0_params, ifs_1_params, ifs_2_params]


def second_params_list() -> List[IFSParams]:
    # 0.849 0.037 -0.037 0.849 0.075 0.1830
    ifs_0_params = IFSParams(a=0.849, b=0.037, c=-0.037, d=0.849, e=0.075, f=0.1830)

    # 0.197 -0.226 0.226 0.197 0.400 0.049
    ifs_1_params = IFSParams(a=0.197, b=-0.226, c=0.226, d=0.197, e=0.400, f=0.049)

    # -0.150 0.283 0.260 0.237 0.575 -0.084
    ifs_2_params = IFSParams(a=-0.150, b=0.283, c=0.260, d=0.237, e=0.575, f=-0.084)

    # 0.000 0.000 0.000 0.160 0.500 0.000
    ifs_3_params = IFSParams(a=0.000, b=0.000, c=0.000, d=0.160, e=0.500, f=0.000)

    return [
        ifs_0_params,
        ifs_1_params,
        ifs_2_params,
        ifs_3_params,
    ]


def snakes() -> List[IFSParams]:
    ifs_0_params = IFSParams(0.0, -0.5, 0.85, 0.0, -1.732366, 3.366182)

    ifs_1_params = IFSParams(0.3, 0.0, 0.0, 0.3, -0.027891, 5.014877)

    ifs_2_params = IFSParams(0.0, 0.5, -0.85, 0.0, 1.620804, 3.310401)

    return [
        ifs_0_params,
        ifs_1_params,
        ifs_2_params,
    ]


def triangle() -> List[IFSParams]:
    ifs_0_params = IFSParams(0.5, 0.5, -0.5, 0.5, 0.0, 0.0)

    ifs_1_params = IFSParams(0.5, -0.5, 0.5, 0.5, 0.5, -0.5)

    return [
        ifs_0_params,
        ifs_1_params,
    ]


def chaos() -> List[IFSParams]:
    ifs_0_params = IFSParams(0.33, .0, 0, 0.5, .333, .0)

    ifs_1_params = IFSParams(-0.33, .0, 0, -0.5, .666, .0)

    ifs_2_params = IFSParams(0.33, .0, 0, 0.5, .666, .5)

    ifs_3_params = IFSParams(0.75, 0.00, 0.00, 0.30, -0.20, 0.00)

    ifs_4_params = IFSParams(0.75, 0.00, 0.00, 0.30, 0.20, 0.00)

    ifs_5_params = IFSParams(0.50, 0.00, 0.00, 0.80, 0.00, 0.20)

    ifs_6_params = IFSParams(a=0.5, b=0.0, c=0.0, d=0.5, e=0.0, f=0.0)

    return [
        ifs_0_params,
        ifs_1_params,
        ifs_2_params,
        ifs_3_params,
        ifs_4_params,
        ifs_5_params,
        ifs_6_params,
    ]


class IFS:
    def __init__(self, params_list: List[IFSParams], mode=None, p=None):
        self.params_list = params_list
        self.mode = "default" if mode is None else "roulette"
        self.p = p

    def iteration(self, img):

        def iter_default():
            new_img = np.ones((SIZE, SIZE))
            for y in range(SIZE):
                for x in range(SIZE):
                    if img[y, x] == 0:
                        for i in range(len(self.params_list)):
                            new_x, new_y = self.compute_new_position((x, y), i)
                            new_img[new_y, new_x] = 0

            return new_img

        def iter_roulette():
            def pick_w_random():
                return lambda: random.randint(0, len(self.params_list) - 1)

            def pick_w_with_p(p):
                population = list(range(0, len(self.params_list)))
                return lambda: random.choices(population=population, weights=p)[0]

            pick_w_fun = pick_w_random() if self.p is None else pick_w_with_p(self.p)

            new_img = np.ones((SIZE, SIZE))
            for y in range(SIZE):
                for x in range(SIZE):
                    if img[y, x] == 0:
                        w_idx = pick_w_fun()
                        new_x, new_y = self.compute_new_position((x, y), w_idx)
                        new_img[new_y, new_x] = 0

            return new_img

        if self.mode == "default":
            return iter_default()
        elif self.mode == "roulette":
            return iter_roulette()
        else:
            raise Exception(f'Unknown mode {self.mode}')

    def compute_new_position(self, point: Tuple[int, int], params_idx: int) -> Tuple[int, int]:
        x, y = point

        params = self.params_list[params_idx]
        new_x = params.a * x + params.b * y + params.e
        new_y = params.c * x + params.d * y + params.f

        return int(new_x), int(new_y)


class IFSParams:
    def __init__(self, a, b, c, d, e, f):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.f = f


def show_img(img):
    plt.imshow(img, cmap='gray', vmin=0, vmax=1)
    plt.show()


def generate_L_shaped_image():
    img = np.ones((SIZE, SIZE))

    img[:, :548] = 0
    img[1600:, :] = 0

    return img


main()
