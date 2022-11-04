from __future__ import annotations

from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt

SIZE = 2048


def main():
    img = generate_L_shaped_image()
    show_img(img)

    ifs_0_params = IFSParams(
        a=0.0,
        b=-0.5,
        c=0.5,
        d=0.0,
        e=0.5,
        f=0.0
    )

    ifs_1_params = IFSParams(
        a=0.0,
        b=0.5,
        c=-0.5,
        d=0.0,
        e=0.5,
        f=0.5
    )

    ifs_2_params = IFSParams(
        a=0.5,
        b=0.0,
        c=0.0,
        d=0.5,
        e=0.25,
        f=0.5
    )

    #
    # ifs_3_params = IFSParams(
    #     a=0.849,
    #     b=0.037,
    #     c=-0.037,
    #     d=0.849,
    #     e=0.075,
    #     f=0.1830
    # )

    ifs = IFS(params_list=[
        ifs_0_params,
        ifs_1_params,
        ifs_2_params
    ])

    for i in range(10):
        img = ifs.iteration(img)
        show_img(img)


def show_img(img):
    plt.imshow(img, cmap='gray', vmin=0, vmax=1)
    plt.show()


def generate_L_shaped_image():
    img = np.ones((SIZE, SIZE))

    img[:, :548] = 0
    img[1600:, :] = 0

    return img


class IFS:
    def __init__(self, params_list: List[IFSParams]):
        self.params_list = params_list
        pass

    def iteration(self, img):
        new_img = np.ones((SIZE, SIZE))
        for y in range(SIZE):
            for x in range(SIZE):
                if img[y, x] == 0:
                    for i in range(len(self.params_list)):
                        new_x, new_y = self.compute_new_position((x, y), i)
                        new_img[new_y, new_x] = 0

        return new_img

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


main()
