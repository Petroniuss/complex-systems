from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation


# 1. Deterministic Cellular Automata
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


def evolution(mappings: dict, n=100, p=0.5, iterations=100):
    ones_num = int(p * n)
    zeros_num = n - ones_num
    xs = [1] * ones_num + [0] * zeros_num
    np.random.shuffle(xs)

    first_row = ''.join(map(str, xs))
    return algo(first_row, mappings, iterations)


# 2. Nagel-Schreckenberg model
def simulate(L=100, v_max=2, p=0.2, vehicle_density=0.24, n_iterations=100):
    initial_state = generate_initial_state(L, vehicle_density)
    states = [initial_state]
    snapshots = [to_image(initial_state, v_max)]
    for i in range(n_iterations):
        new_state = compute_new_state(states[-1], v_max, p)
        new_image = to_image(new_state, v_max)

        states.append(new_state)
        snapshots.append(new_image)

    animate(snapshots, p, vehicle_density)


def animate(snapshots, p, vehicle_density):
    fps = 5

    fig = plt.figure(figsize=(8, 8))
    a = snapshots[0]
    im = plt.imshow(a, cmap='binary')

    def animate_func(i):
        im.set_array(snapshots[i])
        return [im]

    anim = animation.FuncAnimation(
        fig,
        animate_func,
        frames=len(snapshots),
        interval=1000 / fps,
    )
    writer = animation.FFMpegWriter(fps=fps)
    anim.save(f'./recordings/anim_p_{p:.2f}v_dens_{vehicle_density:.2f}.mp4', writer=writer)
    plt.show()


def compute_distances(state: List[int]) -> List[int]:
    n = len(state)
    distances = [0] * n
    last_vehicle_idx = None
    for i in range(n):
        if state[i] >= 0:
            if last_vehicle_idx is not None:
                distances[last_vehicle_idx] = i - last_vehicle_idx
            last_vehicle_idx = i

    for i in range(n):
        if state[i] >= 0:
            distances[last_vehicle_idx] = i + (n - last_vehicle_idx)

    return distances


def move_vehicles(state: List[int]) -> List[int]:
    n = len(state)
    moved = [-1] * n
    for i in range(n):
        velocity = state[i]
        if velocity >= 0:
            new_position = (i + velocity) % n
            moved[new_position] = velocity

    return moved


def compute_new_state(current_state: List[int], v_max: int, p: float):
    n = len(current_state)
    distances = compute_distances(current_state)
    new_state = [-1] * n

    for i in range(n):
        if current_state[i] >= 0:
            current_velocity = current_state[i]
            current_distance = distances[i]

            new_velocity_partial = min(
                current_velocity + 1,
                current_distance - 1,
                v_max
            )

            velocity = np.random.choice([
                max(new_velocity_partial - 1, 0),
                new_velocity_partial
            ], p=[p, 1 - p])

            new_state[i] = velocity

    new_state = move_vehicles(new_state)
    return new_state


def generate_initial_state(L, vehicle_density):
    vehicles_num = int(L * vehicle_density)
    empty_space_num = L - vehicles_num
    state = [0] * vehicles_num + [-1] * empty_space_num
    np.random.shuffle(state)

    return state


def to_image(state: List[int], v_max, vehicle_width=1, vehicle_height=12):
    n = len(state)
    image = np.zeros((vehicle_height, n * vehicle_width))
    for i in range(n):
        image[:, i * vehicle_width: i * (vehicle_width + 1) + 1] = (state[i] + 1) / v_max

    return image


if __name__ == '__main__':
    # ex 1.
    first_row = '111110101100011010001000'
    mapping = {
        '111': '1',
        '110': '0',
        '101': '1',
        '100': '1',
        '011': '1',
        '010': '0',
        '001': '0',
        '000': '0'
    }
    image = evolution(mapping, n=100, p=0.5, iterations=100)
    plt.imshow(image, cmap='binary')
    plt.show()

    # ex 2.
    simulate(vehicle_density=0.24)
    simulate(vehicle_density=0.48)
    simulate(vehicle_density=0.2, p=.4)
    simulate(vehicle_density=0.2, p=.1)

    pass
