import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from collections import defaultdict
from tqdm import tqdm


def main():
    simulation_1()


def simulation_1():
    n_iters = 3000
    L = 20
    p = 0.5
    alpha = 4
    simulation(n_iters=n_iters, L=L, p=p, alpha=alpha)


def simulation(n_iters: int, L: int, p: float, alpha: int):
    def show(M):
        pprint(M)
        plt.imshow(M, cmap='binary')
        plt.show()

    M = automata(L, p)
    show(M)
    for _ in tqdm(range(n_iters)):
        automata_iteration(alpha=alpha, M=M)

    show(M)


def automata_iteration(alpha: float, M: np.ndarray):
    n = len(M)

    def clamp_idx(idx: int):
        return idx % n

    def neighbourhood(j: int, i: int):
        j_min = j - 1
        i_min = i - 1
        j_max = j + 2
        i_max = i + 2

        indices = np.array([[clamp_idx(nj), clamp_idx(ni)]
                            for nj in range(j_min, j_max)
                            for ni in range(i_min, i_max)]
                           )
        Y = np.transpose(indices)[0]
        X = np.transpose(indices)[1]
        xs = M[Y, X].reshape(3, 3)

        return xs

    def count_neighbours(j: int, i: int):
        agent_type = M[j, i]
        neigh = neighbourhood(j, i)
        return np.sum(neigh == agent_type) - 1

    def pick_empty_spot(j: int, i: int):
        # a bug might be here.
        # bug is here.. fix it!
        neigh = neighbourhood(j, i)
        empty_spots = np.transpose(np.where(neigh == 0))
        empty_spots_size = len(empty_spots)
        if empty_spots_size == 0:
            return None

        chosen_idx = np.random.randint(low=0, high=len(empty_spots))
        dj, di = empty_spots[chosen_idx]

        chosen_j = clamp_idx(j + (dj - 1))
        chosen_i = clamp_idx(i + (di - 1))

        return chosen_j, chosen_i

    # dictionary new_position => [old_position].
    state = defaultdict(list)
    for j in range(n):
        for i in range(n):
            agent_type = M[j][i]
            if agent_type > 0:
                s = count_neighbours(j, i)
                if s < alpha:
                    empty_spot = pick_empty_spot(j, i)
                    if empty_spot is None:
                        continue

                    if M[empty_spot[0]][empty_spot[1]] > 0:
                        raise Exception('foo')

                    state[empty_spot].append((j, i))

    # move agents.
    for ((nj, ni), old_positions) in state.items():
        if len(old_positions) > 1:
            continue

        (j, i) = old_positions[0]
        M[nj, ni] = M[j, i]
        M[j, i] = 0


def automata(L: int, p: float) -> np.ndarray:
    M = np.zeros((L, L), dtype=np.int32)

    size = L * L
    taken = 0
    taken_limit = int(p * size)
    while taken < taken_limit:
        [idx_y, idx_x] = np.random.randint(low=0, high=L, size=2)
        if M[idx_y][idx_x] == 0:
            agent_type = np.random.randint(low=1, high=3)
            M[idx_y][idx_x] = agent_type
            taken += 1

    return M


if __name__ == '__main__':
    main()
