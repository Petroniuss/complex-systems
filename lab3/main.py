import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def algo(m, n, m_0=1):
    G = nx.Graph()
    degrees = np.array([0] * n)
    for u in range(m_0):
        G.add_node(u)
        degrees[u] = 1

    degrees_sum = m_0
    for u in range(m_0, n):
        for edge_i in range(m):
            v = np.random.choice(n, p=degrees / degrees_sum)
            G.add_edge(u, v)
            degrees[v] += 1
            degrees_sum += 1

        degrees[u] = m + 1
        degrees_sum += m + 1

    return G


def show_graph(G):
    nx.draw(G)
    plt.show()


def main():
    n = 10 ** 2
    m = 3
    m_0 = 1

    G = algo(m, n, m_0)
    show_graph(G)


if __name__ == '__main__':
    main()
