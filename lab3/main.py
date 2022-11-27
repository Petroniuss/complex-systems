import os
from collections import Counter

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from tqdm import tqdm


def scale_free_graph(m, n, m_0=1):
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


def generate_graphs():
    def show_graph(G):
        nx.draw(G)
        plt.show()

    def generate_graph(m, n):
        G = scale_free_graph(m, n)
        show_graph(G)
        nx.write_gpickle(G, f"./graphs/graph_n_{n}_m_{m}.gpickle")

    params = [(3, 10 ** 2), (3, 10 ** 3), (2, 10 ** 4)]
    for (m, n) in tqdm(params):
        generate_graph(m, n)


def read_graphs():
    graphs_dir = './graphs'
    graphs = []
    for path in os.listdir(graphs_dir):
        path_full = os.path.join(graphs_dir, path)
        if os.path.isfile(path_full):
            G = nx.read_gpickle(path_full)
            graphs.append((G, path))

    return sorted(graphs, key=lambda e: e[1], reverse=True)


def draw_degree_distribution(graph: nx.Graph, name: str):
    degree_count = Counter([degree for _node, degree in graph.degree()])
    xs = np.array(list(degree_count.keys()))
    ys = np.array(list(degree_count.values()))

    def compute_coeffs(xs, ys):
        xlog = np.log(xs)
        ylog = np.log(ys)

        A = np.vstack([xlog, np.ones(len(xlog))]).T
        m, c = np.linalg.lstsq(A, ylog)[0]
        return xlog, ylog, m, c

    xlog, ylog, m, c = compute_coeffs(xs, ys)
    plt.scatter(xlog, ylog)
    plt.title(f'{name}, coeff: {m:.2f} ')
    plt.plot(xlog, m * xlog + c, 'orange')
    plt.show()


def draw_graph(graph: nx.Graph, name: str, nx_draw_graph_fun):
    nx_draw_graph_fun(graph, node_color=[v for _, v in graph.degree()])
    plt.title(name)
    plt.show()


def main():
    graphs = read_graphs()
    print(graphs)

    graph, name = graphs[0]
    # draw_degree_distribution(graph, name)
    draw_graph(graph, name, nx.draw_kamada_kawai)

    # ..



if __name__ == '__main__':
    main()
