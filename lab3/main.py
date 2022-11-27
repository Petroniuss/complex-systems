import os
from collections import Counter

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
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


## ----------------------------------

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


## ----------------------------------
def avg_shortest_path(graph: nx.Graph):
    return nx.average_shortest_path_length(graph)


def efficiency(graph: nx.Graph):
    return nx.global_efficiency(graph)


def average_shortest_path_length(graph: nx.Graph):
    """average of average lengths of each connected component"""
    avg_lengths = 0
    components = list(nx.connected_components(graph))
    for C in (graph.subgraph(c).copy() for c in components):
        avg_lengths += nx.average_shortest_path_length(C)

    return avg_lengths / len(components)


def measure_resiliency(graph: nx.Graph, title: str, select_node_fun):
    graph = graph.copy()
    nodes = graph.number_of_nodes()

    x, L, E = [], [], []
    for i in range(nodes - 1):
        node = select_node_fun(graph)
        graph.remove_node(node)

        L.append(average_shortest_path_length(graph))
        E.append(nx.global_efficiency(graph))
        x.append(i)

    fs = (np.array(x, dtype=np.float32) / nodes * 100.)

    plt.title(f'{title}: Average Shortest Path Length')
    plt.plot(fs, L, c='b')
    plt.show()

    plt.title(f'{title}: Global efficiency')
    plt.plot(fs, E, c='r')
    plt.show()


def random_error(graph: nx.Graph):
    return np.random.choice(graph.nodes())


def attack(graph: nx.Graph):
    degrees = [degree + 1 for _node, degree in graph.degree()]
    degrees_sum = sum(degrees)
    return np.random.choice(graph.nodes(), p=[deg / degrees_sum for deg in degrees])


def main():
    graphs = read_graphs()
    print(graphs)

    graph, name = graphs[0]
    # draw_degree_distribution(graph, name)
    # draw_graph(graph, name, nx.draw_kamada_kawai)

    measure_resiliency(graph, f'random error - {name}', random_error)
    measure_resiliency(graph, f'attack - {name}', attack)


if __name__ == '__main__':
    main()
