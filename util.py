
import matplotlib.pyplot as plt
import numpy as np
import imageio
import networkx as nx

def draw_graph(G, pos, ax=None):
    nx.draw(G, pos, ax, with_labels=True)

def draw_graph_file(G, filename, n, m, k):
    pos = get_graph_pos(n, m, k)
    fig, ax = plt.subplots(figsize=(15, 10))
    draw_graph(G, pos, ax)
    plt.savefig(filename)

def draw_graph_colored(G, pos, edge_colors, ax=None):
    nx.draw(G, pos, ax, edge_color=edge_colors, with_labels=True)

def get_graph_pos(n, m, k):
    pos = {
        (0, 0) : ((m+1) / 2, 0),
        (n+1, 0) : ((m+1) / 2, 2*(n+1))
    }
    for l in range(1, n + 1):
        for v in range(1, m + 1):
            pos[(l, v)] = (v, 2*l - 0.25 + (0.5 / (m/2)**2)*(v - (m/2))**2)

    return pos

def draw_graph_pos(G, n, m, k, ax=None):
    draw_graph(G, get_graph_pos(n, m, k), ax)

def draw_graph_path(G, silver, normal, pos, ax=None):
    non_path = [e for e in G.edges if (e not in silver and e not in normal)]

    nx.draw_networkx_nodes(G, pos, G.nodes, ax=ax)

    #nx.draw_networkx_edges(G, pos, edgelist=non_path, ax=ax, edge_color="black")
    nx.draw_networkx_edges(G, pos, edgelist=silver, ax=ax, edge_color="g")
    nx.draw_networkx_edges(G, pos, edgelist=normal, ax=ax, edge_color="b")

def draw_graph_path_image(G, silver, normal, pos):
    fig, ax = plt.subplots(figsize=(15, 10))
    draw_graph_path(G, silver, normal, pos, ax)
    fig.canvas.draw()  # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return image

def draw_graph_path_animated(G, path, filename, n, m, k):
    kwargs_write = {'fps': 1.0, 'quantizer': 'nq'}
    pos = get_graph_pos(n, m, k)
    images = []
    silver = []
    normal = []
    visited_vertices = [(0, 0)]
    for i in range(len(path)):
        edge = path[i]
        v1, v2 = edge
        if v2 not in visited_vertices:
            silver.append(edge)
            visited_vertices.append(v2)
        else:
            normal.append(edge)
        images.append(draw_graph_path_image(G, silver, normal, pos))
    print("[*] Silver edges: {0}, expected: {1}".format(len(silver), n*m+1))
    imageio.mimsave(filename, images, fps=1)

def reverse_path(path):
    new_path = []
    for i in range(len(path)-1, -1, -1):
        edge = path[i]
        v1, v2 = edge
        new_path.append((v2, v1))
    return new_path

def flip_graph(G, n, m, k):
    new = nx.Graph()
    #new.add_nodes_from(G.nodes)
    edges = []
    for edge in reversed([e for e in G.edges]):
        v1, v2 = edge
        nv1, nv2 = v1, v2
        if v1[0] != 0 and v1[0] != n+1:
            nv1 = (v1[0], m - v1[1]+1)
        if v2[0] != 0 and v2[0] != n+1:
            nv2 = (v2[0], m - v2[1]+1)
        new.add_edge(v1, v2)
    return new
