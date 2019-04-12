import networkx as nx
import os
import matplotlib.pyplot as plt
import util

def create_graph(n, m, k, file):
    G = nx.Graph()

    #nodes

    for l in range(1, n+1):
        for v in range(1, m+1):
            G.add_node((l, v))

    G.add_node((0, 0))
    G.add_node((n+1, 0))

    #edges

    for l in range(1, n+1):
        for v in range(1, m+1):
            if v % 2 == 0:
                if v < m:
                    G.add_edge((l, v), (l, v+1))
                else:
                    G.add_edge((l, v), (l, 1))

    for v in range(1, m+1):
        G.add_edge((0, 0), (1, v))
        G.add_edge((n+1, 0), (n, v))

    for l in range(1, n):
        for j in range(1, m+1):
            line = file.readline()
            numbers = line.split(" ")
            numbers = [int(x) for x in numbers]
            for v in numbers:
                G.add_edge((l, j), (l+1, v))

    for l in range(1, n+1):
        for v in range(1, m+1):
            if v % 2 == 1:
                if v < m:
                    G.add_edge((l, v), (l, v+1))
                else:
                    G.add_edge((l, v), (l, 1))

    return G

from networkx.utils import arbitrary_element

D_CACHE = {}

def degree(G, n):
    global D_CACHE
    if n in D_CACHE:
        return D_CACHE[n]
    D_CACHE[n] = G.degree(n)
    return D_CACHE[n]

def best_element(edges, m, player = 1, silver = [], G = None):
    best_for_now = arbitrary_element(edges)
    print(silver)
    max_l = 100000
    max_deg = 0
    #best_for_now = edges[0]
    for e in edges:
        if e not in silver:
            return e
        else:
            best_for_now = e
            continue
        v1, v2 = e
        ne = v1[1] + 1
        pre = v1[1] - 1
        if ne > m:
            ne = 1
        if pre < 1:
            pre = m
        if v1[0] == v2[0] and ((v1[1] % 2 == 1 and (ne if player == 1 else pre) == v2[1]) or (v1[1] % 2 == 0 and (pre if player == 1 else ne) == v2[1])):
            return e
        ed = v1, v2
        if v1 > v2:
            ed = v2, v1
        l = ed[1][0]
        if (v1[0] != v2[0]) and ed not in silver:
            #best_for_now = e
            #return e
            deg = degree(G, v2)
            if deg > max_deg:
                return e
                best_for_now = e
                max_deg = deg

    return best_for_now


def random_tour(G, source):
    v = source
    W = []
    while len(G.edges(v)) > 0:
        _, v_next = best_element(G.edges(v))
        W.append((v, v_next))
        G.remove_edge(v, v_next)
        v = v_next
    return W


def euler(G, source):
    W = random_tour(G, source)
    if len(W) == 0:
        return W
    v_slow, v_next = W[0]
    new_W = []
    i = 0
    for e in W:
        new_W.append(e)
        v_prev, v = e
        if len(G.edges(v)) > 0:
            new_W += random_tour(G, v)
    return new_W

def euler2(G, source, m, player = 1, silver = []):
    global D_CACHE
    D_CACHE = {}
    stack = [source]
    C = []
    while len(stack) > 0:
        current = stack[-1]
        if G.degree(current) == 0:
            stack.pop()
            C.append(current)
        else:
            _, n = arbitrary_element(G.edges(current))#best_element(G.edges(current), m, player, silver, G)
            #D_CACHE[n] = degree(G, n) - 1
            #D_CACHE[current] = degree(G, current) - 1
            G.remove_edge(current, n)
            stack.append(n)
    ret = []
    C.reverse()
    prev = None
    silver = set()
    visited = set()
    for v in C:
        if prev is not None:
            ret.append((prev, v))
            if False:
                if v not in visited:
                    visited.add(v)
                    silver.add((prev, v))
                    silver.add((v, prev))
        prev = v
    return ret

def get_silver(path):
    silver = set()
    normal = []
    visited_vertices = set()
    visited_vertices.add((0,0))
    for i in range(len(path)):
        edge = path[i]
        v1, v2 = edge
        if v2 not in visited_vertices:
            ed = v1, v2
            if v1 > v2:
                ed = v2, v1
            silver.add(ed)
            visited_vertices.add(v2)
        else:
            normal.append(edge)
    return silver

def correct_v(v, l, player):
    return (v % 2 == l % 2) if player == 1 else (v % 2 != l % 2)

def weird_mod(v, m):
    if v > m:
        return 1
    if v < 1:
        return m
    return v

def direction(layer, v_layer, v, player):
    if player == 1:
        if layer % 2 == v_layer % 2:
            if v % 2 == 0:
                return 1
            return -1
        else:
            if v % 2 == 0:
                return -1
            return 1
    else:
        if layer % 2 == v_layer % 2:
            if v % 2 == 0:
                return -1
            return 1
        else:
            if v % 2 == 0:
                return 1
            return -1


def ceremony_tour(G, reduced, source, n, m, k, layer, player = 1):
    path = []

    current = None

    while current != source:
        if current is None:
            current = source
        next_vert = None#arbitrary_element(reduced.edges(current))
        l, v = current
        if l == layer:
            if (v == 0 and l == 0) or not correct_v(v, l, player):
                for e in reduced.edges(current):
                    _, v2 = e
                    l_next, v_next = v2
                    if l_next == layer + 1 and (correct_v(v_next, l_next, player) or v_next == 0):
                        next_vert = v2
                        break
            else:
                ideal_v = weird_mod(v + direction(layer, l, v, player), m)
                next_vert = (l, ideal_v)

        elif l == layer + 1:
            if (v == 0 and l == n+1) or not correct_v(v, l, player):
                for e in reduced.edges(current):
                    _, v2 = e
                    l_next, v_next = v2
                    if l_next == layer and (correct_v(v_next, l_next, player) or v_next == 0):
                        next_vert = v2
                        break
            else:
                ideal_v = weird_mod(v + direction(layer, l, v, player), m)
                next_vert = (l, ideal_v)

        if next_vert is None:
            _, next_vert = arbitrary_element(reduced.edges(current))
        edge = current, next_vert
        G.remove_edge(current, next_vert)
        reduced.remove_edge(current, next_vert)
        path.append(edge)
        current = next_vert

    return path

def reduce_graph(G, n, m, k):
    reduced = G.copy()

    for l in range(1, n):
        nodes = [(l, x) for x in range(1, m + 1)] + [(l + 1, x) for x in range(1, m + 1)]
        sub2 = G.subgraph(nodes)
        edges2 = {e for e in sub2.edges if e[0][0] != e[1][0]}
        sub2 = sub2.edge_subgraph(edges2)
        matching = find_perfect_matching(sub2, [(l, x) for x in range(1, m + 1)])
        for e in edges2:
            if e not in matching:
                reduced.remove_edge(e[0], e[1])
    return reduced


def solve_graph(G, reduced, n, m, k, player = 1, silver = []):
    full_path = []
    for l in range(0, n+1):
        if l == 0:
            while len(full_path) != 3 * (m / 2):
                full_path += ceremony_tour(G, reduced, (0, 0), n, m, k, l, player)
        else:
            new_full_path = full_path
            for i in range(len(full_path) - 1, -1, -1):
                edge = full_path[i]
                v1, v2 = edge
                if v1[0] == l and reduced.degree(v1) > 0:
                    path = ceremony_tour(G, reduced, v1, n, m, k, l, player)
                    new_full_path = new_full_path[:i] + path + new_full_path[i:]
            full_path = new_full_path

    new_full_path = full_path
    for i in range(len(full_path) - 1, -1, -1):
        edge = full_path[i]
        v1, v2 = edge
        if G.degree(v1) > 0:
            path = euler2(G, v1, m, player, silver)
            new_full_path = new_full_path[:i] + path + new_full_path[i:]
    full_path = new_full_path
    return full_path, get_silver(full_path)


def solve_graph_old2(G, n, m, k, player = 1, silver = []):
    global D_CACHE
    D_CACHE = {}
    #return euler2(G, (0, 0), m, player, silver)
    full_path = []
    source = (0, 0)
    for l in range(0, n+1):
        nodes = []
        if l == 0:
            nodes.append((0, 0))
        else:
            nodes += [(l, m) for m in range(1, m+1)]
        if l == n:
            nodes.append((n+1, 0))
        else:
            nodes += [(l+1, m) for m in range(1, m + 1)]
        sub = nx.Graph()
        for e in G.subgraph(nodes).edges:
            sub.add_edge(e[0], e[1])
        for v in range(1, m+1, 2):
            l_add = 0 if player == 1 else 1
            v_act = v + 1
            if v == m:
                v_act = 1
            try:
                sub.remove_edge((l+l_add, v), (l+l_add, v_act))
            except:
                pass
        for v in range(2, m+1, 2):
            l_add = 1 if player == 1 else 0
            v_act = v + 1
            if v == m:
                v_act = 1
            try:
                sub.remove_edge((l+l_add, v), (l+l_add, v_act))
            except:
                pass
        #print(sub.edges)
        #nx.draw(sub, with_labels=True)
        #plt.show()
        if l == 0:
            if player == 1:
                for v in range(m, 0, -2):
                    full_path += [((0, 0), (1, v)), ((1, v), (1, v-1)), ((1, v-1), (0, 0))]
            else:
                full_path = euler2(G, (0,0), m, player, silver)
        else:
            new_full_path = full_path
            visited_verts = []
            for i in range(len(full_path)-1, -1, -1):
                edge = full_path[i]
                v1, v2 = edge
                if v1[0] == l and v1 not in visited_verts:
                    path = euler2(sub, v1, m, player, silver)
                    new_full_path = new_full_path[:i] + path + new_full_path[i:]
                    visited_verts += [v1 for v1, v2 in path]
            full_path = new_full_path

    return full_path, get_silver(full_path)

def solve_graph_old(G, n, m, k, player = 1):
    start_vert = (0, 0)
    prev_vertical_edges = []
    full_path = []
    for l in range(0, n + 1):
        matching = []
        edges = []
        vert_edges = []
        if l == 0:
            for v in range(1, m + 1):
                vert_edges.append(((0, 0), (1, v)))
        elif l == n:
            for v in range(1, m + 1):
                vert_edges.append(((n, v), (n + 1, 0)))
        else:
            nodes = [(l, x) for x in range(1, m + 1)] + [(l + 1, x) for x in range(1, m + 1)]
            sub2 = G.subgraph(nodes)
            edges2 = [e for e in sub2.edges if e[0][0] != e[1][0]]
            sub2 = sub2.edge_subgraph(edges2)
            matching = find_perfect_matching(sub2, [(l, x) for x in range(1, m+1)])
            vert_edges += matching
        if l != n:
            for v in range(1 if player == 1 else 2, m + 1, 2):
                if v < m:
                    edges.append(((l + 1, v), (l + 1, v + 1)))
                else:
                    edges.append(((l + 1, v), (l + 1, 1)))
        if l != 0:
            for v in range(2 if player == 1 else 1, m + 1, 2):
                if v < m:
                    edges.append(((l, v), (l, v + 1)))
                else:
                    edges.append(((l, v), (l, 1)))
        if l > 1:
            nodes = [(l - 1, x) for x in range(1, m + 1)] + [(l, x) for x in range(1, m + 1)]
            sub2 = G.subgraph(nodes)
            other_vert = [e for e in sub2.edges if e not in prev_vertical_edges and e[0][0] != e[1][0]]
            vert_edges += other_vert
        prev_vertical_edges = matching
        edges += vert_edges
        sub = nx.Graph()
        for e in edges:
            u, v = e
            sub.add_edge(u, v)
        print(sub.edges((0,0)))

        nx.draw(sub, with_labels=True)
        #plt.show()
        visited_verts = []
        new_full_path = full_path
        if l == 0:
            path = [e for e in nx.eulerian_circuit(sub, start_vert)]
            new_full_path += path
        else:
            for i in range(len(full_path)-1, -1, -1):
                edge = full_path[i]
                v1, v2 = edge
                if v1[0] == l and v1 not in visited_verts:
                    reachable = nx.node_connected_component(sub, v1)
                    reach_edges = [e for e in sub.edges if e[0] in reachable and e[1] in reachable]
                    reach = nx.Graph()
                    for e in reach_edges:
                        reach.add_edge(e[0], e[1])
                    #reach = sub.subgraph(reachable)
                    nx.draw(reach, with_labels=True)
                    #plt.show()
                    path = []
                    try:
                        path = [e for e in nx.eulerian_circuit(reach, v1)]
                    except:
                        util.draw_graph_path_animated(G, full_path, "./error_before.gif", n, m, k)
                        util.draw_graph_path_animated(G, new_full_path, "./error_right_before.gif", n, m, k)
                        util.draw_graph_file(sub, "./error_subgraph.png", n, m, k)
                        util.draw_graph_file(reach, "./error_reach.png", n, m, k)
                        print("[!] ERROR", l)
                        return full_path
                    new_full_path = new_full_path[:i] + path + new_full_path[i:]
                    visited_verts += reachable

        full_path = new_full_path
        #util.draw_graph_path_animated(G, full_path, "./solution_player-{1}_{0}.gif".format(l, player), n, m, k)
    return full_path

def find_perfect_matching(G, sets):
    matching = nx.algorithms.bipartite.maximum_matching(G, sets)
    return {(key, value) for key, value in matching.items()}

def read_and_solve_one(file):
    numbers = file.readline()
    numbers = [int(x) for x in numbers.split(" ")]
    n, m, k = numbers
    G = create_graph(n, m, k, file)

    path, path2 = solve_graph_twice(G, n, m, k)
    print_paths(path, path2)

def solve_graph_twice(G, n, m, k):
    G2 = G.copy()
    path, silver = solve_graph(G, n, m, k)

    path2, _ = solve_graph(G2, n, m, k, 2, silver)
    return (path, path2)

def print_paths(path, path2):
    for v1, v2 in path:
        print("{} {}".format(v2[0], v2[1]))
    for v1, v2 in path2:
        print("{} {}".format(v2[0], v2[1]))

def read_and_solve(file):
    testcases = int(file.readline())
    for t in range(0, testcases):
        #print("Testcase " + str(t))
        print("Case {}: Yes".format(t+1))
        read_and_solve_one(file)



def main():
    filein = "testcase.txt"
    fileout = "test1.out.txt"
    with open(filein) as f:
        read_and_solve(f)

if __name__ == "__main__":
    main()
