import matplotlib
matplotlib.use('TKAgg')
import networkx as nx
import random
import matplotlib.pyplot as plt
import main
import util


def create_random_test(n, m, k):
    G = nx.Graph()
    pos = {
        (0, 0) : ((m+1) / 2, 0),
        (n+1, 0) : ((m+1) / 2, 2*(n+1))
    }
    # nodes

    for l in range(1, n + 1):
        for v in range(1, m + 1):
            G.add_node((l, v))
            pos[(l, v)] = (v, 2*l - 0.25 + (0.5 / (m/2)**2)*(v - (m/2))**2)

    G.add_node((0, 0))
    G.add_node((n + 1, 0))

    # edges

    for l in range(1, n+1):
        for v in range(1, m+1):
            if v % 2 == 0:
                if v < m:
                    G.add_edge((l, v), (l, v+1))
                else:
                    G.add_edge((l, v), (l, 1))

    for l in range(1, n+1):
        for v in range(1, m+1):
            if v % 2 == 1:
                if v < m:
                    G.add_edge((l, v), (l, v+1))
                else:
                    G.add_edge((l, v), (l, 1))

    for v in range(1, m+1):
        G.add_edge((0, 0), (1, v))
        G.add_edge((n+1, 0), (n, v))

    for l in range(1, n):
        G.add_edges_from(random_selection(1, k, m, l))
        G.add_edges_from(random_selection(0, k, m, l))


    return (G, pos)

def random_selection(even, k, m, l):
    init = list(range(1+even, m+1, 2))
    first = init.copy()
    random.shuffle(first)
    good_lists = [first]
    for i in range(1, k):
        isGood = False
        next_list = init.copy()
        while not isGood:
            isGood = True
            next_list = init.copy()
            random.shuffle(next_list)
            for g_list in good_lists:
                for j in range(0, len(g_list)):
                    if g_list[j] == next_list[j]:
                        isGood = False
                        break
                if not isGood:
                    break
        good_lists.append(next_list)
    ret = []
    for g_list in good_lists:
        v = 1 + even
        for x in g_list:
            ret.append(((l, v), (l+1, x)))
            v += 2
    return ret



def incr_by_one(v, m):
    i = v[1]
    i += 1
    if i > m:
        i = 1
    return (v[0], i)

def solve_random_test(n, m, k):
    (G, pos) = create_random_test(n, m, k)
    #nx.draw(G, pos=pos, with_labels=True)
    #plt.show()
    print(G.edges((1,1)))
    reduced = main.reduce_graph(G, n, m, k)
    reduced2 = reduced.copy()
    G2 = G.copy()
    path, silver = main.solve_graph(G, reduced, n, m, k)
    print(path)
    #util.draw_graph_path_animated(G, path, "./solution.gif", n, m, k)
    print("[*] Path length: {0}, expected length: {1}".format(len(path), m*(n*k - k + n + 2)))
    print("[*] Infos:")
    #print(flipped.edges((0,0)))
    #print(nx.utils.arbitrary_element(flipped.edges((0,0))))
    pos = util.get_graph_pos(n, m, k)
    #nx.draw(flipped, pos=pos, with_labels=True)
    #plt.show()
    path2, _ = main.solve_graph(G2, reduced2, n, m, k, 2, silver)
    #util.draw_graph_path_animated(G2, path2, "./solution_player2.gif", n, m, k)
    print("[*] Path2 length: {0}, expected length: {1}".format(len(path2), m * (n * k - k + n + 2)))
    print("[*] Test Case:")
    #print_random_test(G, n, m, k)
    print("[*] Actual Output:")
    print("Case 1: Yes")
    #for v1, v2 in path2:
    #    print("{} {}".format(v2[0], v2[1]))
    #for v1, v2 in path:
    #    print("{} {}".format(v2[0], v2[1]))
    silver1 = main.get_silver(path)
    silver2 = main.get_silver(path2)
    print(len(silver1), len(silver2))
    for e in silver1:
        if e in silver2:
            print("[!!!] Error, we have edges being colored multiple times!: ", e)

def random_test_to_str(G, n, m, k):
    res = str(1) + "\n"
    res += "{} {} {}".format(n, m, k) + "\n"
    for l in range(1, n):
        for v in range(1, m+1):
            edges = G.edges((l, v))
            stelae = [str(v2[1]) for v1, v2 in edges if v2[0] == l+1]
            res += " ".join(stelae) + "\n"

    return res

def print_random_test(G, n, m, k):
    print(random_test_to_str(G, n, m, k))

if __name__ == "__main__":
    #print_random_test(3, 8, 3)
    solve_random_test(300, 12, 5)


