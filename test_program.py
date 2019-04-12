import os
import subprocess
from subprocess import Popen, PIPE, STDOUT
import test
import networkx as nx
import util
import time
import main

program = ["./submit"]

print("[*] Generating random test case...")
n, m, k = 1000, 14, 5
(G, pos) = test.create_random_test(n, m, k)
testcase = test.random_test_to_str(G, n, m, k)
print("[*] Random Test Case:")
with open("testcase.txt", "w") as f:
    f.write(testcase)
#print(testcase)
print("[*] Running Program: {0}...".format(program))

start = time.time()
p = Popen(program, stdout=PIPE, stdin=PIPE, stderr=PIPE)
output = "".join([s.decode("utf-8") for s in p.communicate(input=testcase.encode("utf-8"))])
end = time.time()

print("[*] Program output:")
print(output)

print("Time taken for test: {}s".format(end - start))

paths = []
variables = [()]
current_path = None
prev_vertex = None
for line in output.split("\n"):
    if "Variables" in line:
        if current_path is not None:
            paths.append(current_path)
        nums = line.split(" ")
        variables.append((int(nums[1]), int(nums[2]), int(nums[3])))
        current_path = []
        prev_vertex = (0, 0)
    elif current_path is not None and "Case" not in line and len(line) > 2:
        nums = line.split(" ")
        current = int(nums[0]), int(nums[1])
        if prev_vertex is not None:
            current_path.append((prev_vertex, current))
        prev_vertex = current

if current_path is not None:
    paths.append(current_path)

i = 1

for path in paths:
    mid = int(len(path)/2)
    first, second = path[:mid], path[mid:]
    n, m, k = variables[i]
    silver1 = main.get_silver(first)
    silver2 = main.get_silver(second)
    print("[*] Path length: {0} / {2}, expected length: {1}".format(len(first), m*(n*k - k + n + 2), len(second)))
    print("[*] Silver length: {0} / {1}, expected length: {}", len(silver1), len(silver2))
    for e in first:
        if first.count(e) > 1 or (e[1], e[0]) in first:
            print("[!!!] Error, we have an edge appearing multiple times!: ", e)
    for e in silver1:
        if e in silver2:
            print("[!!!] Error, we have edges being colored multiple times!: ", e)

    for player in range(1, 3):
        G = nx.Graph()
        p = first if player == 1 else second
        for e in p:
            G.add_edge(e[0], e[1])

        util.draw_graph_path_animated(G, p, "./solution_program_{0}_player_{1}.gif".format(i, player), n, m, k)
    i += 1

