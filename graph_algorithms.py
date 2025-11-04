import networkx as nx
import matplotlib.pyplot as plt
import random
from heapq import heappush, heappop
class Graph:
    def __init__(self, vertices, edges, gtype="undirected"):
        self.vertices = vertices
        self.edges = edges
        self.gtype = gtype
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = nx.DiGraph() if self.gtype == "directed" else nx.Graph()
        for u, v, w in self.edges:
            graph.add_edge(u, v, weight=w)
        return graph

def dijkstra(graph, source):
    dist = {v: float("inf") for v in graph.vertices}
    dist[source] = 0
    pq = [(0, source)]

    while pq:
        d, u = heappop(pq)
        if d > dist[u]:
            continue
        for v in graph.graph.neighbors(u):
            w = graph.graph[u][v]["weight"]
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heappush(pq, (dist[v], v))
    return dist


# =============================
# KRUSKAL MST
# =============================
def find(parent, i):
    if parent[i] != i:
        parent[i] = find(parent, parent[i])
    return parent[i]


def union(parent, rank, x, y):
    xr, yr = find(parent, x), find(parent, y)
    if rank[xr] < rank[yr]:
        parent[xr] = yr
    elif rank[xr] > rank[yr]:
        parent[yr] = xr
    else:
        parent[yr] = xr
        rank[xr] += 1


def kruskal_mst(graph):
    edges = sorted(graph.edges, key=lambda x: x[2])
    parent, rank = {}, {}
    for v in graph.vertices:
        parent[v] = v
        rank[v] = 0

    mst = []
    for u, v, w in edges:
        x, y = find(parent, u), find(parent, v)
        if x != y:
            mst.append((u, v, w))
            union(parent, rank, x, y)
    return mst


# =============================
# TOPOLOGICAL SORT + CYCLE DETECTION
# =============================
def topological_sort(graph):
    visited = set()
    stack = []
    cycle = [False]

    def dfs(v, path):
        visited.add(v)
        path.add(v)
        for nbr in graph.graph.neighbors(v):
            if nbr not in visited:
                dfs(nbr, path)
            elif nbr in path:
                cycle[0] = True
        path.remove(v)
        stack.append(v)

    for v in graph.vertices:
        if v not in visited:
            dfs(v, set())

    stack.reverse()
    return stack, cycle[0]


# =============================
# DRAW GRAPH
# =============================
def draw_graph(graph, title, filename, highlight_edges=None):
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(graph.graph, seed=42)
    nx.draw(
        graph.graph, pos, with_labels=True, node_color="lightblue",
        node_size=900, font_weight="bold"
    )
    labels = nx.get_edge_attributes(graph.graph, "weight")
    nx.draw_networkx_edge_labels(graph.graph, pos, edge_labels=labels)

    if highlight_edges:
        nx.draw_networkx_edges(graph.graph, pos, edgelist=highlight_edges, width=3, edge_color="r")

    plt.title(title)
    plt.savefig(filename)
    plt.close()


# =============================
# GRAPH GENERATORS
# =============================
def generate_random_graph(num_nodes, num_edges, directed=False):
    vertices = [f"V{i}" for i in range(1, num_nodes + 1)]
    edges = set()
    while len(edges) < num_edges:
        u, v = random.sample(vertices, 2)
        if u != v:
            w = random.randint(1, 20)
            edges.add((u, v, w))
            if not directed:
                edges.add((v, u, w))  # ensure symmetry for undirected
    edges = list(edges)[:num_edges]
    gtype = "directed" if directed else "undirected"
    return Graph(vertices, edges, gtype)


# =============================
# MAIN EXECUTION
# =============================
def run_all():
    report_lines = []

    # Problem 1: Undirected graph (21 nodes, 40 edges)
    g1 = generate_random_graph(21, 40, directed=False)
    mst = kruskal_mst(g1)
    source = g1.vertices[0]
    sp = dijkstra(g1, source)
    draw_graph(g1, "Problem 1: Undirected Graph (MST)", "problem1_mst.png", mst)
    report_lines.append("=== Problem 1: Undirected Graph ===")
    report_lines.append(f"Nodes: {len(g1.vertices)}, Edges: {len(g1.edges)}")
    report_lines.append(f"Shortest Paths from {source}: {sp}")
    report_lines.append(f"MST Edges: {mst}\n")

    # Problem 2: Directed graph (19 nodes, 37 edges)
    g2 = generate_random_graph(19, 37, directed=True)
    topo, has_cycle = topological_sort(g2)
    source = g2.vertices[0]
    sp = dijkstra(g2, source)
    draw_graph(g2, "Problem 2: Directed Graph (Topo Sort)", "problem2_topo.png")
    report_lines.append("=== Problem 2: Directed Graph ===")
    report_lines.append(f"Nodes: {len(g2.vertices)}, Edges: {len(g2.edges)}")
    report_lines.append(f"Shortest Paths from {source}: {sp}")
    report_lines.append(f"Topological Order: {topo}")
    report_lines.append(f"Contains Cycle: {has_cycle}\n")

    # Problem 3: Directed graph (19 nodes, 37 edges)
    g3 = generate_random_graph(19, 37, directed=True)
    topo, has_cycle = topological_sort(g3)
    source = g3.vertices[0]
    sp = dijkstra(g3, source)
    draw_graph(g3, "Problem 3: Directed Graph (Cycle Detection)", "problem3_cycle.png")
    report_lines.append("=== Problem 3: Directed Graph ===")
    report_lines.append(f"Nodes: {len(g3.vertices)}, Edges: {len(g3.edges)}")
    report_lines.append(f"Shortest Paths from {source}: {sp}")
    report_lines.append(f"Topological Order: {topo}")
    report_lines.append(f"Contains Cycle: {has_cycle}\n")

    with open("report.txt", "w") as f:
        f.write("\n".join(report_lines))

    print("\n✅ All problems executed successfully!")
    print("PNG files: problem1_mst.png, problem2_topo.png, problem3_cycle.png")
    print("Text report: report.txt")


# =============================
# ENTRY POINT
# =============================
if __name__ == "__main__":
    random.seed(42)  # reproducible results
    run_all()
