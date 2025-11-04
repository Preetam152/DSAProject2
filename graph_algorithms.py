import heapq
import networkx as nx
import matplotlib.pyplot as plt

# =========================
# Basic Graph Structure
# =========================
class Graph:
    def __init__(self, vertices, edges, gtype="undirected"):
        self.vertices = vertices
        self.edges = edges
        self.type = gtype

    def to_networkx(self):
        G = nx.DiGraph() if self.type == "directed" else nx.Graph()
        for u, v, w in self.edges:
            G.add_edge(u, v, weight=w)
        return G

# =========================
# Problem 1: Dijkstra’s Algorithm (with heapq)
# =========================
def dijkstra(graph, source):
    G = graph.to_networkx()

    # Initialize distances and paths
    dist = {v: float("inf") for v in G.nodes}
    dist[source] = 0
    path = {v: [] for v in G.nodes}
    path[source] = [source]

    # Priority queue: (distance, node)
    pq = [(0, source)]
    while pq:
        current_dist, u = heapq.heappop(pq)
        if current_dist > dist[u]:
            continue
        for v in G.neighbors(u):
            weight = G[u][v]['weight']
            if dist[u] + weight < dist[v]:
                dist[v] = dist[u] + weight
                path[v] = path[u] + [v]
                heapq.heappush(pq, (dist[v], v))
    return dist, path

# =========================
# Problem 2: Kruskal’s MST
# =========================
def kruskal_mst(graph):
    parent, rank = {}, {}

    def find(v):
        if parent[v] != v:
            parent[v] = find(parent[v])
        return parent[v]

    def union(u, v):
        ru, rv = find(u), find(v)
        if ru != rv:
            if rank[ru] < rank[rv]:
                parent[ru] = rv
            elif rank[ru] > rank[rv]:
                parent[rv] = ru
            else:
                parent[rv] = ru
                rank[ru] += 1

    for v in graph.vertices:
        parent[v] = v
        rank[v] = 0

    mst, total_cost = [], 0
    for u, v, w in sorted(graph.edges, key=lambda e: e[2]):
        if find(u) != find(v):
            union(u, v)
            mst.append((u, v, w))
            total_cost += w

    return mst, total_cost

# =========================
# Problem 3: DFS, Topo Sort, Cycle Detection
# =========================
def dfs_topological_and_cycles(graph):
    G = graph.to_networkx()
    visited, stack, topo, cycles = set(), set(), [], []

    def dfs(v, path):
        visited.add(v)
        stack.add(v)
        path.append(v)
        for n in G.neighbors(v):
            if n not in visited:
                dfs(n, path)
            elif n in stack:
                i = path.index(n)
                cycles.append(path[i:] + [n])
        stack.remove(v)
        path.pop()
        topo.append(v)

    for v in G.nodes:
        if v not in visited:
            dfs(v, [])

    if cycles:
        return None, cycles
    topo.reverse()
    return topo, None

# =========================
# Visualization
# =========================
def draw_graph(graph, title, filename, highlight_edges=None):
    G = graph.to_networkx()
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(9, 7))
    nx.draw(G, pos, with_labels=True, node_color="skyblue", node_size=1100, font_size=12)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    if highlight_edges:
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(u, v) for u, v, _ in highlight_edges],
            width=5, edge_color="black"
        )

    plt.title(title)
    plt.savefig(filename)
    plt.close()

# =========================
# Fixed Example Graph Data
# =========================
def get_directed_graph():
    vertices = [f"V{i}" for i in range(1, 21)]
    edges = [
        ("V1", "V2", 4), ("V1", "V3", 2), ("V2", "V4", 3), ("V3", "V4", 1),
        ("V4", "V5", 2), ("V5", "V6", 5), ("V2", "V6", 7), ("V3", "V7", 3),
        ("V7", "V8", 2), ("V8", "V9", 4), ("V9", "V10", 6), ("V10", "V11", 2),
        ("V11", "V12", 3), ("V6", "V12", 1), ("V12", "V13", 2), ("V13", "V14", 3),
        ("V14", "V15", 2), ("V15", "V16", 4), ("V16", "V17", 1), ("V17", "V18", 3),
        ("V18", "V19", 5), ("V19", "V20", 2), ("V8", "V11", 4), ("V9", "V13", 2),
        ("V5", "V9", 3), ("V7", "V10", 4), ("V14", "V19", 3), ("V2", "V8", 5),
        ("V6", "V10", 3), ("V10", "V15", 2), ("V12", "V17", 4), ("V17", "V20", 5),
        ("V11", "V16", 6), ("V15", "V18", 2), ("V3", "V9", 7), ("V4", "V8", 4),
        ("V5", "V7", 3), ("V13", "V18", 2)
    ]
    return Graph(vertices, edges, "directed")

def get_undirected_graph():
    vertices = [f"V{i}" for i in range(1, 22)]
    edges = [
        ("V1", "V2", 2), ("V2", "V3", 3), ("V3", "V4", 4), ("V4", "V5", 5),
        ("V5", "V6", 2), ("V6", "V7", 3), ("V7", "V8", 4), ("V8", "V9", 5),
        ("V9", "V10", 6), ("V10", "V11", 3), ("V11", "V12", 4), ("V12", "V13", 2),
        ("V13", "V14", 5), ("V14", "V15", 3), ("V15", "V16", 4), ("V16", "V17", 2),
        ("V17", "V18", 3), ("V18", "V19", 4), ("V19", "V20", 5), ("V20", "V21", 2),
        ("V1", "V5", 3), ("V2", "V6", 2), ("V3", "V7", 5), ("V4", "V8", 3),
        ("V5", "V9", 4), ("V6", "V10", 3), ("V7", "V11", 2), ("V8", "V12", 5),
        ("V9", "V13", 4), ("V10", "V14", 3), ("V11", "V15", 2), ("V12", "V16", 3),
        ("V13", "V17", 4), ("V14", "V18", 5), ("V15", "V19", 2), ("V16", "V20", 3),
        ("V17", "V21", 4), ("V1", "V10", 6), ("V5", "V15", 3), ("V9", "V19", 2)
    ]
    return Graph(vertices, edges, "undirected")

# =========================
# Main Runner
# =========================
def run_all():
    report = []

    # ---------- Problem 1 ----------
    report.append("=== Problem 1: Dijkstra’s Shortest Path ===")

    # Directed Graph
    g1 = get_directed_graph()
    dist1, path1 = dijkstra(g1, "V1")
    report.append("\nDirected Graph Shortest Paths (source: V1):")
    for node, cost in dist1.items():
        report.append(f"{node}: cost = {cost}, path = {path1[node]}")
    draw_graph(g1, "Problem 1 - Directed Shortest Paths", "problem1_directed.png")

    # Undirected Graph
    g2 = get_undirected_graph()
    dist2, path2 = dijkstra(g2, "V1")
    report.append("\nUndirected Graph Shortest Paths (source: V1):")
    for node, cost in dist2.items():
        report.append(f"{node}: cost = {cost}, path = {path2[node]}")
    draw_graph(g2, "Problem 1 - Undirected Shortest Paths", "problem1_undirected.png")

    # ---------- Problem 2 ----------
    report.append("\n=== Problem 2: Minimum Spanning Tree (Kruskal) ===")
    mst, total_cost = kruskal_mst(g2)
    for e in mst:
        report.append(f"Edge: {e[0]} - {e[1]} | Weight: {e[2]}")
    report.append(f"Total Cost of MST: {total_cost}")
    draw_graph(g2, "Problem 2 - MST", "problem2_mst.png", mst)

    # ---------- Problem 3 ----------
    report.append("\n=== Problem 3: DFS, Topological Sort & Cycles ===")
    g3 = get_directed_graph()
    topo, cycles = dfs_topological_and_cycles(g3)
    if topo:
        report.append("Graph is Acyclic.\nTopological Order:")
        report.append(" -> ".join(topo))
        draw_graph(g3, "Problem 3 - Topological Sort", "problem3_topo.png")
    else:
        report.append("Graph has Cycles:")
        for c in cycles:
            report.append(f"Cycle: {' -> '.join(c)} (length {len(c)-1})")
        draw_graph(g3, "Problem 3 - Cycles", "problem3_cycles.png")

    # Save report
    with open("graph_report.txt", "w") as f:
        f.write("\n".join(report))

# =========================
# Execute
# =========================
if __name__ == "__main__":
    run_all()
