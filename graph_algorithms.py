import heapq
import networkx as nx
import matplotlib.pyplot as plt
import os
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

def read_graph_from_file(filename):
    with open(filename, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    first = lines[0].split()
    n, e, gtype = int(first[0]), int(first[1]), first[2]
    edges = []
    for line in lines[1:-1]:
        u, v, w = line.split()
        edges.append((u, v, int(w)))
    source = lines[-1]
    vertices = [f"V{i}" for i in range(1, n + 1)]
    gtype = "directed" if gtype.upper() == "D" else "undirected"
    return Graph(vertices, edges, gtype), source


def dijkstra(graph, source):
    G = graph.to_networkx()
    dist = {v: float("inf") for v in G.nodes}
    dist[source] = 0
    path = {v: [] for v in G.nodes}
    path[source] = [source]

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

def kruskal_mst(graph):
    if graph.type.lower() != "undirected":
        raise ValueError("Kruskal's algorithm requires an undirected graph.")
    G_nx = nx.Graph()
    for u, v, w in graph.edges:
        G_nx.add_edge(u, v, weight=w)
    if not nx.is_connected(G_nx):
        raise ValueError("Graph is disconnected. MST cannot span all vertices.")

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


def draw_graph(graph, title, filename, highlight_edges=None):
    G = graph.to_networkx()
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(9, 7))
    nx.draw(G, pos, with_labels=True, node_color="skyblue", node_size=1100, font_size=12)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    if highlight_edges:
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v, _ in highlight_edges], width=5, edge_color="black")
    plt.title(title)
    plt.savefig(filename)
    plt.close()


def run_all_iterations():
    run_folders = ["RUN1", "RUN2", "RUN3", "RUN4"]

    for run in run_folders:
        print(f"\n--- Processing {run} ---")
        report = []

        dir_graph_path = os.path.join(run, "directed_graph.txt")
        undir_graph_path = os.path.join(run, "undirected_graph.txt")

        # ---------- Problem 1 ----------
        report.append(f"=== {run}: Problem 1 - Dijkstra’s Shortest Path ===")
        g1, source1 = read_graph_from_file(dir_graph_path)
        dist1, path1 = dijkstra(g1, source1)
        report.append("\nDirected Graph Shortest Paths:")
        for node, cost in dist1.items():
            report.append(f"{node}: cost = {cost}, path = {path1[node]}")
        draw_graph(g1, f"{run} - Directed Shortest Paths", os.path.join(run, f"{run}_problem1_directed.png"))

        g2, source2 = read_graph_from_file(undir_graph_path)
        dist2, path2 = dijkstra(g2, source2)
        report.append("\nUndirected Graph Shortest Paths:")
        for node, cost in dist2.items():
            report.append(f"{node}: cost = {cost}, path = {path2[node]}")
        draw_graph(g2, f"{run} - Undirected Shortest Paths", os.path.join(run, f"{run}_problem1_undirected.png"))

        # ---------- Problem 2 ----------
        report.append(f"\n=== {run}: Problem 2 - Minimum Spanning Tree (Kruskal) ===")
        mst, total_cost = kruskal_mst(g2)
        for e in mst:
            report.append(f"Edge: {e[0]} - {e[1]} | Weight: {e[2]}")
        report.append(f"Total Cost of MST: {total_cost}")
        draw_graph(g2, f"{run} - MST", os.path.join(run, f"{run}_problem2_mst.png"), mst)

        # ---------- Problem 3 ----------
        report.append(f"\n=== {run}: Problem 3 - DFS, Topological Sort & Cycles ===")
        topo, cycles = dfs_topological_and_cycles(g1)
        if topo:
            report.append("Graph is Acyclic.\nTopological Order:")
            report.append(" -> ".join(topo))
            draw_graph(g1, f"{run} - Topological Sort", os.path.join(run, f"{run}_problem3_topo.png"))
        else:
            report.append("Graph has Cycles:")
            for c in cycles:
                report.append(f"Cycle: {' -> '.join(c)} (length {len(c)-1})")
            draw_graph(g1, f"{run} - Cycles", os.path.join(run, f"{run}_problem3_cycles.png"))

        report_path = os.path.join(run, f"{run}_graph_report.txt")
        with open(report_path, "w") as f:
            f.write("\n".join(report))

if __name__ == "__main__":
    run_all_iterations()
