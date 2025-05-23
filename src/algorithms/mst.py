from typing import List, Set, Tuple
from dataclasses import dataclass
from src.data_structures.graph import TransportationGraph, Edge, Node
import folium
from streamlit_folium import folium_static

class DisjointSet:
    """Disjoint Set data structure for Kruskal's algorithm."""
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return

        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

def kruskal_mst(graph: TransportationGraph, critical_nodes: Set[str] = None) -> List[Edge]:
    """
    Implements Kruskal's algorithm to find the Minimum Spanning Tree.
    
    Args:
        graph: The transportation graph
        critical_nodes: Set of node IDs (as strings) that must be connected in the MST
        
    Returns:
        List of edges forming the MST
    """
    if critical_nodes is None:
        critical_nodes = set()

    # Map node IDs (str) to integer indices for DisjointSet
    node_id_to_idx = {node_id: idx for idx, node_id in enumerate(graph.nodes.keys())}
    idx_to_node_id = {idx: node_id for node_id, idx in node_id_to_idx.items()}

    # Create list of all edges
    edges = list(graph.edges.values())
    edges.sort(key=lambda e: e.distance)  # Sort edges by distance

    # Initialize disjoint set
    ds = DisjointSet(len(graph.nodes))
    
    # Initialize MST
    mst_edges = []
    connected_nodes = set()

    # First, ensure critical nodes are connected
    for edge in edges:
        src_idx = node_id_to_idx[edge.source]
        tgt_idx = node_id_to_idx[edge.target]
        if edge.source in critical_nodes or edge.target in critical_nodes:
            if ds.find(src_idx) != ds.find(tgt_idx):
                ds.union(src_idx, tgt_idx)
                mst_edges.append(edge)
                connected_nodes.add(edge.source)
                connected_nodes.add(edge.target)

    # Then, add remaining edges
    for edge in edges:
        src_idx = node_id_to_idx[edge.source]
        tgt_idx = node_id_to_idx[edge.target]
        if ds.find(src_idx) != ds.find(tgt_idx):
            ds.union(src_idx, tgt_idx)
            mst_edges.append(edge)
            connected_nodes.add(edge.source)
            connected_nodes.add(edge.target)

    # Verify all critical nodes are connected
    if critical_nodes and not all(node in connected_nodes for node in critical_nodes):
        raise ValueError("Not all critical nodes can be connected in the MST")

    return mst_edges

def calculate_mst_cost(mst_edges: List[Edge]) -> float:
    """Calculate the total cost of the MST."""
    return sum(edge.distance for edge in mst_edges)

def verify_mst_connectivity(mst_edges: List[Edge], graph: TransportationGraph) -> bool:
    """Verify that the MST connects all nodes in the graph."""
    if not mst_edges:
        return False

    # Create adjacency list for MST
    mst_adj = {node_id: [] for node_id in graph.nodes}
    for edge in mst_edges:
        mst_adj[edge.source].append(edge.target)
        mst_adj[edge.target].append(edge.source)

    # Check connectivity using DFS
    visited = set()
    
    def dfs(node_id: int) -> None:
        visited.add(node_id)
        for neighbor in mst_adj[node_id]:
            if neighbor not in visited:
                dfs(neighbor)

    # Start DFS from any node
    start_node = next(iter(graph.nodes))
    dfs(start_node)

    # Check if all nodes were visited
    return len(visited) == len(graph.nodes)

def create_map(graph: TransportationGraph) -> folium.Map:
    """Create a map of the transportation graph."""
    m = folium.Map(location=[graph.nodes[next(iter(graph.nodes))].latitude, graph.nodes[next(iter(graph.nodes))].longitude], zoom_start=12)
    for node_id, node in graph.nodes.items():
        folium.CircleMarker(
            location=[node.latitude, node.longitude],
            radius=5,
            color='blue',
            fill=True,
            fill_color='blue',
            fill_opacity=0.7,
            popup=node_id
        ).add_to(m)
    return m

def visualize_mst(graph: TransportationGraph, mst_edges: List[Edge]):
    """Visualize the MST on a map."""
    mst_graph = TransportationGraph()
    mst_graph.nodes = graph.nodes.copy()
    mst_graph.edges = {edge.source + '-' + edge.target: edge for edge in mst_edges}

    m = create_map(graph)
    # Draw MST edges in green
    for u, v in mst_graph.edges.items():
        u, v = u.split('-')
        folium.PolyLine(
            locations=[
                [graph.nodes[u].latitude, graph.nodes[u].longitude],
                [graph.nodes[v].latitude, graph.nodes[v].longitude]
            ],
            color='green',
            weight=4,
            dash_array='5, 10'
        ).add_to(m)
    folium_static(m) 