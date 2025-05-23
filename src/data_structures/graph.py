from typing import Dict, List, Set, Tuple, Optional
import heapq
from dataclasses import dataclass
from datetime import datetime, time

@dataclass
class Node:
    """Represents a location in the transportation network."""
    id: str  # Changed from int to str to support facility IDs
    name: str
    latitude: float
    longitude: float
    node_type: str  # 'intersection', 'bus_stop', 'train_station', etc.
    is_critical: bool = False  # For critical facilities like hospitals

@dataclass
class Edge:
    """Represents a road or connection between nodes."""
    source: str  # Changed from int to str
    target: str  # Changed from int to str
    distance: float  # in meters
    base_time: float  # base travel time in minutes
    capacity: int  # maximum vehicles per hour
    road_type: str  # 'highway', 'arterial', 'local', etc.
    lanes: int
    is_one_way: bool = False

@dataclass
class TimeDependentEdge(Edge):
    """Edge with time-dependent travel times."""
    time_variations: Dict[Tuple[time, time], float] = None

    def __post_init__(self):
        if self.time_variations is None:
            self.time_variations = {}

class TransportationGraph:
    """Graph representation of the transportation network."""
    def __init__(self):
        self.nodes: Dict[str, Node] = {}  # Changed from int to str
        self.edges: Dict[str, Edge] = {}
        self.adjacency_list: Dict[str, List[Tuple[str, Edge]]] = {}  # Changed from int to str
        self.reverse_adjacency_list: Dict[str, List[Tuple[str, Edge]]] = {}  # Changed from int to str

    def add_node(self, node: Node) -> None:
        """Add a node to the graph."""
        # Convert integer IDs to strings
        node.id = str(node.id)
        self.nodes[node.id] = node
        if node.id not in self.adjacency_list:
            self.adjacency_list[node.id] = []
        if node.id not in self.reverse_adjacency_list:
            self.reverse_adjacency_list[node.id] = []

    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the graph."""
        # Convert integer IDs to strings
        edge.source = str(edge.source)
        edge.target = str(edge.target)
        edge_id = f"{edge.source}-{edge.target}"
        self.edges[edge_id] = edge
        
        # Initialize adjacency list for source node if it doesn't exist
        if edge.source not in self.adjacency_list:
            self.adjacency_list[edge.source] = []
        
        # Initialize adjacency list for target node if it doesn't exist
        if edge.target not in self.adjacency_list:
            self.adjacency_list[edge.target] = []
        
        # Add edge to adjacency list
        self.adjacency_list[edge.source].append((edge.target, edge))
        if not edge.is_one_way:
            self.adjacency_list[edge.target].append((edge.source, edge))

    def get_neighbors(self, node_id: str) -> List[Tuple[str, Edge]]:  # Changed from int to str
        """Get all neighbors of a node."""
        node_id = str(node_id)  # Convert integer IDs to strings
        return self.adjacency_list.get(node_id, [])

    def get_edge(self, source: str, target: str) -> Optional[Edge]:  # Changed from int to str
        """Get the edge between two nodes if it exists."""
        source = str(source)  # Convert integer IDs to strings
        target = str(target)  # Convert integer IDs to strings
        edge_id = f"{source}-{target}"
        edge = self.edges.get(edge_id)
        if edge is None:
            # Check the reverse direction for undirected edges
            reverse_edge_id = f"{target}-{source}"
            edge = self.edges.get(reverse_edge_id)
            if edge and not edge.is_one_way:
                return edge
        return edge

    def get_time_dependent_weight(self, edge: Edge, current_time: datetime) -> float:
        """Calculate the time-dependent weight of an edge."""
        if not isinstance(edge, TimeDependentEdge) or not edge.time_variations:
            return edge.base_time

        current_time = current_time.time()
        for (start_time, end_time), factor in edge.time_variations.items():
            if start_time <= current_time <= end_time:
                return edge.base_time * (1 + factor)
        return edge.base_time

    def get_all_critical_nodes(self) -> List[Node]:
        """Get all nodes marked as critical facilities."""
        return [node for node in self.nodes.values() if node.is_critical]

    def get_connected_components(self) -> List[Set[str]]:  # Changed return type to Set[str]
        """Get all connected components in the graph using DFS."""
        visited = set()
        components = []

        def dfs(node_id: str, component: Set[str]) -> None:  # Changed parameter types to str
            visited.add(node_id)
            component.add(node_id)
            for neighbor, _ in self.get_neighbors(node_id):
                if neighbor not in visited:
                    dfs(neighbor, component)

        for node_id in self.nodes:
            if node_id not in visited:
                component = set()
                dfs(node_id, component)
                components.append(component)

        return components 