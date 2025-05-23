from typing import Dict, List, Tuple, Optional
import heapq
from datetime import datetime
from src.data_structures.graph import TransportationGraph, Edge, Node

class ShortestPathResult:
    """Class to store the result of a shortest path calculation."""
    def __init__(self):
        self.distances: Dict[str, float] = {}
        self.previous: Dict[str, Optional[str]] = {}
        self.path: List[str] = []
        self.total_distance: float = 0.0
        self.total_time: float = 0.0

def dijkstra(
    graph: TransportationGraph,
    start: str,
    end: str,
    current_time: datetime,
    use_time_dependent: bool = True
) -> ShortestPathResult:
    """
    Implements Dijkstra's algorithm with time-dependent weights.
    
    Args:
        graph: The transportation graph
        start: Starting node ID
        end: Destination node ID
        current_time: Current time for time-dependent weights
        use_time_dependent: Whether to use time-dependent weights
        
    Returns:
        ShortestPathResult containing the path and metrics
    """
    result = ShortestPathResult()
    
    # Convert IDs to strings
    start = str(start)
    end = str(end)
    
    # Initialize distances
    for node_id in graph.nodes:
        result.distances[node_id] = float('inf')
        result.previous[node_id] = None
    
    result.distances[start] = 0
    
    # Priority queue for Dijkstra's algorithm
    pq = [(0, start)]
    
    while pq:
        current_dist, current_node = heapq.heappop(pq)
        
        # If we've reached the destination, we can stop
        if current_node == end:
            break
            
        # If we've found a better path already, skip
        if current_dist > result.distances[current_node]:
            continue
            
        # Check all neighbors
        for neighbor, edge in graph.get_neighbors(current_node):
            # Calculate edge weight
            if use_time_dependent:
                weight = graph.get_time_dependent_weight(edge, current_time)
            else:
                weight = edge.base_time
                
            # Calculate new distance
            new_dist = result.distances[current_node] + weight
            
            # If we found a better path, update it
            if new_dist < result.distances[neighbor]:
                result.distances[neighbor] = new_dist
                result.previous[neighbor] = current_node
                heapq.heappush(pq, (new_dist, neighbor))
    
    # Reconstruct the path
    if result.distances[end] != float('inf'):
        current = end
        while current is not None:
            result.path.append(current)
            current = result.previous[current]
        result.path.reverse()
        
        # Calculate total distance and time
        for i in range(len(result.path) - 1):
            edge = graph.get_edge(result.path[i], result.path[i + 1])
            if edge:
                result.total_distance += edge.distance
                if use_time_dependent:
                    result.total_time += graph.get_time_dependent_weight(edge, current_time)
                else:
                    result.total_time += edge.base_time
    
    return result

def a_star(
    graph: TransportationGraph,
    start: str,
    end: str,
    current_time: datetime,
    use_time_dependent: bool = True
) -> ShortestPathResult:
    """
    Implements A* algorithm with time-dependent weights.
    
    Args:
        graph: The transportation graph
        start: Starting node ID
        end: Destination node ID
        current_time: Current time for time-dependent weights
        use_time_dependent: Whether to use time-dependent weights
        
    Returns:
        ShortestPathResult containing the path and metrics
    """
    result = ShortestPathResult()
    
    # Convert IDs to strings
    start = str(start)
    end = str(end)
    
    # Initialize distances
    for node_id in graph.nodes:
        result.distances[node_id] = float('inf')
        result.previous[node_id] = None
    
    result.distances[start] = 0
    
    # Priority queue for A* algorithm
    pq = [(0, start)]
    
    # Heuristic function (Euclidean distance)
    def heuristic(node1: str, node2: str) -> float:
        n1 = graph.nodes[node1]
        n2 = graph.nodes[node2]
        return ((n1.latitude - n2.latitude) ** 2 + 
                (n1.longitude - n2.longitude) ** 2) ** 0.5
    
    while pq:
        current_f, current_node = heapq.heappop(pq)
        
        # If we've reached the destination, we can stop
        if current_node == end:
            break
            
        # If we've found a better path already, skip
        if current_f > result.distances[current_node] + heuristic(current_node, end):
            continue
            
        # Check all neighbors
        for neighbor, edge in graph.get_neighbors(current_node):
            # Calculate edge weight
            if use_time_dependent:
                weight = graph.get_time_dependent_weight(edge, current_time)
            else:
                weight = edge.base_time
                
            # Calculate new distance
            new_dist = result.distances[current_node] + weight
            
            # If we found a better path, update it
            if new_dist < result.distances[neighbor]:
                result.distances[neighbor] = new_dist
                result.previous[neighbor] = current_node
                f_score = new_dist + heuristic(neighbor, end)
                heapq.heappush(pq, (f_score, neighbor))
    
    # Reconstruct the path
    if result.distances[end] != float('inf'):
        current = end
        while current is not None:
            result.path.append(current)
            current = result.previous[current]
        result.path.reverse()
        
        # Calculate total distance and time
        for i in range(len(result.path) - 1):
            edge = graph.get_edge(result.path[i], result.path[i + 1])
            if edge:
                result.total_distance += edge.distance
                if use_time_dependent:
                    result.total_time += graph.get_time_dependent_weight(edge, current_time)
                else:
                    result.total_time += edge.base_time
    
    return result 