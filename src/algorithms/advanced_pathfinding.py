from typing import Dict, List, Tuple, Optional, Set
import heapq
from datetime import datetime
from src.data_structures.graph import TransportationGraph, Edge, Node
from src.algorithms.shortest_path import ShortestPathResult

def greedy_shortest_path(
    graph: TransportationGraph,
    start: str,
    end: str,
    current_time: datetime,
    use_time_dependent: bool = True
) -> ShortestPathResult:
    """
    Implements a greedy best-first search algorithm that always expands the most promising node according to the greedy score.
    This is more robust than the classic greedy approach and can find a path if one exists.
    """
    result = ShortestPathResult()
    
    # Ensure node IDs are strings
    start = str(start)
    end = str(end)
    graph.nodes = {str(k): v for k, v in graph.nodes.items()}
    
    # Priority queue: (score, node)
    pq = [(0, start)]
    visited = set()
    result.distances = {start: 0}
    result.previous = {start: None}
    
    while pq:
        current_score, current = heapq.heappop(pq)
        if current in visited:
            continue
        visited.add(current)
        if current == end:
            break
        for neighbor, edge in graph.get_neighbors(current):
            if neighbor in visited:
                continue
            # Calculate the greedy score
            distance_factor = edge.distance / 1000  # km
            capacity_factor = 5000 / (edge.capacity + 1)
            if use_time_dependent:
                time_factor = graph.get_time_dependent_weight(edge, current_time)
            else:
                time_factor = edge.base_time
            score = distance_factor + capacity_factor + time_factor
            new_score = current_score + score
            if neighbor not in result.distances or new_score < result.distances[neighbor]:
                result.distances[neighbor] = new_score
                result.previous[neighbor] = current
                heapq.heappush(pq, (new_score, neighbor))
    
    # Reconstruct the path
    if end in result.previous:
        path = []
        current = end
        while current is not None:
            path.append(current)
            current = result.previous[current]
        result.path = path[::-1]
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

class DPShortestPath:
    """Dynamic Programming solution for finding shortest paths with multiple criteria."""
    
    def __init__(self, graph: TransportationGraph):
        self.graph = graph
        self.memo: Dict[Tuple[str, str, int], ShortestPathResult] = {}
        self.max_depth = 20  # Prevent infinite recursion
    
    def find_path(
        self,
        start: str,
        end: str,
        current_time: datetime,
        max_stops: int = 10,
        use_time_dependent: bool = True
    ) -> ShortestPathResult:
        """
        Find shortest path using dynamic programming approach.
        This method considers:
        1. Number of stops
        2. Time-dependent weights
        3. Road capacity constraints
        4. Multiple possible paths
        
        Args:
            start: Starting node ID
            end: Destination node ID
            current_time: Current time for weights
            max_stops: Maximum number of intermediate stops
            use_time_dependent: Whether to use time-dependent weights
        """
        # Convert IDs to strings
        start = str(start)
        end = str(end)
        
        def dp(current: str, target: str, stops_left: int) -> ShortestPathResult:
            # Base cases
            if stops_left < 0:
                result = ShortestPathResult()
                return result
                
            if current == target:
                result = ShortestPathResult()
                result.path = [current]
                return result
                
            # Check memoization
            key = (current, target, stops_left)
            if key in self.memo:
                return self.memo[key]
            
            best_result = ShortestPathResult()
            best_result.distances[target] = float('inf')
            
            # Try all possible next steps
            for neighbor, edge in self.graph.get_neighbors(current):
                # Calculate the weight for this edge
                if use_time_dependent:
                    weight = self.graph.get_time_dependent_weight(edge, current_time)
                else:
                    weight = edge.base_time
                
                # Recursively find the best path from neighbor
                sub_result = dp(neighbor, target, stops_left - 1)
                
                # If we found a valid path and it's better than what we had
                if sub_result.path and (
                    best_result.distances[target] == float('inf') or
                    weight + sub_result.total_time < best_result.total_time
                ):
                    best_result = ShortestPathResult()
                    best_result.path = [current] + sub_result.path
                    best_result.total_time = weight + sub_result.total_time
                    best_result.total_distance = edge.distance + sub_result.total_distance
                    best_result.distances[target] = best_result.total_time
                    
            # Memoize and return
            self.memo[key] = best_result
            return best_result
        
        # Start the dynamic programming solution
        return dp(start, end, max_stops)

def multi_criteria_path(
    graph: TransportationGraph,
    start: str,
    end: str,
    current_time: datetime,
    weight_distance: float = 0.4,
    weight_time: float = 0.3,
    weight_capacity: float = 0.3
) -> ShortestPathResult:
    """
    Find path considering multiple criteria using a combination of greedy and DP approaches.
    
    Args:
        graph: The transportation graph
        start: Starting node ID
        end: Destination node ID
        current_time: Current time
        weight_distance: Weight for distance factor (0-1)
        weight_time: Weight for time factor (0-1)
        weight_capacity: Weight for road capacity factor (0-1)
    """
    result = ShortestPathResult()
    
    # Convert IDs to strings
    start = str(start)
    end = str(end)
    
    # Initialize distances
    for node_id in graph.nodes:
        result.distances[node_id] = float('inf')
    result.distances[start] = 0
    
    # Priority queue with weighted criteria
    pq = [(0, start)]
    
    while pq:
        current_score, current = heapq.heappop(pq)
        
        if current == end:
            break
            
        for neighbor, edge in graph.get_neighbors(current):
            # Calculate individual factors
            distance_factor = edge.distance / 1000  # km
            time_factor = graph.get_time_dependent_weight(edge, current_time)
            capacity_factor = 5000 / (edge.capacity + 1)
            
            # Weighted score
            score = (
                weight_distance * distance_factor +
                weight_time * time_factor +
                weight_capacity * capacity_factor
            )
            
            new_score = current_score + score
            
            if new_score < result.distances[neighbor]:
                result.distances[neighbor] = new_score
                result.previous[neighbor] = current
                heapq.heappush(pq, (new_score, neighbor))
    
    # Reconstruct path
    if result.distances[end] != float('inf'):
        current = end
        while current is not None:
            result.path.append(current)
            if current in result.previous:
                current = result.previous[current]
            else:
                break
        result.path.reverse()
        
        # Calculate actual distance and time
        for i in range(len(result.path) - 1):
            edge = graph.get_edge(result.path[i], result.path[i + 1])
            if edge:
                result.total_distance += edge.distance
                result.total_time += graph.get_time_dependent_weight(edge, current_time)
    
    return result 