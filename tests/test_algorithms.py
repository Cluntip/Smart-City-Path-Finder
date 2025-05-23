import pytest
from datetime import datetime, time
from src.data_structures.graph import TransportationGraph, Node, Edge, TimeDependentEdge
from src.algorithms.mst import kruskal_mst, calculate_mst_cost, verify_mst_connectivity
from src.algorithms.shortest_path import dijkstra, a_star

@pytest.fixture
def sample_graph():
    """Create a sample graph for testing."""
    graph = TransportationGraph()
    
    # Add nodes
    nodes = [
        Node(1, "A", 0, 0, "intersection"),
        Node(2, "B", 1, 1, "intersection"),
        Node(3, "C", 2, 0, "intersection"),
        Node(4, "D", 1, -1, "intersection"),
        Node(5, "E", 0, -2, "intersection", True),  # Critical node
    ]
    
    for node in nodes:
        graph.add_node(node)
    
    # Add edges
    edges = [
        Edge(1, 2, 100, 10, 1000, "local", 2),
        Edge(2, 3, 150, 15, 1200, "arterial", 2),
        Edge(3, 4, 200, 20, 1500, "highway", 4),
        Edge(4, 5, 100, 10, 1000, "local", 2),
        Edge(1, 4, 300, 30, 2000, "highway", 4),
        Edge(2, 4, 250, 25, 1800, "arterial", 3),
    ]
    
    # Add time-dependent edge
    rush_hour_edge = TimeDependentEdge(1, 3, 400, 40, 2500, "highway", 4)
    rush_hour_edge.time_variations = {
        (time(7, 0), time(9, 0)): 2.0,    # Morning rush hour
        (time(16, 0), time(18, 0)): 2.0,  # Evening rush hour
    }
    edges.append(rush_hour_edge)
    
    for edge in edges:
        graph.add_edge(edge)
    
    return graph

def test_kruskal_mst(sample_graph):
    """Test Kruskal's algorithm for MST."""
    # Test MST without critical nodes
    mst_edges = kruskal_mst(sample_graph)
    assert len(mst_edges) == len(sample_graph.nodes) - 1
    assert verify_mst_connectivity(mst_edges, sample_graph)
    
    # Test MST with critical node
    critical_nodes = {5}  # Node E is critical
    mst_edges = kruskal_mst(sample_graph, critical_nodes)
    assert len(mst_edges) == len(sample_graph.nodes) - 1
    assert verify_mst_connectivity(mst_edges, sample_graph)
    
    # Verify critical node is connected
    connected_nodes = set()
    for edge in mst_edges:
        connected_nodes.add(edge.source)
        connected_nodes.add(edge.target)
    assert 5 in connected_nodes

def test_dijkstra(sample_graph):
    """Test Dijkstra's algorithm."""
    # Test without time-dependent weights
    result = dijkstra(sample_graph, 1, 5, datetime.now(), use_time_dependent=False)
    assert result.path
    assert result.path[0] == 1
    assert result.path[-1] == 5
    assert result.total_distance > 0
    assert result.total_time > 0
    
    # Test with time-dependent weights during rush hour
    rush_hour = datetime.now().replace(hour=8, minute=0)
    result = dijkstra(sample_graph, 1, 3, rush_hour, use_time_dependent=True)
    assert result.path
    assert result.path[0] == 1
    assert result.path[-1] == 3
    assert result.total_time > 0

def test_a_star(sample_graph):
    """Test A* algorithm."""
    # Test without time-dependent weights
    result = a_star(sample_graph, 1, 5, datetime.now(), use_time_dependent=False)
    assert result.path
    assert result.path[0] == 1
    assert result.path[-1] == 5
    assert result.total_distance > 0
    assert result.total_time > 0
    
    # Test with time-dependent weights during rush hour
    rush_hour = datetime.now().replace(hour=8, minute=0)
    result = a_star(sample_graph, 1, 3, rush_hour, use_time_dependent=True)
    assert result.path
    assert result.path[0] == 1
    assert result.path[-1] == 3
    assert result.total_time > 0

def test_mst_cost(sample_graph):
    """Test MST cost calculation."""
    mst_edges = kruskal_mst(sample_graph)
    cost = calculate_mst_cost(mst_edges)
    assert cost > 0
    assert cost == sum(edge.distance for edge in mst_edges)

def test_time_dependent_weights(sample_graph):
    """Test time-dependent edge weights."""
    # Get the time-dependent edge
    edge = sample_graph.get_edge(1, 3)
    assert isinstance(edge, TimeDependentEdge)
    
    # Test during rush hour
    rush_hour = datetime.now().replace(hour=8, minute=0)
    rush_hour_weight = sample_graph.get_time_dependent_weight(edge, rush_hour)
    assert rush_hour_weight == edge.base_time * 2.0
    
    # Test during non-rush hour
    normal_hour = datetime.now().replace(hour=14, minute=0)
    normal_weight = sample_graph.get_time_dependent_weight(edge, normal_hour)
    assert normal_weight == edge.base_time 