import sys
from PyQt5.QtWidgets import QApplication
from data_structures.graph import TransportationGraph, Node, Edge, TimeDependentEdge
from visualization.graph_visualizer import GraphVisualizer
from datetime import time

def create_sample_graph() -> TransportationGraph:
    """Create a sample transportation graph for testing."""
    graph = TransportationGraph()
    
    # Add some sample nodes (using approximate Cairo coordinates)
    nodes = [
        Node(1, "Downtown Cairo", 30.0444, 31.2357, "intersection", True),
        Node(2, "Giza", 30.0131, 31.2089, "intersection"),
        Node(3, "Heliopolis", 30.1022, 31.3403, "intersection"),
        Node(4, "Maadi", 29.9627, 31.2717, "intersection"),
        Node(5, "Nasr City", 30.0500, 31.3667, "intersection"),
        Node(6, "6th of October City", 29.9667, 30.9500, "intersection"),
        Node(7, "New Cairo", 30.0300, 31.4700, "intersection"),
        Node(8, "Shubra", 30.1167, 31.2333, "intersection"),
    ]
    
    for node in nodes:
        graph.add_node(node)
    
    # Add some sample edges
    edges = [
        Edge(1, 2, 10000, 20, 2000, "highway", 4),  # Downtown to Giza
        Edge(1, 3, 8000, 15, 1500, "arterial", 3),  # Downtown to Heliopolis
        Edge(1, 4, 7000, 12, 1200, "arterial", 2),  # Downtown to Maadi
        Edge(2, 6, 25000, 30, 1800, "highway", 4),  # Giza to 6th of October
        Edge(3, 5, 5000, 10, 1000, "local", 2),     # Heliopolis to Nasr City
        Edge(4, 7, 15000, 25, 1600, "arterial", 3), # Maadi to New Cairo
        Edge(5, 7, 12000, 20, 1400, "arterial", 2), # Nasr City to New Cairo
        Edge(1, 8, 6000, 12, 1000, "local", 2),     # Downtown to Shubra
    ]
    
    # Add time-dependent variations for some edges
    rush_hour_edge = TimeDependentEdge(3, 5, 5000, 10, 1000, "local", 2)
    rush_hour_edge.time_variations = {
        (time(7, 0), time(9, 0)): 2.5,    # Morning rush hour
        (time(16, 0), time(18, 0)): 2.5,  # Evening rush hour
    }
    edges.append(rush_hour_edge)
    
    for edge in edges:
        graph.add_edge(edge)
    
    return graph

def main():
    """Main entry point for the application."""
    app = QApplication(sys.argv)
    
    # Create and populate the transportation graph
    graph = create_sample_graph()
    
    # Create and show the visualization window
    visualizer = GraphVisualizer(graph)
    visualizer.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 