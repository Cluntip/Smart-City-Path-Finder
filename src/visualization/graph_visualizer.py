from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QComboBox, QSpinBox)
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QPainter, QPen, QColor, QBrush, QPixmap
from ..data_structures.graph import TransportationGraph, Node, Edge
from ..algorithms.shortest_path import ShortestPathResult
from ..algorithms.traffic_signals import TrafficSignalOptimizer
import math
from typing import Dict, List, Tuple

class GraphVisualizer(QMainWindow):
    """Main window for visualizing the transportation network."""
    def __init__(self, graph: TransportationGraph):
        super().__init__()
        self.graph = graph
        self.selected_path: List[int] = []
        self.node_positions: Dict[int, QPointF] = {}
        self.traffic_signal_optimizer = TrafficSignalOptimizer(graph)
        self.setup_ui()
        self.calculate_node_positions()

    def setup_ui(self):
        """Set up the user interface."""
        self.setWindowTitle("Cairo Transportation Network")
        self.setGeometry(100, 100, 1200, 800)

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)

        # Create control panel
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_panel.setMaximumWidth(200)

        # Add algorithm selection
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems(["Dijkstra's", "A*"])
        control_layout.addWidget(QLabel("Algorithm:"))
        control_layout.addWidget(self.algorithm_combo)

        # Add start/end node selection
        self.start_node_combo = QComboBox()
        self.end_node_combo = QComboBox()
        for node in self.graph.nodes.values():
            self.start_node_combo.addItem(f"{node.name} ({node.id})", node.id)
            self.end_node_combo.addItem(f"{node.name} ({node.id})", node.id)
        
        control_layout.addWidget(QLabel("Start Node:"))
        control_layout.addWidget(self.start_node_combo)
        control_layout.addWidget(QLabel("End Node:"))
        control_layout.addWidget(self.end_node_combo)

        # Add find path button
        find_path_btn = QPushButton("Find Path")
        find_path_btn.clicked.connect(self.find_path)
        control_layout.addWidget(find_path_btn)

        # Add visualization options
        self.show_mst_btn = QPushButton("Show MST")
        self.show_mst_btn.clicked.connect(self.toggle_mst)
        control_layout.addWidget(self.show_mst_btn)

        control_layout.addStretch()
        layout.addWidget(control_panel)

        # Create visualization area
        self.visualization_area = VisualizationArea(self)
        layout.addWidget(self.visualization_area)

    def calculate_node_positions(self):
        """Calculate positions for nodes in the visualization."""
        # Find min/max coordinates for scaling
        min_lat = min(node.latitude for node in self.graph.nodes.values())
        max_lat = max(node.latitude for node in self.graph.nodes.values())
        min_lon = min(node.longitude for node in self.graph.nodes.values())
        max_lon = max(node.longitude for node in self.graph.nodes.values())

        # Calculate scaling factors
        lat_scale = 700 / (max_lat - min_lat)
        lon_scale = 900 / (max_lon - min_lon)
        scale = min(lat_scale, lon_scale)

        # Calculate positions
        for node_id, node in self.graph.nodes.items():
            x = (node.longitude - min_lon) * scale + 50
            y = (node.latitude - min_lat) * scale + 50
            self.node_positions[node_id] = QPointF(x, y)

    def find_path(self):
        """Find and display the shortest path between selected nodes."""
        start_id = self.start_node_combo.currentData()
        end_id = self.end_node_combo.currentData()
        
        # TODO: Implement path finding using selected algorithm
        # For now, just highlight the selected nodes
        self.selected_path = [start_id, end_id]
        self.visualization_area.update()

    def toggle_mst(self):
        """Toggle display of Minimum Spanning Tree."""
        # TODO: Implement MST visualization
        pass

class VisualizationArea(QWidget):
    """Widget for drawing the transportation network."""
    def __init__(self, parent: GraphVisualizer):
        super().__init__(parent)
        self.parent = parent
        self.setMinimumSize(900, 700)

    def paintEvent(self, event):
        """Handle painting of the visualization."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw edges with color based on capacity/traffic
        for edge in self.parent.graph.edges.values():
            if edge.source in self.parent.node_positions and edge.target in self.parent.node_positions:
                start_pos = self.parent.node_positions[edge.source]
                end_pos = self.parent.node_positions[edge.target]
                # Color: green (high capacity), yellow (medium), red (low)
                if edge.capacity > 3000:
                    color = QColor(0, 200, 0)
                elif edge.capacity > 1500:
                    color = QColor(255, 215, 0)
                else:
                    color = QColor(200, 0, 0)
                pen = QPen(color, 2)
                painter.setPen(pen)
                painter.drawLine(start_pos, end_pos)

        # Draw nodes with icons or colored circles
        for node_id, pos in self.parent.node_positions.items():
            node = self.parent.graph.nodes[node_id]
            # Example: use icons for bus/train, else colored circle
            if node.node_type == 'bus_stop':
                # Draw a bus icon (replace with actual icon path if available)
                # pixmap = QPixmap('bus_icon.png')
                # painter.drawPixmap(pos.x()-8, pos.y()-8, 16, 16, pixmap)
                painter.setBrush(QBrush(QColor(255, 255, 0)))
                painter.setPen(QPen(Qt.black, 1))
                painter.drawEllipse(pos, 8, 8)
            elif node.node_type == 'train_station':
                # pixmap = QPixmap('train_icon.png')
                # painter.drawPixmap(pos.x()-8, pos.y()-8, 16, 16, pixmap)
                painter.setBrush(QBrush(QColor(0, 255, 255)))
                painter.setPen(QPen(Qt.black, 1))
                painter.drawEllipse(pos, 8, 8)
            elif node.is_critical:
                painter.setBrush(QBrush(QColor(255, 0, 0)))
                painter.setPen(QPen(Qt.black, 2))
                painter.drawEllipse(pos, 10, 10)
            else:
                painter.setBrush(QBrush(QColor(0, 0, 255)))
                painter.setPen(QPen(Qt.black, 1))
                painter.drawEllipse(pos, 7, 7)

            # Draw traffic light if intersection
            if node.node_type == 'intersection':
                signal = self.parent.traffic_signal_optimizer.signals.get(node_id)
                if signal:
                    # Draw a small traffic light icon (circle) next to node
                    phase, duration = signal.phases[signal.current_phase]
                    # Color: green if this phase is active, else red
                    color = QColor(0, 200, 0) if duration > 20 else QColor(200, 0, 0)
                    painter.setBrush(QBrush(color))
                    painter.setPen(QPen(Qt.black, 1))
                    painter.drawEllipse(pos.x()+12, pos.y()-12, 8, 8)
                    # Optionally, draw phase number
                    painter.setPen(QPen(Qt.black))
                    painter.drawText(pos.x()+22, pos.y()-6, f"P{signal.current_phase+1}")

        # Draw selected path
        if self.parent.selected_path:
            pen = QPen(QColor(0, 255, 0), 2, Qt.DashLine)
            painter.setPen(pen)
            for i in range(len(self.parent.selected_path) - 1):
                start_pos = self.parent.node_positions[self.parent.selected_path[i]]
                end_pos = self.parent.node_positions[self.parent.selected_path[i + 1]]
                painter.drawLine(start_pos, end_pos)