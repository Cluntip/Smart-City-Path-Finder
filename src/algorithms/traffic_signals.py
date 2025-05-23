from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import time
from src.data_structures.graph import TransportationGraph, Node, Edge

@dataclass
class TrafficSignal:
    """Represents a traffic signal at an intersection."""
    node_id: int
    cycle_time: int  # Total cycle time in seconds
    phases: List[Tuple[List[int], int]]  # List of (incoming_roads, duration) pairs
    current_phase: int = 0
    time_in_current_phase: int = 0

class TrafficSignalOptimizer:
    """Optimizes traffic signal timings using a greedy approach."""
    def __init__(self, graph: TransportationGraph):
        self.graph = graph
        self.signals: Dict[int, TrafficSignal] = {}
        self.initialize_signals()

    def initialize_signals(self):
        """Initialize traffic signals for all intersections."""
        for node_id, node in self.graph.nodes.items():
            if node.node_type == "intersection":
                # Get all incoming roads
                incoming_roads = []
                for neighbor, edge in self.graph.get_neighbors(node_id):
                    if not edge.is_one_way or edge.target == node_id:
                        incoming_roads.append(edge)

                # Group roads into phases (opposing directions)
                phases = self._group_roads_into_phases(incoming_roads)
                
                # Create traffic signal
                signal = TrafficSignal(
                    node_id=node_id,
                    cycle_time=120,  # Default 2-minute cycle
                    phases=phases
                )
                self.signals[node_id] = signal

    def _group_roads_into_phases(self, roads: List[Edge]) -> List[Tuple[List[int], int]]:
        """Group roads into phases based on their directions."""
        if not roads:
            return []

        # Simple grouping: each road gets its own phase
        # In a real implementation, this would consider road directions and conflicts
        phases = []
        for road in roads:
            # Calculate phase duration based on traffic volume
            duration = max(20, min(60, road.capacity // 100))  # 20-60 seconds
            phases.append(([road.source], duration))

        return phases

    def optimize_signals(self, current_time: time):
        """Optimize traffic signal timings based on current conditions."""
        for signal in self.signals.values():
            # Adjust cycle time based on time of day
            if time(7, 0) <= current_time <= time(9, 0) or \
               time(16, 0) <= current_time <= time(18, 0):
                # Rush hour: longer cycle time
                signal.cycle_time = 180
            else:
                # Normal hours: shorter cycle time
                signal.cycle_time = 120

            # Adjust phase durations based on traffic volume
            total_volume = 0
            for phase_roads, _ in signal.phases:
                for road_id in phase_roads:
                    edge = self.graph.get_edge(road_id, signal.node_id)
                    if edge:
                        total_volume += edge.capacity

            # Update phase durations proportionally to traffic volume
            for i, (phase_roads, _) in enumerate(signal.phases):
                phase_volume = sum(
                    self.graph.get_edge(road_id, signal.node_id).capacity
                    for road_id in phase_roads
                    if self.graph.get_edge(road_id, signal.node_id)
                )
                if total_volume > 0:
                    duration = int((phase_volume / total_volume) * signal.cycle_time)
                    signal.phases[i] = (phase_roads, max(20, min(60, duration)))

    def update_signal_state(self, elapsed_seconds: int):
        """Update the state of all traffic signals."""
        for signal in self.signals.values():
            signal.time_in_current_phase += elapsed_seconds
            
            # Check if it's time to change phase
            if signal.time_in_current_phase >= signal.phases[signal.current_phase][1]:
                signal.time_in_current_phase = 0
                signal.current_phase = (signal.current_phase + 1) % len(signal.phases)

    def get_current_phase(self, node_id: int) -> Tuple[List[int], int]:
        """Get the current phase for a traffic signal."""
        signal = self.signals.get(node_id)
        if signal:
            return signal.phases[signal.current_phase]
        return ([], 0)

    def emergency_preemption(self, node_id: int, emergency_route: List[int]):
        """Handle emergency vehicle preemption."""
        if node_id not in self.signals:
            return

        signal = self.signals[node_id]
        
        # Find the phase that includes the emergency route
        for i, (phase_roads, _) in enumerate(signal.phases):
            if any(road in emergency_route for road in phase_roads):
                # Switch to the emergency phase immediately
                signal.current_phase = i
                signal.time_in_current_phase = 0
                # Extend the phase duration for emergency vehicles
                signal.phases[i] = (phase_roads, 30)  # 30 seconds for emergency 