import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import networkx as nx
from datetime import datetime, time
import plotly.express as px
import plotly.graph_objects as go

from src.data_structures.graph import TransportationGraph, Node, Edge, TimeDependentEdge
from src.algorithms.shortest_path import dijkstra, a_star
from src.algorithms.advanced_pathfinding import greedy_shortest_path, DPShortestPath, multi_criteria_path
from src.algorithms.mst import kruskal_mst, calculate_mst_cost, visualize_mst
from src.algorithms.traffic_signals import TrafficSignalOptimizer

def load_data():
    """Load all CSV data into pandas DataFrames."""
    neighborhoods = pd.read_csv('data/neighborhoods.csv')
    facilities = pd.read_csv('data/facilities.csv')
    existing_roads = pd.read_csv('data/existing_roads.csv')
    potential_roads = pd.read_csv('data/potential_roads.csv')
    traffic_flow = pd.read_csv('data/traffic_flow.csv')
    metro_lines = pd.read_csv('data/metro_lines.csv')
    bus_routes = pd.read_csv('data/bus_routes.csv')
    return neighborhoods, facilities, existing_roads, potential_roads, traffic_flow, metro_lines, bus_routes

def get_traffic_multiplier(current_time: time) -> float:
    """Calculate traffic multiplier based on time of day."""
    if time(7, 0) <= current_time <= time(9, 0):
        return 1.5  # Morning peak
    elif time(12, 0) <= current_time <= time(14, 0):
        return 1.2  # Afternoon
    elif time(16, 0) <= current_time <= time(18, 0):
        return 1.4  # Evening peak
    elif time(22, 0) <= current_time <= time(5, 0):
        return 0.8  # Night
    return 1.0  # Normal hours

def create_transportation_graph(neighborhoods, facilities, existing_roads, traffic_flow, current_time=None):
    """Create the transportation graph from the data."""
    graph = TransportationGraph()
    
    # Add neighborhood nodes
    for _, row in neighborhoods.iterrows():
        node = Node(
            id=row['ID'],
            name=row['Name'],
            latitude=row['Y'],
            longitude=row['X'],
            node_type=row['Type'],
            is_critical=False
        )
        graph.add_node(node)
    
    # Add facility nodes
    for _, row in facilities.iterrows():
        node = Node(
            id=row['ID'],
            name=row['Name'],
            latitude=row['Y'],
            longitude=row['X'],
            node_type=row['Type'],
            is_critical=row['Type'] in ['Medical', 'Airport', 'Transit Hub']
        )
        graph.add_node(node)
    
    # Add edges with time-dependent weights
    for _, row in existing_roads.iterrows():
        edge = TimeDependentEdge(
            source=row['FromID'],
            target=row['ToID'],
            distance=row['DistanceKM'] * 1000,  # Convert to meters
            base_time=row['DistanceKM'] * 2,  # Rough estimate: 2 minutes per km
            capacity=row['Capacity'],
            road_type='highway' if row['Capacity'] > 3000 else 'arterial',
            lanes=4 if row['Capacity'] > 3000 else 2
        )
        
        # Add time variations based on traffic flow
        road_id = f"{row['FromID']}-{row['ToID']}"
        traffic = traffic_flow[traffic_flow['RoadID'] == road_id]
        if not traffic.empty:
            edge.time_variations = {
                (time(7, 0), time(9, 0)): traffic['MorningPeak'].iloc[0] / row['Capacity'],
                (time(12, 0), time(14, 0)): traffic['Afternoon'].iloc[0] / row['Capacity'],
                (time(16, 0), time(18, 0)): traffic['EveningPeak'].iloc[0] / row['Capacity'],
                (time(22, 0), time(5, 0)): traffic['Night'].iloc[0] / row['Capacity']
            }
            
            # Apply current time multiplier if provided
            if current_time:
                multiplier = get_traffic_multiplier(current_time)
                edge.base_time *= multiplier
        
        graph.add_edge(edge)
    
    return graph

def create_map(graph, path=None):
    """Create a Folium map with the transportation network."""
    # Calculate center point
    center_lat = sum(node.latitude for node in graph.nodes.values()) / len(graph.nodes)
    center_lon = sum(node.longitude for node in graph.nodes.values()) / len(graph.nodes)
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11)
    
    # Add nodes
    for node in graph.nodes.values():
        color = 'red' if node.is_critical else 'blue'
        folium.CircleMarker(
            location=[node.latitude, node.longitude],
            radius=5,
            popup=f"{node.name} ({node.node_type})",
            color=color,
            fill=True
        ).add_to(m)
    
    # Add edges
    for edge in graph.edges.values():
        source = graph.nodes[edge.source]
        target = graph.nodes[edge.target]
        
        # Calculate color based on capacity
        color = 'red' if edge.capacity > 3000 else 'orange' if edge.capacity > 2000 else 'blue'
        
        folium.PolyLine(
            locations=[[source.latitude, source.longitude], [target.latitude, target.longitude]],
            color=color,
            weight=2,
            popup=f"Capacity: {edge.capacity}"
        ).add_to(m)
    
    # Add path if provided
    if path:
        path_coords = [[graph.nodes[node_id].latitude, graph.nodes[node_id].longitude] for node_id in path]
        folium.PolyLine(
            locations=path_coords,
            color='green',
            weight=4,
            dash_array='5, 10'
        ).add_to(m)
    
    return m

def main():
    st.set_page_config(page_title="Cairo Transportation Network", layout="wide")
    st.title("Cairo Transportation Network Optimization")
    
    # Load data
    neighborhoods, facilities, existing_roads, potential_roads, traffic_flow, metro_lines, bus_routes = load_data()
    
    # Sidebar controls
    st.sidebar.header("Controls")
    
    # Time selection with more detailed options
    st.sidebar.subheader("Time Settings")
    time_period = st.sidebar.selectbox(
        "Select Time Period",
        [
            "Custom Time",
            "Morning Peak (7:00-9:00)",
            "Afternoon (12:00-14:00)",
            "Evening Peak (16:00-18:00)",
            "Night (22:00-5:00)"
        ],
        index=0  # Make Custom Time the default selection
    )
    
    # Set time based on period or custom selection
    if time_period == "Custom Time":
        col1, col2 = st.sidebar.columns(2)
        with col1:
            hours = st.number_input("Hour (0-23)", min_value=0, max_value=23, value=datetime.now().hour)
        with col2:
            minutes = st.number_input("Minute (0-59)", min_value=0, max_value=59, value=datetime.now().minute)
        current_time = time(hour=hours, minute=minutes)
        
        # Show time period category
        if time(7, 0) <= current_time <= time(9, 0):
            period_category = "Morning Peak"
        elif time(12, 0) <= current_time <= time(14, 0):
            period_category = "Afternoon"
        elif time(16, 0) <= current_time <= time(18, 0):
            period_category = "Evening Peak"
        elif time(22, 0) <= current_time <= time(5, 0):
            period_category = "Night"
        else:
            period_category = "Normal Hours"
        
        st.sidebar.info(f"Selected time falls in: {period_category}")
    else:
        if "Morning Peak" in time_period:
            current_time = time(8, 0)
        elif "Afternoon" in time_period:
            current_time = time(13, 0)
        elif "Evening Peak" in time_period:
            current_time = time(17, 0)
        else:  # Night
            current_time = time(23, 0)
    
    # Show current traffic conditions with more detail
    traffic_multiplier = get_traffic_multiplier(current_time)
    traffic_status = "Heavy" if traffic_multiplier > 1.3 else "Moderate" if traffic_multiplier > 1.1 else "Light"
    traffic_color = "red" if traffic_multiplier > 1.3 else "orange" if traffic_multiplier > 1.1 else "green"
    
    st.sidebar.markdown("""---""")
    st.sidebar.markdown("### Current Traffic Status")
    st.sidebar.markdown(f"""
    - **Time:** {current_time.strftime('%H:%M')}
    - **Traffic Level:** <span style='color: {traffic_color}'>{traffic_status}</span>
    - **Travel Time Multiplier:** {traffic_multiplier:.1f}x
    
    Traffic Impact:
    - Light Traffic: 1.0x travel time
    - Moderate Traffic: 1.1-1.3x travel time
    - Heavy Traffic: >1.3x travel time
    """, unsafe_allow_html=True)
    
    # Create graph with current time
    graph = create_transportation_graph(neighborhoods, facilities, existing_roads, traffic_flow, current_time)
    
    # Algorithm selection
    algorithm = st.sidebar.selectbox(
        "Select Algorithm",
        [
            "Shortest Path (Dijkstra)",
            "Shortest Path (A*)",
            "Greedy Path",
            "Dynamic Programming Path",
            "Multi-Criteria Path",
            "Minimum Spanning Tree"
        ]
    )
    
    if algorithm != "Minimum Spanning Tree":
        # Source and destination selection
        nodes = [(node.id, f"{node.name} ({node.node_type})") for node in graph.nodes.values()]
        source = st.sidebar.selectbox("Source", nodes, format_func=lambda x: x[1])
        destination = st.sidebar.selectbox("Destination", nodes, format_func=lambda x: x[1])
        
        # Additional parameters for multi-criteria path
        if algorithm == "Multi-Criteria Path":
            st.sidebar.subheader("Path Criteria Weights")
            weight_distance = st.sidebar.slider("Distance Weight", 0.0, 1.0, 0.4, 0.1)
            weight_time = st.sidebar.slider("Time Weight", 0.0, 1.0, 0.3, 0.1)
            weight_capacity = st.sidebar.slider("Road Capacity Weight", 0.0, 1.0, 0.3, 0.1)
            
            # Normalize weights
            total = weight_distance + weight_time + weight_capacity
            weight_distance /= total
            weight_time /= total
            weight_capacity /= total
        
        if st.sidebar.button("Find Path"):
            current_datetime = datetime.now().replace(
                hour=current_time.hour,
                minute=current_time.minute
            )
            
            if algorithm == "Shortest Path (Dijkstra)":
                result = dijkstra(graph, source[0], destination[0], current_datetime)
            elif algorithm == "Shortest Path (A*)":
                result = a_star(graph, source[0], destination[0], current_datetime)
            elif algorithm == "Greedy Path":
                result = greedy_shortest_path(graph, source[0], destination[0], current_datetime)
            elif algorithm == "Dynamic Programming Path":
                dp_solver = DPShortestPath(graph)
                result = dp_solver.find_path(source[0], destination[0], current_datetime)
            else:  # Multi-Criteria Path
                result = multi_criteria_path(
                    graph,
                    source[0],
                    destination[0],
                    current_datetime,
                    weight_distance,
                    weight_time,
                    weight_capacity
                )
            
            if result.path:
                # Calculate base time and actual time with traffic
                base_distance = result.total_distance / 1000  # km
                base_time = base_distance * 2  # minutes (2 min per km)
                actual_time = result.total_time
                
                st.success(f"""
                Path found!
                - Distance: {base_distance:.1f} km
                - Base travel time: {base_time:.1f} minutes
                - Actual travel time with traffic: {actual_time:.1f} minutes
                - Traffic delay: {(actual_time - base_time):.1f} minutes
                """)
                
                m = create_map(graph, result.path)
                folium_static(m)
                
                # Show path details
                st.subheader("Path Details")
                path_details = []
                for i in range(len(result.path) - 1):
                    current = result.path[i]
                    next_node = result.path[i + 1]
                    edge = graph.get_edge(current, next_node)
                    if edge:
                        base_time = (edge.distance / 1000) * 2  # 2 min per km
                        actual_time = graph.get_time_dependent_weight(edge, current_datetime)
                        path_details.append({
                            'From': graph.nodes[current].name,
                            'To': graph.nodes[next_node].name,
                            'Distance (km)': edge.distance / 1000,
                            'Base Time (min)': base_time,
                            'Actual Time (min)': actual_time,
                            'Delay (min)': actual_time - base_time,
                            'Road Capacity': edge.capacity
                        })
                if path_details:
                    st.table(pd.DataFrame(path_details))
            else:
                st.error("No path found!")
    
    elif algorithm == "Minimum Spanning Tree":
        if st.sidebar.button("Calculate MST"):
            critical_nodes = {node.id for node in graph.nodes.values() if node.is_critical}
            mst_edges = kruskal_mst(graph, critical_nodes)
            cost = calculate_mst_cost(mst_edges)
            st.success(f"MST calculated! Total cost: {cost/1000:.1f} km")

            # Use visualize_mst from mst.py
            visualize_mst(graph, mst_edges)
    
    # Display statistics
    st.sidebar.header("Network Statistics")
    st.sidebar.write(f"Total Nodes: {len(graph.nodes)}")
    st.sidebar.write(f"Total Edges: {len(graph.edges)}")
    st.sidebar.write(f"Critical Facilities: {len([n for n in graph.nodes.values() if n.is_critical])}")
    
    # Traffic flow visualization
    st.header("Traffic Flow Analysis")
    selected_time_period = "Morning Peak" if "Morning" in time_period else \
                          "Afternoon" if "Afternoon" in time_period else \
                          "Evening Peak" if "Evening" in time_period else "Night"
    
    flow_data = traffic_flow.melt(
        id_vars=['RoadID'],
        value_vars=['MorningPeak', 'Afternoon', 'EveningPeak', 'Night'],
        var_name='Time Period',
        value_name='Flow'
    )
    
    fig = px.bar(
        flow_data[flow_data['Time Period'] == selected_time_period.replace(" ", "")],
        x='RoadID',
        y='Flow',
        title=f"Traffic Flow - {selected_time_period}"
    )
    st.plotly_chart(fig)

if __name__ == "__main__":
    main() 