import osmnx as ox
import networkx as nx
import folium
import heapq
import math
import pandas as pd
from collections import deque
from typing import Dict, List, Tuple, Any, Optional

class CampusPathfinder:
    def __init__(self, osm_file_path: str):
        """Initialize the pathfinder with OSM data."""
        self.graph = ox.graph_from_xml(osm_file_path, simplify=False)
        self.nodes, self.edges = ox.graph_to_gdfs(self.graph)
        self.center = (self.nodes.geometry.y.mean(), self.nodes.geometry.x.mean())
        
        # Points of Interest with coordinates (lat, lon)
        self.POIS = {
            "Flag post": (13.22169, 77.75495),
            "Entry gate": (13.22020, 77.75417),
            "Exit gate": (13.22017, 77.75508),
            "Check post 1": (13.22140, 77.75507),
            "Check post 2": (13.22128, 77.75528),
            "Acad 1": (13.22218, 77.75537),
            "Acad 2": (13.22339, 77.75595),
            "Library": (13.22199, 77.75540),
            "Food Court": (13.22488, 77.75716),
            "Faculty Block": (13.22359, 77.75726),
            "Hostel Block": (13.22458, 77.75886),
            "Cricket Ground": (13.22898, 77.75718),
            "Basket Ball": (13.22883, 77.75813),
            "Volley Ball": (13.22869, 77.75856),
            "Tennis Ball": (13.22840, 77.75837),
            "Foot Ball": (13.22769, 77.75642),
            "Rest Area": (13.22834, 77.75775)
        }
        
        # Walking speed in meters per second (average human walking speed)
        self.WALKING_SPEED = 1.4
    
    def euclidean_heuristic(self, node1: int, node2: int) -> float:
        """Calculate Euclidean distance heuristic for A*."""
        y1, x1 = self.graph.nodes[node1]['y'], self.graph.nodes[node1]['x']
        y2, x2 = self.graph.nodes[node2]['y'], self.graph.nodes[node2]['x']
        return math.dist([y1, x1], [y2, x2]) * 111000  # Convert to meters
    
    def manhattan_heuristic(self, node1: int, node2: int) -> float:
        """Calculate Manhattan distance heuristic for A*."""
        y1, x1 = self.graph.nodes[node1]['y'], self.graph.nodes[node1]['x']
        y2, x2 = self.graph.nodes[node2]['y'], self.graph.nodes[node2]['x']
        return (abs(y1 - y2) + abs(x1 - x2)) * 111000  # Convert to meters
    
    def combined_heuristic(self, node1: int, node2: int) -> float:
        """Calculate combined weighted heuristic (0.7 * Euclidean + 0.3 * Manhattan)."""
        euclidean = self.euclidean_heuristic(node1, node2)
        manhattan = self.manhattan_heuristic(node1, node2)
        return 0.7 * euclidean + 0.3 * manhattan
    
    # Backward compatibility
    def heuristic(self, node1: int, node2: int) -> float:
        """Default heuristic (Euclidean distance) for backward compatibility."""
        return self.euclidean_heuristic(node1, node2)
    
    def bfs_osm(self, start: int, end: int) -> Tuple[Optional[List[int]], set]:
        """Breadth-First Search implementation."""
        frontier = deque([[start]])
        explored = set()
        
        while frontier:
            path = frontier.popleft()
            node = path[-1]
            
            if node == end:
                return path, explored
            
            if node not in explored:
                explored.add(node)
                for nbr in self.graph.neighbors(node):
                    if nbr not in explored:
                        frontier.append(path + [nbr])
        
        return None, explored
    
    def dfs_osm(self, start: int, end: int) -> Tuple[Optional[List[int]], set]:
        """Depth-First Search implementation."""
        frontier = [[start]]
        explored = set()
        
        while frontier:
            path = frontier.pop()
            node = path[-1]
            
            if node == end:
                return path, explored
            
            if node not in explored:
                explored.add(node)
                for nbr in self.graph.neighbors(node):
                    if nbr not in explored:
                        frontier.append(path + [nbr])
        
        return None, explored
    
    def ucs_osm(self, start: int, end: int) -> Tuple[Optional[List[int]], Optional[float], set]:
        """Uniform Cost Search implementation."""
        frontier = [(0, [start])]
        explored = set()
        
        while frontier:
            cost, path = heapq.heappop(frontier)
            node = path[-1]
            
            if node == end:
                return path, cost, explored
            
            if node not in explored:
                explored.add(node)
                for nbr in self.graph.neighbors(node):
                    if nbr not in explored:
                        edge_data = self.graph.get_edge_data(node, nbr)
                        if edge_data:
                            weight = min([d.get('length', 1) for d in edge_data.values()])
                            heapq.heappush(frontier, (cost + weight, path + [nbr]))
        
        return None, None, explored
    
    def astar_osm(self, start: int, end: int, heuristic_type: str = "euclidean") -> Tuple[Optional[List[int]], Optional[float], set]:
        """A* Search implementation with selectable heuristic."""
        # Select heuristic function
        if heuristic_type == "manhattan":
            heuristic_func = self.manhattan_heuristic
        elif heuristic_type == "combined":
            heuristic_func = self.combined_heuristic
        else:  # default to euclidean
            heuristic_func = self.euclidean_heuristic
        
        frontier = [(heuristic_func(start, end), 0, [start])]
        explored = set()
        
        while frontier:
            f, g, path = heapq.heappop(frontier)
            node = path[-1]
            
            if node == end:
                return path, g, explored
            
            if node not in explored:
                explored.add(node)
                for nbr in self.graph.neighbors(node):
                    if nbr not in explored:
                        edge_data = self.graph.get_edge_data(node, nbr)
                        if edge_data:
                            weight = min([d.get('length', 1) for d in edge_data.values()])
                            new_g = g + weight
                            new_f = new_g + heuristic_func(nbr, end)
                            heapq.heappush(frontier, (new_f, new_g, path + [nbr]))
        
        return None, None, explored
    
    def astar_euclidean(self, start: int, end: int) -> Tuple[Optional[List[int]], Optional[float], set]:
        """A* with Euclidean heuristic."""
        return self.astar_osm(start, end, "euclidean")
    
    def astar_manhattan(self, start: int, end: int) -> Tuple[Optional[List[int]], Optional[float], set]:
        """A* with Manhattan heuristic."""
        return self.astar_osm(start, end, "manhattan")
    
    def astar_combined(self, start: int, end: int) -> Tuple[Optional[List[int]], Optional[float], set]:
        """A* with combined heuristic."""
        return self.astar_osm(start, end, "combined")
    
    def calculate_path_distance(self, path: List[int]) -> float:
        """Calculate total distance of a path in meters."""
        total_distance = 0.0
        
        for i in range(len(path) - 1):
            edge_data = self.graph.get_edge_data(path[i], path[i + 1])
            if edge_data:
                weight = min([d.get('length', 0) for d in edge_data.values()])
                total_distance += weight
        
        return total_distance
    
    def calculate_walking_time(self, distance: float) -> float:
        """Calculate estimated walking time in minutes."""
        return (distance / self.WALKING_SPEED) / 60  # Convert to minutes
    
    def get_location_info(self, location_name: str) -> Dict[str, Any]:
        """Get information about a specific location."""
        coordinates = self.POIS.get(location_name, (0, 0))
        
        return {
            'name': location_name,
            'coordinates': f"{coordinates[0]:.5f}, {coordinates[1]:.5f}",
            'type': self._categorize_location(location_name)
        }
    
    def _categorize_location(self, location_name: str) -> str:
        """Categorize location by type."""
        academic = ['Acad 1', 'Acad 2', 'Library']
        facilities = ['Food Court', 'Faculty Block', 'Hostel Block']
        sports = ['Cricket Ground', 'Basket Ball', 'Volley Ball', 'Tennis Ball', 'Foot Ball']
        security = ['Entry gate', 'Exit gate', 'Check post 1', 'Check post 2']
        
        if location_name in academic:
            return "Academic"
        elif location_name in facilities:
            return "Facility"
        elif location_name in sports:
            return "Sports"
        elif location_name in security:
            return "Security"
        else:
            return "Other"
    
    def create_base_map(self) -> folium.Map:
        """Create a base map with all roads and POIs."""
        m = folium.Map(location=self.center, zoom_start=17)
        
        # Add all roads in gray
        for _, row in self.edges.iterrows():
            coords = [(lat, lon) for lon, lat in row.geometry.coords]
            folium.PolyLine(coords, color="gray", weight=2, opacity=0.4).add_to(m)
        
        # Add POI markers
        for name, (lat, lon) in self.POIS.items():
            folium.Marker(
                (lat, lon),
                popup=f"{name}",
                tooltip=name,
                icon=folium.Icon(color="blue", icon="info-sign")
            ).add_to(m)
        
        return m
    
    def find_path(self, start_name: str, end_name: str, algorithm: str) -> Dict[str, Any]:
        """Find path between two locations using specified algorithm."""
        start_latlon = self.POIS[start_name]
        end_latlon = self.POIS[end_name]
        
        start_node = ox.distance.nearest_nodes(self.graph, start_latlon[1], start_latlon[0])
        end_node = ox.distance.nearest_nodes(self.graph, end_latlon[1], end_latlon[0])
        
        # Run the selected algorithm
        if algorithm == "BFS":
            path, explored = self.bfs_osm(start_node, end_node)
            cost = self.calculate_path_distance(path) if path else None
        elif algorithm == "DFS":
            path, explored = self.dfs_osm(start_node, end_node)
            cost = self.calculate_path_distance(path) if path else None
        elif algorithm == "UCS":
            path, cost, explored = self.ucs_osm(start_node, end_node)
        elif algorithm == "A* (Euclidean)":
            path, cost, explored = self.astar_euclidean(start_node, end_node)
        elif algorithm == "A* (Manhattan)":
            path, cost, explored = self.astar_manhattan(start_node, end_node)
        elif algorithm == "A* (Combined)":
            path, cost, explored = self.astar_combined(start_node, end_node)
        else:  # Default A*
            path, cost, explored = self.astar_osm(start_node, end_node)
        
        if not path:
            raise Exception("No path found between the selected locations")
        
        # Create visualization map
        m = folium.Map(location=self.center, zoom_start=17)
        
        # Add all roads in gray
        for _, row in self.edges.iterrows():
            coords = [(lat, lon) for lon, lat in row.geometry.coords]
            folium.PolyLine(coords, color="gray", weight=2, opacity=0.4).add_to(m)
        
        # Add explored nodes in orange
        for node in explored:
            y, x = self.graph.nodes[node]['y'], self.graph.nodes[node]['x']
            folium.CircleMarker(
                (y, x),
                radius=3,
                color="orange",
                opacity=0.5,
                popup=f"Explored: {node}"
            ).add_to(m)
        
        # Add final path
        if path:
            coords = [(self.graph.nodes[n]['y'], self.graph.nodes[n]['x']) for n in path]
            # White outline
            folium.PolyLine(coords, color="white", weight=8, opacity=0.8).add_to(m)
            # Blue path
            folium.PolyLine(coords, color="blue", weight=4, opacity=1).add_to(m)
        
        # Add start and end markers
        folium.Marker(
            start_latlon,
            popup=f"Start: {start_name}",
            icon=folium.Icon(color="green", icon="play")
        ).add_to(m)
        
        folium.Marker(
            end_latlon,
            popup=f"End: {end_name}",
            icon=folium.Icon(color="red", icon="stop")
        ).add_to(m)
        
        # Calculate metrics
        distance = cost if cost else self.calculate_path_distance(path)
        walking_time = self.calculate_walking_time(distance)
        
        return {
            'map': m,
            'metrics': {
                'distance': distance,
                'time': walking_time,
                'nodes_explored': len(explored),
                'start_location': start_name,
                'end_location': end_name
            }
        }
    
    def compare_algorithms(self) -> List[Dict[str, Any]]:
        """Compare all algorithms on multiple test routes."""
        # Define test cases
        test_routes = [
            ("Entry gate", "Library"),
            ("Library", "Food Court"),
            ("Cricket Ground", "Hostel Block")
        ]
        
        algorithms = ["BFS", "DFS", "UCS", "A*"]
        results = []
        
        for algo in algorithms:
            total_distance = 0
            total_nodes = 0
            successful_runs = 0
            
            for start, end in test_routes:
                try:
                    result = self.find_path(start, end, algo)
                    total_distance += result['metrics']['distance']
                    total_nodes += result['metrics']['nodes_explored']
                    successful_runs += 1
                except:
                    continue
            
            if successful_runs > 0:
                results.append({
                    'Algorithm': algo,
                    'Average Distance (m)': round(total_distance / successful_runs, 2),
                    'Average Nodes Explored': round(total_nodes / successful_runs, 2),
                    'Average Time (min)': round(self.calculate_walking_time(total_distance / successful_runs), 2),
                    'Success Rate': f"{successful_runs}/{len(test_routes)}"
                })
        
        return results
    
    def compare_heuristics(self) -> List[Dict[str, Any]]:
        """Compare A* algorithm with different heuristics on multiple test routes."""
        # Define test cases
        test_routes = [
            ("Entry gate", "Library"),
            ("Library", "Food Court"),
            ("Cricket Ground", "Hostel Block"),
            ("Acad 1", "Rest Area"),
            ("Food Court", "Basket Ball")
        ]
        
        heuristics = [
            "A* (Euclidean)",
            "A* (Manhattan)", 
            "A* (Combined)"
        ]
        results = []
        
        for heuristic in heuristics:
            total_distance = 0
            total_nodes = 0
            total_time = 0
            successful_runs = 0
            
            for start, end in test_routes:
                try:
                    result = self.find_path(start, end, heuristic)
                    total_distance += result['metrics']['distance']
                    total_nodes += result['metrics']['nodes_explored']
                    total_time += result['metrics']['time']
                    successful_runs += 1
                except:
                    continue
            
            if successful_runs > 0:
                heuristic_name = heuristic.replace("A* (", "").replace(")", "")
                results.append({
                    'Heuristic Type': heuristic_name,
                    'Average Distance (m)': round(total_distance / successful_runs, 2),
                    'Average Nodes Explored': round(total_nodes / successful_runs, 2),
                    'Average Time (min)': round(total_time / successful_runs, 2),
                    'Efficiency Score': round((total_distance / successful_runs) / (total_nodes / successful_runs), 4),
                    'Success Rate': f"{successful_runs}/{len(test_routes)}"
                })
        
        return results
