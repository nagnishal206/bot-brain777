# Overview

This is an Interactive Campus Pathfinding System built with Streamlit that provides route planning and navigation capabilities for a university campus. The system integrates multiple pathfinding algorithms (BFS, DFS, UCS, A*) with an interactive map interface and AI-powered assistance through Google's Gemini API. Users can select start and end locations from predefined Points of Interest (POIs) and visualize optimal routes on an interactive map using real campus OpenStreetMap data.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Streamlit Web Application**: Single-page application with sidebar controls and main map display
- **Interactive Map Visualization**: Uses Folium library for rendering maps with route overlays
- **Streamlit-Folium Integration**: Enables interactive map components within the Streamlit interface
- **Responsive Layout**: Wide layout configuration with expandable sidebar for route planning controls

## Backend Architecture
- **Modular Design**: Three main components separated into distinct modules:
  - `app.py`: Main Streamlit application and UI orchestration
  - `pathfinding.py`: Core pathfinding algorithms and graph processing
  - `gemini_integration.py`: AI assistant functionality
- **Graph Processing**: Uses OSMnx library to process OpenStreetMap data into navigable graph structures
- **Algorithm Implementation**: Multiple pathfinding algorithms with performance comparison capabilities
- **Caching Strategy**: Streamlit resource caching for expensive operations like graph loading and AI client initialization

## Data Storage Solutions
- **Static Campus Data**: Predefined Points of Interest (POIs) stored as coordinate dictionaries
- **OpenStreetMap Integration**: Campus map data loaded from OSM XML files for accurate geographical representation
- **Graph Representation**: NetworkX graph structure for efficient pathfinding operations
- **No Database Required**: All data is processed in-memory from static files

## Pathfinding Engine
- **Multiple Algorithms**: BFS, DFS, Uniform Cost Search (UCS), and A* implementations
- **Heuristic Functions**: Straight-line distance calculations for A* optimization
- **Performance Metrics**: Node exploration tracking and algorithm comparison capabilities
- **Real-world Mapping**: Integration with actual campus geography using coordinate systems

# External Dependencies

## Core Libraries
- **Streamlit**: Web application framework for the user interface
- **OSMnx**: OpenStreetMap data processing and graph creation
- **NetworkX**: Graph data structure and algorithms
- **Folium**: Interactive map visualization
- **Pandas**: Data manipulation and analysis
- **Streamlit-Folium**: Bridge between Streamlit and Folium components

## AI Integration
- **Google Gemini API**: AI-powered query processing and campus assistance
- **Model**: Uses Gemini 2.5 Flash for natural language processing
- **Fallback System**: Graceful degradation when AI services are unavailable

## Map Data Source
- **OpenStreetMap**: Campus geographical data in OSM XML format
- **Local Assets**: Campus map stored in `attached_assets/` directory
- **Coordinate System**: Latitude/longitude based positioning for POIs

## Environment Configuration
- **API Keys**: Environment variable management for Gemini API access
- **Error Handling**: Robust fallback mechanisms for missing dependencies or API failures