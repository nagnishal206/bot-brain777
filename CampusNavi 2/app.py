import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import os
from pathfinding import CampusPathfinder
from gemini_integration import GeminiAssistant

# Page configuration
st.set_page_config(
    page_title="Campus Pathfinding System",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize pathfinder and Gemini assistant
@st.cache_resource
def initialize_pathfinder():
    return CampusPathfinder("attached_assets/map_1758707724808.osm")

@st.cache_resource
def initialize_gemini():
    return GeminiAssistant()

try:
    pathfinder = initialize_pathfinder()
    gemini = initialize_gemini()
    
    # Main title
    st.title("🗺️ Interactive Campus Pathfinding System")
    st.markdown("---")
    
    # Sidebar for controls
    with st.sidebar:
        st.header("🎯 Route Planning")
        
        # Location selection
        locations = list(pathfinder.POIS.keys())
        start_location = st.selectbox("📍 Start Location", locations, index=0)
        end_location = st.selectbox("🏁 End Location", locations, index=1)
        
        # Algorithm selection
        st.subheader("🧠 Algorithm Selection")
        algorithm = st.selectbox(
            "Choose Algorithm",
            ["BFS", "DFS", "UCS", "A*", "A* (Euclidean)", "A* (Manhattan)", "A* (Combined)"],
            index=3,  # Default to A*
            help="Select the pathfinding algorithm to use. A* variants use different heuristics."
        )
        
        # Run pathfinding button
        run_pathfinding = st.button("🚀 Find Path", type="primary", use_container_width=True)
        
        st.markdown("---")
        
        # Algorithm comparison section
        st.subheader("📊 Algorithm Comparison")
        
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🔬 Compare Algorithms"):
                st.session_state['run_comparison'] = True
        
        with col_b:
            if st.button("🎯 Compare Heuristics"):
                st.session_state['run_heuristic_comparison'] = True
        
        st.markdown("---")
        
        # Gemini AI section
        st.subheader("🤖 AI Assistant")
        user_query = st.text_area(
            "Ask about routes or campus info:",
            placeholder="e.g., 'How do I get from Library to Food Court?' or 'Tell me about the cricket ground'",
            height=100
        )
        
        if st.button("💬 Ask AI", use_container_width=True) and user_query:
            st.session_state['ai_query'] = user_query
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🗺️ Interactive Campus Map")
        
        # Initialize map display
        if 'current_map' not in st.session_state:
            st.session_state['current_map'] = pathfinder.create_base_map()
        
        # Handle pathfinding request
        if run_pathfinding:
            try:
                with st.spinner(f"Finding path using {algorithm}..."):
                    result = pathfinder.find_path(start_location, end_location, algorithm)
                    st.session_state['current_map'] = result['map']
                    st.session_state['path_metrics'] = result['metrics']
                    st.session_state['algorithm_used'] = algorithm
                    
                st.success(f"✅ Path found using {algorithm}!")
            except Exception as e:
                st.error(f"❌ Error finding path: {str(e)}")
        
        # Display map
        if st.session_state['current_map']:
            map_data = st_folium(
                st.session_state['current_map'],
                width=700,
                height=500,
                returned_objects=["last_clicked"]
            )
    
    with col2:
        st.subheader("📋 Route Information")
        
        # Display path metrics if available
        if 'path_metrics' in st.session_state:
            metrics = st.session_state['path_metrics']
            algorithm_used = st.session_state.get('algorithm_used', 'Unknown')
            
            st.metric("🚶 Algorithm Used", algorithm_used)
            st.metric("📏 Total Distance", f"{metrics['distance']:.0f} meters")
            st.metric("⏱️ Estimated Time", f"{metrics['time']:.1f} minutes")
            st.metric("🔍 Nodes Explored", f"{metrics['nodes_explored']:,}")
            
            # Building information
            st.subheader("🏢 Location Details")
            
            start_info = pathfinder.get_location_info(metrics['start_location'])
            end_info = pathfinder.get_location_info(metrics['end_location'])
            
            st.write("**Start Location:**")
            st.info(f"📍 {start_info['name']}\n📍 Coordinates: {start_info['coordinates']}")
            
            st.write("**End Location:**")
            st.info(f"🏁 {end_info['name']}\n📍 Coordinates: {end_info['coordinates']}")
        
        # Display AI response if available
        if 'ai_query' in st.session_state:
            with st.spinner("🤖 AI is thinking..."):
                try:
                    ai_response = gemini.process_query(
                        st.session_state['ai_query'],
                        pathfinder.POIS,
                        pathfinder
                    )
                    st.subheader("🤖 AI Assistant Response")
                    st.write(ai_response)
                    del st.session_state['ai_query']  # Clear after processing
                except Exception as e:
                    st.error(f"❌ AI Error: {str(e)}")
    
    # Algorithm comparison results
    if 'run_comparison' in st.session_state:
        st.markdown("---")
        st.subheader("📊 Algorithm Performance Comparison")
        
        with st.spinner("🔬 Running algorithm comparison..."):
            try:
                comparison_results = pathfinder.compare_algorithms()
                
                # Display results table
                df = pd.DataFrame(comparison_results)
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=False
                )
                
                # Performance insights
                st.subheader("💡 Performance Insights")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    fastest_algo = df.loc[df['Average Distance (m)'].idxmin(), 'Algorithm']
                    st.metric("🏆 Shortest Path", str(fastest_algo))
                
                with col2:
                    least_explored = df.loc[df['Average Nodes Explored'].idxmin(), 'Algorithm']
                    st.metric("⚡ Most Efficient", str(least_explored))
                
                with col3:
                    most_explored = df.loc[df['Average Nodes Explored'].idxmax(), 'Algorithm']
                    st.metric("🔍 Most Thorough", str(most_explored))
                
                del st.session_state['run_comparison']  # Clear after processing
                
            except Exception as e:
                st.error(f"❌ Comparison Error: {str(e)}")
    
    # Heuristic comparison results
    if 'run_heuristic_comparison' in st.session_state:
        st.markdown("---")
        st.subheader("🎯 A* Heuristic Comparison")
        
        with st.spinner("🔬 Running heuristic comparison..."):
            try:
                heuristic_results = pathfinder.compare_heuristics()
                
                # Display results table
                df_heuristic = pd.DataFrame(heuristic_results)
                st.dataframe(
                    df_heuristic,
                    use_container_width=True,
                    hide_index=False
                )
                
                # Heuristic insights
                st.subheader("🧠 Heuristic Analysis")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    best_heuristic = df_heuristic.loc[df_heuristic['Average Distance (m)'].idxmin(), 'Heuristic Type']
                    st.metric("🏆 Best Distance", str(best_heuristic))
                
                with col2:
                    most_efficient = df_heuristic.loc[df_heuristic['Average Nodes Explored'].idxmin(), 'Heuristic Type']
                    st.metric("⚡ Most Efficient", str(most_efficient))
                
                with col3:
                    best_efficiency = df_heuristic.loc[df_heuristic['Efficiency Score'].idxmax(), 'Heuristic Type']
                    st.metric("🎯 Best Efficiency Score", str(best_efficiency))
                
                # Detailed analysis
                st.subheader("📈 Detailed Heuristic Analysis")
                
                st.write("**Key Insights:**")
                st.write("• **Euclidean Distance**: Calculates straight-line distance between points")
                st.write("• **Manhattan Distance**: Sum of horizontal and vertical distances")
                st.write("• **Combined Heuristic**: Weighted combination (70% Euclidean + 30% Manhattan)")
                st.write("• **Efficiency Score**: Distance per node explored (higher = better path quality per exploration)")
                
                del st.session_state['run_heuristic_comparison']  # Clear after processing
                
            except Exception as e:
                st.error(f"❌ Heuristic Comparison Error: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>🎓 Interactive Campus Navigation System | Powered by OSM, NetworkX, and Gemini AI</p>
        </div>
        """,
        unsafe_allow_html=True
    )

except Exception as e:
    st.error(f"❌ Application Error: {str(e)}")
    st.info("Please ensure the OSM file is properly loaded and all dependencies are installed.")
