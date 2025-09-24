import os
import json
from typing import Dict, Any, Tuple

class GeminiAssistant:
    def __init__(self):
        """Initialize Gemini AI assistant."""
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.client = None
        
        # Try to initialize Gemini client
        if self.api_key:
            try:
                from google import genai
                from google.genai import types
                self.client = genai.Client(api_key=self.api_key)
                self.model = "gemini-2.5-flash"
                self.genai = genai
                self.types = types
            except ImportError:
                print("Google Gemini AI not available - using fallback responses")
                self.client = None
    
    def process_query(self, user_query: str, pois: Dict[str, Tuple[float, float]], pathfinder) -> str:
        """Process user query and provide intelligent response."""
        try:
            # Create context about the campus
            campus_context = self._create_campus_context(pois)
            
            # Determine if this is a route query or general query
            is_route_query = self._is_route_query(user_query)
            
            if is_route_query:
                return self._handle_route_query(user_query, pois, pathfinder, campus_context)
            else:
                return self._handle_general_query(user_query, campus_context)
        
        except Exception as e:
            return f"I apologize, but I encountered an error processing your query: {str(e)}. Please try rephrasing your question."
    
    def _create_campus_context(self, pois: Dict[str, Tuple[float, float]]) -> str:
        """Create context about campus locations."""
        context = "Campus Locations and Information:\n\n"
        
        # Categorize locations
        categories = {
            'Academic': ['Acad 1', 'Acad 2', 'Library'],
            'Facilities': ['Food Court', 'Faculty Block', 'Hostel Block'],
            'Sports': ['Cricket Ground', 'Basket Ball', 'Volley Ball', 'Tennis Ball', 'Foot Ball'],
            'Security/Access': ['Entry gate', 'Exit gate', 'Check post 1', 'Check post 2'],
            'Other': ['Flag post', 'Rest Area']
        }
        
        for category, locations in categories.items():
            context += f"{category}:\n"
            for loc in locations:
                if loc in pois:
                    lat, lon = pois[loc]
                    context += f"  - {loc}: ({lat:.5f}, {lon:.5f})\n"
            context += "\n"
        
        context += """
        Available pathfinding algorithms:
        - BFS (Breadth-First Search): Explores all nodes at current depth before moving deeper
        - DFS (Depth-First Search): Explores as far as possible along each branch
        - UCS (Uniform Cost Search): Finds minimum cost path considering edge weights
        - A*: Uses heuristic to find optimal path efficiently
        
        The campus map includes walking paths, roads, and various points of interest.
        Walking time is estimated at 1.4 meters per second average speed.
        """
        
        return context
    
    def _is_route_query(self, query: str) -> bool:
        """Determine if query is asking for route information."""
        route_keywords = [
            'route', 'path', 'way', 'get to', 'go to', 'from', 'to',
            'direction', 'navigate', 'walk', 'distance', 'how far'
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in route_keywords)
    
    def _handle_route_query(self, query: str, pois: Dict[str, Tuple[float, float]], 
                          pathfinder, campus_context: str) -> str:
        """Handle route-related queries."""
        try:
            # If Gemini is available, use AI to extract locations
            if self.client:
                return self._handle_route_query_with_ai(query, pois, pathfinder)
            else:
                # Fallback: simple keyword matching
                return self._handle_route_query_fallback(query, pois, pathfinder)
        except Exception as e:
            return f"I encountered an error processing your route query: {str(e)}. Please try using the manual pathfinding controls above."
    
    def _handle_route_query_with_ai(self, query: str, pois: Dict[str, Tuple[float, float]], pathfinder) -> str:
        """Handle route queries using AI."""
        try:
            extraction_prompt = f"""
            Based on this user query about campus navigation: "{query}"
            
            And these available campus locations:
            {', '.join(pois.keys())}
            
            Extract the start and end locations the user is asking about.
            If locations are not explicitly mentioned but can be inferred, suggest the most likely ones.
            
            Respond with JSON in this format:
            {{"start": "location_name", "end": "location_name", "confidence": "high/medium/low"}}
            
            If you cannot determine locations, set both to null.
            """
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=extraction_prompt
            )
            
            location_data = json.loads(response.text)
            start_loc = location_data.get('start')
            end_loc = location_data.get('end')
            
            if start_loc and end_loc and start_loc in pois and end_loc in pois:
                route_result = pathfinder.find_path(start_loc, end_loc, "A*")
                metrics = route_result['metrics']
                
                return f"""
🗺️ **Route Information: {start_loc} → {end_loc}**

📏 **Distance:** {metrics['distance']:.0f} meters
⏱️ **Estimated Walking Time:** {metrics['time']:.1f} minutes
🧠 **Algorithm Used:** A* (optimal pathfinding)
🔍 **Nodes Explored:** {metrics['nodes_explored']:,}

📍 **Start Location:** {start_loc}
🏁 **End Location:** {end_loc}

💡 **Additional Tips:**
- The route shown on the map uses the most efficient path
- Walking time assumes average speed of 1.4 m/s
- Consider checking for any temporary obstacles or construction
- Use the map visualization above to see the exact path

Would you like me to compare different algorithms for this route or provide information about other locations?
                """
            else:
                return self._provide_location_help(pois)
                
        except Exception as e:
            return self._handle_route_query_fallback(query, pois, pathfinder)
    
    def _handle_route_query_fallback(self, query: str, pois: Dict[str, Tuple[float, float]], pathfinder) -> str:
        """Fallback route query handler using simple keyword matching."""
        query_lower = query.lower()
        
        # Simple location extraction
        found_locations = [loc for loc in pois.keys() if loc.lower() in query_lower]
        
        if len(found_locations) >= 2:
            start_loc, end_loc = found_locations[0], found_locations[1]
            try:
                route_result = pathfinder.find_path(start_loc, end_loc, "A*")
                metrics = route_result['metrics']
                
                return f"""
🗺️ **Route Information: {start_loc} → {end_loc}**

📏 **Distance:** {metrics['distance']:.0f} meters
⏱️ **Estimated Walking Time:** {metrics['time']:.1f} minutes
🧠 **Algorithm Used:** A* (optimal pathfinding)

💡 **Note:** This is a basic response. For more intelligent responses, please ensure the AI service is properly configured.
                """
            except Exception as e:
                return f"Found locations {start_loc} and {end_loc}, but encountered an error: {str(e)}"
        
        return self._provide_location_help(pois)
    
    def _provide_location_help(self, pois: Dict[str, Tuple[float, float]]) -> str:
        """Provide help with available locations."""
        return f"""
I understand you're asking about navigation, but I need clarification on the specific locations.

📍 **Available Campus Locations:**
{self._format_location_list(pois)}

Could you please specify your start and end locations more clearly? For example:
- "How do I get from Library to Food Court?"
- "What's the distance from Entry gate to Cricket Ground?"

Or use the dropdown menus above to select your route manually.
        """
    
    def _handle_general_query(self, query: str, campus_context: str) -> str:
        """Handle general campus information queries."""
        if self.client:
            return self._handle_general_query_with_ai(query, campus_context)
        else:
            return self._handle_general_query_fallback(query, campus_context)
    
    def _handle_general_query_with_ai(self, query: str, campus_context: str) -> str:
        """Handle general queries with AI."""
        try:
            system_prompt = f"""
            You are a helpful campus navigation assistant. Use the following campus information to answer questions:
            
            {campus_context}
            
            Provide helpful, accurate information about campus locations, facilities, and navigation.
            Be friendly and informative. If asked about specific locations, provide details about their purpose and location.
            """
            
            full_prompt = f"{system_prompt}\n\nUser Question: {query}"
            response = self.client.models.generate_content(
                model=self.model,
                contents=full_prompt
            )
            
            if response.text:
                return f"🤖 **Campus Assistant Response:**\n\n{response.text}"
            else:
                return "I apologize, but I couldn't generate a response to your query. Please try rephrasing your question."
        
        except Exception as e:
            return self._handle_general_query_fallback(query, campus_context)
    
    def _handle_general_query_fallback(self, query: str, campus_context: str) -> str:
        """Fallback for general queries."""
        query_lower = query.lower()
        
        # Simple responses for common queries
        if 'food' in query_lower or 'eat' in query_lower:
            return "🍽️ The **Food Court** is available for dining. It's located at coordinates (13.22488, 77.75716). Use the pathfinding tool above to get directions!"
        
        elif 'library' in query_lower or 'book' in query_lower or 'study' in query_lower:
            return "📚 The **Library** is perfect for studying and research. It's located at coordinates (13.22199, 77.75540). Use the route planner to find the best path!"
        
        elif 'sports' in query_lower or 'game' in query_lower or 'play' in query_lower:
            return """🏃‍♂️ We have several sports facilities:
- **Cricket Ground** - Main cricket field
- **Basketball Court** - For basketball games
- **Volleyball Court** - For volleyball matches
- **Tennis Court** - Tennis facility
- **Football Field** - For football/soccer

Use the location selector above to get directions to any of these facilities!"""
        
        elif 'academic' in query_lower or 'class' in query_lower:
            return """🎓 Academic facilities include:
- **Acad 1** - Main academic building 1
- **Acad 2** - Academic building 2
- **Faculty Block** - Faculty offices and departments

Select any of these in the route planner to find your way!"""
        
        else:
            return f"""
🤖 **Campus Assistant (Basic Mode)**

I can help you with information about our campus locations! Here are the main categories:

{self._format_location_list({loc: coords for loc, coords in []})}

**Available Services:**
- Route planning between any locations
- Distance and time calculations
- Multiple pathfinding algorithms (BFS, DFS, UCS, A*)

Use the controls above to plan your route, or ask me specific questions about campus locations!

💡 **Note:** Enhanced AI responses are temporarily unavailable. Basic assistance is provided.
            """
    
    def _format_location_list(self, pois: Dict[str, Tuple[float, float]]) -> str:
        """Format POI list for display."""
        categories = {
            'Academic': ['Acad 1', 'Acad 2', 'Library'],
            'Facilities': ['Food Court', 'Faculty Block', 'Hostel Block'],
            'Sports': ['Cricket Ground', 'Basket Ball', 'Volley Ball', 'Tennis Ball', 'Foot Ball'],
            'Security/Access': ['Entry gate', 'Exit gate', 'Check post 1', 'Check post 2'],
            'Other': ['Flag post', 'Rest Area']
        }
        
        all_locations = ['Flag post', 'Entry gate', 'Exit gate', 'Check post 1', 'Check post 2',
                        'Acad 1', 'Acad 2', 'Library', 'Food Court', 'Faculty Block', 
                        'Hostel Block', 'Cricket Ground', 'Basket Ball', 'Volley Ball',
                        'Tennis Ball', 'Foot Ball', 'Rest Area']
        
        formatted_list = ""
        for category, locations in categories.items():
            formatted_list += f"\n**{category}:** {', '.join(locations)}"
        
        return formatted_list