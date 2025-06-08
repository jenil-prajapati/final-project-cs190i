import os
import time
import random
import google.generativeai as genai
from dotenv import load_dotenv
from typing import Dict, List
from datetime import datetime

class StoryEngine:
    """
    Cloud-based story engine using Gemini for interactive storytelling
    """
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Initialize Gemini
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        # Use gemini-2.0-flash-lite model
        self.model = genai.GenerativeModel('gemini-2.0-flash-lite')
        
        # Generation parameters
        self.generation_config = {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_output_tokens": 800
        }
        
        # API call tracking
        self.api_calls = {
            'count': 0,
            'today': 0,
            'last_reset': datetime.now().date(),
            'history': []
        }
        
        print("Initialized StoryEngine with Gemini 2.0 Flash Lite")
        self.model_loaded = True  # Always true for cloud model

    def _construct_prompt(self, player_input: str, emotion_data: Dict, game_state) -> str:
        """Build the prompt for story generation"""
        player_name = getattr(game_state, 'player_name', 'adventurer')
        difficulty = getattr(game_state, 'difficulty_level', 0.5)
        emotion_label = emotion_data.get('primary_emotion', 'neutral')
        emotion_prompt = emotion_data.get('prompt_addition', '')
        
        # Get recent story context if available
        recent_context = ""
        if hasattr(game_state, 'get_story_summary') and callable(getattr(game_state, 'get_story_summary')):
            try:
                recent_context = game_state.get_story_summary()
                if recent_context:
                    recent_context = f"Recent events:\n{recent_context}\n\n"
            except:
                pass
        
        return f"""You are an expert storyteller creating an interactive fantasy adventure. Maintain the setting and tone at all times.

        Player name: {player_name}
        Player emotion: {emotion_label}
        Difficulty: {difficulty}
        
        {recent_context}Guidelines:
        - ALWAYS use the name {player_name} throughout the narrative
        - Keep responses to 2-3 paragraphs
        - ALWAYS end with at least 2 numbered choices (1, 2, 3, 4)
        - Make sure each choice is distinct and interesting
        - Adapt tone to match player emotion: {emotion_label}
        - Maintain fantasy setting and atmosphere
        - Never break character or mention AI

        The player character {player_name} feels {emotion_label}. {emotion_prompt}

        {player_name} decides to: {player_input}

        What happens next in the story? Remember to end with at least 2 numbered choices."""

    def _generate_mock_response(self, prompt: str) -> str:
        """Fallback mock response generator"""
        mock_responses = [
            "As you proceed, unexpected events unfold... (mock response)",
            "Your action leads to interesting developments... (mock response)",
            "The story takes an unexpected turn... (mock response)"
        ]
        return random.choice(mock_responses)

    def _track_api_call(self, call_type):
        """Track API call usage"""
        today = datetime.now().date()
        
        # Reset daily counter if it's a new day
        if today != self.api_calls['last_reset']:
            self.api_calls['today'] = 0
            self.api_calls['last_reset'] = today
        
        # Update counters
        self.api_calls['count'] += 1
        self.api_calls['today'] += 1
        
        # Add to history
        self.api_calls['history'].append({
            'timestamp': datetime.now().isoformat(),
            'type': call_type
        })
        
        # Print usage stats
        print(f"[API USAGE] Total calls: {self.api_calls['count']}, Today: {self.api_calls['today']}")

    def get_api_usage(self):
        """Get API usage statistics"""
        return self.api_calls

    def generate_story_continuation(self, player_input: str, emotion_data: Dict, game_state) -> str:
        """Generate a continuation of the story based on player input and emotional state"""
        if not player_input:
            player_input = "look around cautiously"
            
        prompt = self._construct_prompt(player_input, emotion_data, game_state)
        
        print("\n[DEBUG] Calling Gemini API for story continuation...")
        print(f"[DEBUG] Emotion: {emotion_data.get('primary_emotion', 'neutral')}")
        
        try:
            # Track API call
            self._track_api_call('story_continuation')
            
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            print("[DEBUG] Successfully received response from Gemini API")
            return response.text
        except Exception as e:
            print(f"Error generating story: {e}")
            return self._generate_mock_response(prompt)

    def generate_initial_scenario(self, player_name="Adventurer") -> str:
        """Generate the starting scenario for the game with player name"""
        prompt = f"""Create an immersive opening scenario for an epic fantasy adventure. Your response should include:
        - A detailed, atmospheric description of the setting with sensory details (sights, sounds, smells)
        - A mysterious situation or challenge that creates immediate intrigue
        - Background on why the character named {player_name} is in this situation
        - At least 2 distinct numbered choices for how to proceed
        - End with a clear question asking what {player_name} will do next
        - At most 400-500 words total
        
        Make the scenario feel like the opening of a professional fantasy novel with rich worldbuilding.
        IMPORTANT: Use the name {player_name} throughout the narrative.
        IMPORTANT: End with at least 2 numbered choices (1, 2, 3, 4).
        """
        
        print(f"\n[DEBUG] Calling Gemini API for initial scenario generation for {player_name}...")
        try:
            # Track API call
            self._track_api_call('initial_scenario')
            
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.8,
                    "top_p": 0.95,
                    "max_output_tokens": 800
                }
            )
            print("[DEBUG] Successfully received initial scenario from Gemini API")
            
            # Extract location from the response
            content = response.text
            
            # Return the generated content
            return content
        except Exception as e:
            print(f"Error generating initial scenario: {e}")
            return """The ancient forest of Eldermist envelops you in its mysterious embrace. Towering trees with gnarled trunks stretch toward the sky, their dense canopy filtering the sunlight into ethereal beams that dance across the moss-covered ground. The air is thick with the scent of damp earth and exotic flowers, while distant bird calls echo through the woods, occasionally interrupted by sounds you cannot identify.

            You've come seeking the legendary Crystal of Azoria, said to grant visions of possible futures to those pure of heart. The village elder entrusted you with this quest after strange omens began appearing in the night sky. According to ancient texts, the crystal lies hidden in a temple deep within these woods, but many adventurers have entered Eldermist never to return.

            As you pause to consult your weathered map, a bone-chilling howl pierces the silence, followed by the sound of heavy footfalls approaching from the north. At the same moment, you notice a faint blue glow emanating from between the trees to the east, while a barely visible path to the west seems to lead deeper into the forest.

            What do you do?

            1) Draw your weapon and prepare to confront whatever approaches from the north
            2) Move quietly toward the mysterious blue glow to the east
            3) Take the hidden path to the west, avoiding potential danger
            4) Climb one of the massive trees to gain a better vantage point and assess the situation"""