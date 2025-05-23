import os
from openai import OpenAI
from decouple import config
from typing import Dict, List

class StoryEngine:
    def __init__(self):
        self.client = OpenAI(api_key=config('OPENAI_API_KEY'))
        self.system_prompt = """You are an expert dungeon master creating an immersive, 
        interactive story experience. Adapt the narrative based on the player's emotional 
        state and actions. Keep responses concise but engaging."""
    
    def generate_story_continuation(self, 
                                  player_input: str, 
                                  emotion_data: Dict, 
                                  game_state: 'GameState') -> str:
        # Construct the prompt based on game state and emotional context
        prompt = self._construct_prompt(player_input, emotion_data, game_state)
        
        # Generate response using GPT
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=150
        )
        
        return response.choices[0].message.content
    
    def _construct_prompt(self, 
                         player_input: str, 
                         emotion_data: Dict, 
                         game_state: 'GameState') -> str:
        # Get recent story context
        story_summary = game_state.get_story_summary()
        
        # Construct prompt with emotional context
        prompt = f"""
Context: {story_summary}

Player's current emotion: {emotion_data['primary_emotion']}
Emotion intensity: {emotion_data['intensity']}
Current difficulty level: {game_state.difficulty_level}

Player action: {player_input}

Generate a response that:
1. Acknowledges the player's emotional state
2. Continues the story in an engaging way
3. Matches the current difficulty level
4. Provides clear options for the player's next action

Response:"""
        
        return prompt
    
    def generate_initial_scenario(self) -> str:
        prompt = """Create an opening scenario for a fantasy adventure. 
        Set the scene and present the player with their first choice."""
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=150
        )
        
        return response.choices[0].message.content 