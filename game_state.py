from dataclasses import dataclass, field
from typing import List, Dict
from datetime import datetime

class GameState:
    def __init__(self):
        self.player_name = ""
        self.context_type = "neutral"
        self.narrative_tone = "neutral"
        self.context_intensity = 0.5
        self.current_description = ""
        self.story_events = []
        self.emotional_state = {
            "current_emotion": "neutral",
            "intensity": 0.5,
            "confidence": 0.5
        }
    
    @property
    def player_name(self):
        return self._player_name

    @player_name.setter
    def player_name(self, value):
        self._player_name = value

    @property
    def context_type(self):
        return self._context_type

    @context_type.setter
    def context_type(self, value):
        self._context_type = value

    @property
    def narrative_tone(self):
        return self._narrative_tone

    @narrative_tone.setter
    def narrative_tone(self, value):
        self._narrative_tone = value

    @property
    def context_intensity(self):
        return self._context_intensity

    @context_intensity.setter
    def context_intensity(self, value):
        self._context_intensity = float(value)

    @property
    def current_description(self):
        return self._current_description

    @current_description.setter
    def current_description(self, value):
        self._current_description = value

    def add_story_event(self, event_text, emotion_data=None):
        if emotion_data is None:
            emotion_data = {
                "emotion": "neutral",
                "intensity": 0.5,
                "confidence": 0.5
            }
        
        self.story_events.append({
            "text": event_text,
            "emotion": emotion_data
        })

    def update_emotional_state(self, emotion_data):
        self.emotional_state = emotion_data

    def get_story_summary(self) -> str:
        """Get a summary of recent story events"""
        recent_events = self.story_events[-3:]  # Last 3 events
        return "\n".join([event['text'] for event in recent_events])
    
    def get_emotional_context(self) -> Dict:
        """Get the current emotional context for story generation"""
        if not self.story_events:
            return {'primary_emotion': 'neutral', 'intensity': 0.5}
        return self.story_events[-1]['emotion']