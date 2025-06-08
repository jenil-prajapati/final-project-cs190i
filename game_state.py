from dataclasses import dataclass, field
from typing import List, Dict
from datetime import datetime

@dataclass
class GameState:
    player_name: str = ""
    story_context: List[Dict] = field(default_factory=list)
    emotional_history: List[Dict] = field(default_factory=list)
    
    def add_story_event(self, event_text: str, emotion_data: Dict):
        """Add a new story event with associated emotional data"""
        self.story_context.append({
            'text': event_text,
            'timestamp': datetime.now().isoformat(),
            'emotion': emotion_data['primary_emotion'],
            'intensity': emotion_data['intensity']
        })
        self.emotional_history.append(emotion_data)
    
    def get_story_summary(self) -> str:
        """Get a summary of recent story events"""
        recent_events = self.story_context[-3:]  # Last 3 events
        return "\n".join([event['text'] for event in recent_events])
    
    def get_emotional_context(self) -> Dict:
        """Get the current emotional context for story generation"""
        if not self.emotional_history:
            return {'primary_emotion': 'neutral', 'intensity': 0.5}
        return self.emotional_history[-1]