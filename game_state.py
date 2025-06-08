from dataclasses import dataclass, field
from typing import List, Dict
from datetime import datetime

@dataclass
class GameState:
    player_name: str = ""
    inventory: List[str] = field(default_factory=list)
    story_context: List[Dict] = field(default_factory=list)
    emotional_history: List[Dict] = field(default_factory=list)
    difficulty_level: float = 0.5  # 0.0 to 1.0
    
    def add_story_event(self, event_text: str, emotion_data: Dict):
        """Add a new story event with associated emotional data"""
        self.story_context.append({
            'text': event_text,
            'timestamp': datetime.now().isoformat(),
            'emotion': emotion_data['primary_emotion'],
            'intensity': emotion_data['intensity']
        })
        self.emotional_history.append(emotion_data)
        
        # Adjust difficulty based on emotional state
        self._adjust_difficulty(emotion_data)
    
    def _adjust_difficulty(self, emotion_data: Dict):
        """Dynamically adjust difficulty based on player's emotional state"""
        # Increase difficulty if player is confident/happy
        if emotion_data['primary_emotion'] in ['confident', 'happy']:
            self.difficulty_level = min(1.0, self.difficulty_level + 0.1)
        # Decrease difficulty if player is fearful/uncertain
        elif emotion_data['primary_emotion'] in ['fearful', 'uncertain']:
            self.difficulty_level = max(0.0, self.difficulty_level - 0.1)
    
    def get_story_summary(self, format_type='default') -> str:
        """
        Get a summary of story events
        format_type: 
            - 'default': Last 3 events (original behavior)
            - 'sentiment': Formatted specifically for sentiment analysis
            - 'full': All events
        """
        if not self.story_context:
            return ""
            
        if format_type == 'full':
            return "\n".join([event['text'] for event in self.story_context])
            
        elif format_type == 'sentiment':
            recent_count = min(3, len(self.story_context))
            recent_events = self.story_context[-recent_count:]
            
            return "\n".join([event['text'] for event in recent_events])
            
        else:
            recent_events = self.story_context[-3:] if len(self.story_context) >= 3 else self.story_context
            return "\n".join([event['text'] for event in recent_events])
    
    def get_emotional_context(self) -> Dict:
        """Get the current emotional context for story generation"""
        if not self.emotional_history:
            return {'primary_emotion': 'neutral', 'intensity': 0.5}
        return self.emotional_history[-1] 