import time
import torch
import gc
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv
from typing import Dict

class EmotionAnalyzer:
    """
    Local emotion analyzer using j-hartmann/emotion-english-distilroberta-base model
    """
    def __init__(self):
        # Load environment variables (for any future needs)
        load_dotenv()
        
        # Model configuration
        self.model_name = "j-hartmann/emotion-english-distilroberta-base"
        self.emotion_labels = [
            'anger', 'disgust', 'fear', 'joy', 
            'neutral', 'sadness', 'surprise'
        ]
        
        print("Loading emotion detection model...")
        start_time = time.time()
        
        try:
            # Free memory before loading model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Load tokenizer and model with optimizations
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                low_cpu_mem_usage=True
            )
            
            # Put model in evaluation mode and move to CPU
            self.model.eval()
            self.model.to('cpu')
            
            print(f"Emotion model loaded in {time.time() - start_time:.2f} seconds")
            
        except Exception as e:
            print(f"Error loading emotion model: {e}")
            self.model = None

    def analyze_emotion(self, text: str, game_state=None) -> Dict:
        """
        Analyze text and return emotion data in the format expected by the app
        Returns:
            {
                "primary_emotion": str, 
                "intensity": float,
                "confidence": float,
                "prompt_addition": str
            }
        """
        if not self.model or not text:
            return {
                "primary_emotion": "neutral",
                "intensity": 0.5,
                "confidence": 0.7,
                "prompt_addition": ""
            }
            
        try:
            # Tokenize and process input
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Get predicted emotion
            probabilities = torch.softmax(outputs.logits, dim=1)
            confidence, pred_label = torch.max(probabilities, dim=1)
            
            emotion = self.emotion_labels[pred_label]
            
            # Map intensity based on emotion type
            intensity_map = {
                'anger': 0.8,
                'disgust': 0.7,
                'fear': 0.9,
                'joy': 0.6,
                'neutral': 0.5,
                'sadness': 0.7,
                'surprise': 0.8
            }
            
            intensity = intensity_map.get(emotion, 0.5) * confidence.item()
            
            return {
                "primary_emotion": emotion,
                "intensity": intensity,
                "confidence": confidence.item(),
                "prompt_addition": f"The character feels {emotion}.",
                "secondary_emotions": {}
            }
            
        except Exception as e:
            print(f"Error analyzing emotion: {e}")
            return {
                "primary_emotion": "neutral",
                "intensity": 0.5,
                "confidence": 0.7,
                "prompt_addition": "",
                "secondary_emotions": {}
            }

    def get_emotion_prompt_addition(self, emotion_data: Dict) -> str:
        """Generate a natural language description of the emotion for prompts"""
        if not emotion_data or emotion_data.get("primary_emotion") == "neutral":
            return ""
        return f"They feel {emotion_data['primary_emotion']}."