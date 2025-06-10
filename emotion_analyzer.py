import time
import torch
import gc
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv
from typing import Dict
from context_feature_extractor import ContextFeatureExtractor

class EmotionAnalyzer:
    """
    Context-aware emotion analyzer using j-hartmann/emotion-english-distilroberta-base model
    with contextual adjustment for gaming scenarios
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
            
            # Initialize context feature extractor if available
            self.context_extractor = None
            if ContextFeatureExtractor:
                try:
                    self.context_extractor = ContextFeatureExtractor()
                    print("Context feature extractor initialized")
                except Exception as e:
                    print(f"Error initializing context extractor: {e}")
                    self.context_extractor = None
            
        except Exception as e:
            print(f"Error loading emotion model: {e}")
            self.model = None
            self.context_extractor = None

    def analyze_emotion(self, text: str, game_state=None, context_aware=True) -> Dict:
        """
        Analyze text with context-awareness and return emotion data in the format expected by the app
        
        Parameters:
            text (str): User input text to analyze
            game_state: Game state object containing story context
            context_aware (bool): Whether to use context-aware analysis
        
        Returns:
            {
                "primary_emotion": str, 
                "intensity": float,
                "confidence": float,
                "prompt_addition": str,
                "secondary_emotions": dict,
                "context_applied": bool (optional),
                "context_type": str (optional)
            }
        """
        if not self.model or not text:
            return {
                "primary_emotion": "neutral",
                "intensity": 0.5,
                "confidence": 0.7,
                "prompt_addition": "",
                "secondary_emotions": {}
            }
            
        try:
            # Get baseline emotion using DistilRoBERTa
            baseline_emotion = self._get_baseline_emotion(text)
            print(f"[SENTIMENT ANALYSIS] Base Sentiment - Emotion: {baseline_emotion['primary_emotion']}, Intensity: {baseline_emotion['intensity']:.2f}, Confidence: {baseline_emotion['confidence']:.2f}")
            
            # If context_aware is False, no game_state, or no context extractor, return baseline
            if not context_aware or not game_state or not self.context_extractor:
                return baseline_emotion
                
            # Extract context features
            context_features = self.context_extractor.extract_features(game_state)
            
            # Adjust emotion based on context
            adjusted_emotion = self._adjust_emotion_with_context(baseline_emotion, context_features)
            print(f"[SENTIMENT ANALYSIS] Context-Adjusted Sentiment - Emotion: {adjusted_emotion['primary_emotion']}, Intensity: {adjusted_emotion['intensity']:.2f}, Confidence: {adjusted_emotion['confidence']:.2f}, Context Type: {adjusted_emotion.get('context_type', 'N/A')}, Narrative Tone: {adjusted_emotion.get('narrative_tone', 'N/A')}")
            
            return adjusted_emotion
            
        except Exception as e:
            print(f"Error analyzing emotion: {e}")
            return {
                "primary_emotion": "neutral",
                "intensity": 0.5,
                "confidence": 0.7,
                "prompt_addition": "",
                "secondary_emotions": {}
            }

    def _get_baseline_emotion(self, text: str) -> Dict:
        """
        Get the raw emotion analysis without context adjustment
        """
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
            print(f"Error in baseline emotion analysis: {e}")
            return {
                "primary_emotion": "neutral",
                "intensity": 0.5,
                "confidence": 0.7,
                "prompt_addition": "",
                "secondary_emotions": {}
            }

    def _adjust_emotion_with_context(self, baseline_emotion: Dict, context_features: Dict) -> Dict:
        adjusted_emotion = baseline_emotion.copy()
        context_type = context_features.get("context_type", "neutral")
        narrative_tone = context_features.get("narrative_tone", "neutral")
        context_intensity = context_features.get("intensity", 0.5)
        context_confidence = context_features.get("context_confidence", 0.0)
        
        # Emotion transformation rules based on context and tone
        emotion_transforms = {
            "danger": {
                "joy": {
                    "negative": "fear",     # Joy in danger + negative tone -> fear
                    "tense": "fear",        # Joy in danger + tense -> fear
                    "dark": "fear",         # Joy in danger + dark -> fear
                    "default": "surprise"    # Default transformation for joy in danger
                },
                "neutral": {
                    "negative": "fear",
                    "tense": "fear",
                    "dark": "fear",
                    "default": "fear"
                },
                "surprise": {
                    "negative": "fear",
                    "tense": "fear",
                    "dark": "fear",
                    "default": "fear"
                }
            },
            "combat": {
                "joy": {
                    "negative": "anger",
                    "tense": "fear",
                    "dark": "fear",
                    "default": "anger"
                },
                "neutral": {
                    "negative": "fear",
                    "tense": "fear",
                    "dark": "fear",
                    "default": "anger"
                }
            }
        }
        
        # Get the base emotion and its transformation rules
        primary_emotion = adjusted_emotion["primary_emotion"]
        base_intensity = adjusted_emotion["intensity"]
        base_confidence = adjusted_emotion["confidence"]
        
        # Transform emotion based on context and tone if applicable
        if context_type in emotion_transforms and primary_emotion in emotion_transforms[context_type]:
            transform_rules = emotion_transforms[context_type][primary_emotion]
            # Apply specific tone transformation or default
            new_emotion = transform_rules.get(narrative_tone, transform_rules.get("default", primary_emotion))
            adjusted_emotion["primary_emotion"] = new_emotion
        
        # Intensity adjustment factors
        intensity_factors = {
            "combat": {
                "anger": 1.3,
                "fear": 1.2,
                "sadness": 0.8,
                "joy": 0.6,
                "surprise": 1.1,
                "disgust": 1.2,
                "neutral": 0.7,
                "tones": {
                    "heroic": {"anger": 1.5, "fear": 1.1, "joy": 0.8},
                    "dark": {"anger": 1.4, "fear": 1.3, "sadness": 1.0},
                    "mysterious": {"fear": 1.3, "surprise": 1.2}
                }
            },
            "danger": {
                "anger": 1.2,
                "fear": 1.3,  # Increased fear intensity in danger
                "sadness": 0.9,
                "joy": 0.5,
                "surprise": 1.0,
                "disgust": 1.1,
                "neutral": 0.6,
                "tones": {
                    "heroic": {"anger": 1.4, "fear": 1.0},  
                    "dark": {"fear": 1.4, "anger": 1.3},    # Increased fear in dark tone
                    "tense": {"fear": 1.4, "surprise": 1.1}, # Increased fear in tense tone
                    "negative": {"fear": 1.4, "anger": 1.2}  # Added negative tone adjustments
                }
            }
        }
        
        # Get the adjustment factor for the current emotion (after transformation)
        current_emotion = adjusted_emotion["primary_emotion"]
        adjustment_factor = intensity_factors.get(context_type, {}).get(current_emotion, 0.8)
        
        # Apply tone-specific adjustments if available
        tone_adjustments = intensity_factors.get(context_type, {}).get("tones", {}).get(narrative_tone, {}).get(current_emotion)
        if tone_adjustments is not None:
            adjustment_factor = tone_adjustments
        
        # Calculate adjusted intensity
        adjusted_intensity = base_intensity * adjustment_factor
        blended_intensity = (adjusted_intensity * 0.6) + (context_intensity * 0.4)
        adjusted_emotion["intensity"] = max(0.1, min(1.0, blended_intensity))
        
        # Adjust confidence based on transformation
        if adjusted_emotion["primary_emotion"] != primary_emotion:
            # Lower confidence when emotion is transformed
            adjusted_confidence = (base_confidence * 0.6) + (context_confidence * 0.4)
        else:
            adjusted_confidence = (base_confidence * 0.7) + (context_confidence * 0.3)
        
        adjusted_emotion["confidence"] = max(0.1, min(1.0, adjusted_confidence))
        adjusted_emotion["context_type"] = context_type
        adjusted_emotion["narrative_tone"] = narrative_tone
        
        # Add explanation of transformation
        if adjusted_emotion["primary_emotion"] != primary_emotion:
            adjusted_emotion["transformation_explanation"] = f"Emotion transformed from {primary_emotion} to {adjusted_emotion['primary_emotion']} due to {context_type} context and {narrative_tone} tone"
        
        return adjusted_emotion

    def get_emotion_prompt_addition(self, emotion_data: Dict) -> str:
        """Generate a natural language description of the emotion for prompts"""
        if not emotion_data or emotion_data.get("primary_emotion") == "neutral":
            return ""
        
        base_text = f"They feel {emotion_data['primary_emotion']}."
        
        if emotion_data.get("context_applied", False):
            if 'prompt_addition' in emotion_data and emotion_data['prompt_addition']:
                return emotion_data['prompt_addition']
        
        return base_text