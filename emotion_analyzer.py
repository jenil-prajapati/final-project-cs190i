import time
import torch
import gc
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv
from typing import Dict
from story_summarizer import StorySummarizer

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
            
            # Initialize the story summarizer for context-aware analysis
            self.summarizer = StorySummarizer()
            
            # Put model in evaluation mode and move to CPU
            self.model.eval()
            self.model.to('cpu')
            
            print(f"Emotion model loaded in {time.time() - start_time:.2f} seconds")
            
        except Exception as e:
            print(f"Error loading emotion model: {e}")
            self.model = None

    def analyze_emotion(self, text: str, game_state=None, context_aware=False) -> Dict:
        """
        Analyze text and return emotion data in the format expected by the app.
        
        Args:
            text: The user input text to analyze
            game_state: Optional game state for context-aware analysis
            context_aware: Whether to use context-aware analysis or just the input text
            
        Returns:
            {
                "primary_emotion": str, 
                "intensity": float,
                "confidence": float,
                "prompt_addition": str,
                "secondary_emotions": {}
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
            # Always analyze the raw user input first for comparison
            baseline_result = self._analyze_text(text)
            print(f"\n[DEBUG] BASELINE SENTIMENT (user input only): {baseline_result['primary_emotion']} "
                  f"(intensity: {baseline_result['intensity']:.2f}, confidence: {baseline_result['confidence']:.2f})")
            
            # If context-aware is not enabled, return the baseline result
            if not context_aware or not game_state or not hasattr(game_state, 'get_story_summary'):
                return baseline_result
                
            # Get story context formatted specifically for sentiment analysis
            try:
                story_context = game_state.get_story_summary(format_type='sentiment')
                
                if story_context:
                    analysis_text = text
                    
                    if self.summarizer:
                        # Use the summarizer to create a balanced summary for sentiment analysis
                        analysis_text = self.summarizer.summarize_for_sentiment(story_context, text)
                        print(f"[DEBUG] Using summarizer for context-aware sentiment analysis")
                    else:
                        # Fallback to simple combination if summarizer isn't available
                        analysis_text = f"Recent story: {story_context}\n\nUser input: {text}"
                        print(f"[DEBUG] Using basic context-aware sentiment analysis")
                        
                    # Analyze the context-aware text
                    context_result = self._analyze_text(analysis_text)
                    print(f"[DEBUG] CONTEXT-AWARE SENTIMENT: {context_result['primary_emotion']} "
                          f"(intensity: {context_result['intensity']:.2f}, confidence: {context_result['confidence']:.2f})")
                    
                    return context_result
                else:
                    return baseline_result
            except Exception as e:
                print(f"[WARNING] Error getting story context: {e}")
                return baseline_result
        except Exception as e:
            print(f"Error analyzing emotion: {e}")
            return {
                "primary_emotion": "neutral",
                "intensity": 0.5,
                "confidence": 0.7,
                "prompt_addition": "",
                "secondary_emotions": {}
            }

    def _analyze_text(self, text: str) -> Dict:
        """
        Helper method to analyze text and return emotion data
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
            
            # Get probabilities
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Get emotion scores
            emotion_scores = {}
            for i, emotion in enumerate(self.emotion_labels):
                emotion_scores[emotion] = probs[0][i].item()
            
            # Get primary emotion and intensity
            primary_emotion = max(emotion_scores, key=emotion_scores.get)
            intensity = emotion_scores[primary_emotion]
            
            # Calculate confidence as difference between top emotion and average of others
            other_emotions = {e: s for e, s in emotion_scores.items() if e != primary_emotion}
            avg_other_score = sum(other_emotions.values()) / len(other_emotions) if other_emotions else 0
            confidence = intensity - avg_other_score
            
            # Get secondary emotions (those with scores > 0.1)
            secondary_emotions = {e: s for e, s in emotion_scores.items() 
                                if e != primary_emotion and s > 0.1}
            
            # Generate prompt addition based on emotion
            prompt_addition = self.get_emotion_prompt_addition({"primary_emotion": primary_emotion, "intensity": intensity})
            
            return {
                "primary_emotion": primary_emotion,
                "intensity": float(intensity),
                "confidence": float(confidence),
                "prompt_addition": prompt_addition,
                "secondary_emotions": secondary_emotions
            }
        except Exception as e:
            print(f"Error in _analyze_text: {e}")
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