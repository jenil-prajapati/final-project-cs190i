import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Dict, List, Optional

class StorySummarizer:
    """
    A lightweight summarizer that combines story context and user prompt
    with equal weightage for sentiment analysis
    """
    
    def __init__(self, model_name="sshleifer/distilbart-cnn-6-6"):
        """
        Initialize the summarizer with a lightweight model
        
        Args:
            model_name: The name of the pre-trained model to use
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            print(f"Loaded summarization model: {model_name}")
        except Exception as e:
            print(f"Error loading summarization model: {e}")
            print("Falling back to rule-based summarization")
            self.tokenizer = None
            self.model = None
    
    def summarize_for_sentiment(self, story_context: str, user_prompt: str) -> str:
        """
        Create a balanced summary combining story context and user prompt
        with equal weightage for sentiment analysis using a two-pass approach:
        1. First summarize the story context
        2. Then combine with user prompt
        3. Finally summarize the combined text
        
        Args:
            story_context: Recent story events as a string
            user_prompt: User's current input
            
        Returns:
            A combined summary suitable for sentiment analysis
        """
        # If model isn't loaded, use rule-based approach
        if not self.model or not self.tokenizer:
            return self._rule_based_summary(story_context, user_prompt)
        
        try:
            # PASS 1: Summarize the story context first
            if len(story_context) > 100:  # Only summarize if story context is long
                story_prompt = f"Briefly summarize this story situation in 1-2 sentences: {story_context}"
                
                # Tokenize and generate story summary
                story_inputs = self.tokenizer(story_prompt, return_tensors="pt", max_length=512, truncation=True)
                
                with torch.no_grad():
                    story_summary_ids = self.model.generate(
                        story_inputs["input_ids"],
                        max_length=100,  # Keep story summary concise
                        min_length=20,   # Ensure it's not too short
                        num_beams=4,     # Beam search for better quality
                        early_stopping=True
                    )
                
                story_summary = self.tokenizer.decode(story_summary_ids[0], skip_special_tokens=True)
                print(f"[DEBUG] Story summary: {story_summary}")
            else:
                # If story is already short, use as is
                story_summary = story_context
            
            # PASS 2: Combine story summary with user prompt and summarize again
            combined_text = f"Story situation: {story_summary}\n\nUser action: {user_prompt}"
            
            # Tokenize and generate final summary
            final_inputs = self.tokenizer(combined_text, return_tensors="pt", max_length=512, truncation=True)
            
            with torch.no_grad():
                final_summary_ids = self.model.generate(
                    final_inputs["input_ids"],
                    max_length=150,      # Allow for a reasonable summary length
                    min_length=50,       # Ensure it's not too short
                    num_beams=4,         # Beam search for better quality
                    early_stopping=False, # Don't stop early to ensure complete thoughts
                    no_repeat_ngram_size=3,  # Avoid repetition
                    length_penalty=1.5,  # Encourage longer summaries
                    do_sample=True,      # Add some randomness
                    top_p=0.95,         # Nucleus sampling
                    temperature=0.7      # Control randomness
                )
            
            final_summary = self.tokenizer.decode(final_summary_ids[0], skip_special_tokens=True)
            
            # ALWAYS ensure the user action is included
            if user_prompt.lower() not in final_summary.lower() and "user wants to" not in final_summary.lower():
                if not final_summary.endswith('.'):
                    final_summary += '.'
                final_summary += f" The user wants to {user_prompt}"
            
            print(f"Summary: {final_summary}")
            return final_summary
            
        except Exception as e:
            print(f"Error in model-based summarization: {e}")
            return self._rule_based_summary(story_context, user_prompt)
    
    def _rule_based_summary(self, story_context: str, user_prompt: str) -> str:
        """
        Fallback method that uses rule-based approach to combine
        story context and user prompt
        """
        # Limit story context to reasonable length
        if len(story_context) > 200:
            # Try to find sentence boundaries
            sentences = story_context.split('. ')
            story_context = '. '.join(sentences[-3:]) + '.'
        
        # Extract key elements from story context
        key_story_elements = self._extract_key_elements(story_context)
        
        # Format with equal emphasis on story context and user action
        return f"Current situation: {key_story_elements}\n\nUser wants to: {user_prompt}"
    
    def _extract_key_elements(self, text: str) -> str:
        """
        Extract key elements from story context for rule-based summarization
        """
        # If text is short enough, just return it
        if len(text) < 150:
            return text
            
        # Try to find the most important sentences
        sentences = text.split('. ')
        
        # Look for sentences with important keywords
        important_keywords = ['fear', 'danger', 'threat', 'worry', 'concern', 
                             'discover', 'found', 'revealed', 'appeared',
                             'chasm', 'city', 'ancient', 'mysterious']
                             
        key_sentences = []
        
        # First pass: look for sentences with important keywords
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in important_keywords):
                key_sentences.append(sentence)
                
        # If we didn't find any, take the last 2-3 sentences as they often contain the most recent situation
        if not key_sentences:
            key_sentences = sentences[-min(3, len(sentences)):]
            
        # Join with periods and ensure proper formatting
        result = '. '.join(key_sentences)
        if not result.endswith('.'):
            result += '.'
            
        return result
        
    def extract_emotional_content(self, text: str) -> str:
        """
        Extract emotionally relevant content from text
        
        Args:
            text: Text to analyze
            
        Returns:
            Text focused on emotional content
        """
        # List of emotional keywords
        emotional_keywords = [
            # Basic emotions
            'happy', 'sad', 'angry', 'afraid', 'scared', 'excited', 'worried',
            'nervous', 'anxious', 'calm', 'peaceful', 'frustrated', 'annoyed',
            'delighted', 'upset', 'terrified', 'furious', 'joyful', 'miserable',
            
            # Emotional verbs
            'love', 'hate', 'fear', 'enjoy', 'regret', 'despise', 'adore',
            'cry', 'laugh', 'scream', 'yell', 'sob', 'weep', 'smile', 'frown',
            
            # Emotional states
            'feeling', 'felt', 'emotion', 'mood', 'tears', 'rage', 'joy',
            'sorrow', 'anger', 'fear', 'disgust', 'surprise', 'trust'
        ]
        
        # If model is available, use it for emotional content extraction
        if self.model and self.tokenizer:
            try:
                # Create prompt for emotional content extraction
                prompt = f"Extract emotional content from this text: {text}"
                
                inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
                
                with torch.no_grad():
                    output_ids = self.model.generate(
                        inputs["input_ids"],
                        max_length=100,
                        num_beams=2,
                        early_stopping=True
                    )
                
                emotional_content = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                if len(emotional_content) > 10:  # Ensure we got meaningful output
                    return emotional_content
            except Exception as e:
                print(f"Error extracting emotional content: {e}")
        
        # Fallback: Rule-based approach
        sentences = [s.strip() for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]
        
        emotional_sentences = []
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in emotional_keywords):
                emotional_sentences.append(sentence)
        
        if emotional_sentences:
            return '. '.join(emotional_sentences[:2]) + '.'
        else:
            # If no emotional content found, return a short version of the text
            return text[:100] + '...' if len(text) > 100 else text
