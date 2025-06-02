import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import Dict, List

class StoryEngine:
    def __init__(self):
        # Using GPT-2-medium for better story generation
        self.model_name = "gpt2-medium"
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Determine the appropriate device
        if torch.backends.mps.is_available():
            # For Apple Silicon, use CPU for now due to autocast limitations
            self.device = "cpu"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
            
        self.model = GPT2LMHeadModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32 if self.device == "cpu" else torch.float16,
        ).to(self.device)
        
        self.story_prefix = """SCENE:
You are in [location]. The atmosphere is [description].
You feel [emotion] as [reason].

The story continues:"""

        self.story_suffix = "\n\nWhat do you do next?"
    
    def generate_story_continuation(self, 
                                  player_input: str, 
                                  emotion_data: Dict, 
                                  game_state: 'GameState') -> str:
        # Construct the prompt based on game state and emotional context
        prompt = f"""SCENE DESCRIPTION:
Setting: A meaningful moment in your life
Current emotion: {emotion_data.get('primary_emotion', 'neutral')}
Your desire: {player_input}

The scene unfolds:
You find yourself"""
        
        # Format prompt for completion
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Generate response with controlled parameters
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=input_ids.shape[1] + 100,
                min_length=input_ids.shape[1] + 50,
                temperature=0.8,
                top_k=50,
                top_p=0.92,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
            )
        
        response = self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        response = response.strip()
        
        # Ensure the response is a proper scene
        if not response.lower().startswith(("in ", "at ", "sitting", "standing", "walking")):
            response = "in " + response
            
        return response
    
    def generate_initial_scenario(self) -> str:
        prompt = """SCENE DESCRIPTION:
You find yourself at a meaningful moment. The world is full of possibilities.
Your heart beats with anticipation as you prepare to embark on a personal journey.

The scene begins:
You are"""
        
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=input_ids.shape[1] + 100,
                min_length=input_ids.shape[1] + 50,
                temperature=0.8,
                top_k=50,
                top_p=0.92,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
            )
        
        response = self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        response = response.strip()
        
        # Ensure it's a proper scene
        if not response.lower().startswith(("in ", "at ", "sitting", "standing", "walking")):
            response = "in " + response
            
        return response 