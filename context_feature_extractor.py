import torch
from typing import Dict, List
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

class ContextFeatureExtractor:
    """
    ML-based context feature extractor that uses transformer models
    to create semantic representations of game context for sentiment analysis
    """
    
    def __init__(self):
        """Initialize the feature extractor with pre-trained models"""
        # Load a lightweight model for semantic text analysis
        print("Loading context feature extraction model...")
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            
            # Put model in evaluation mode
            self.model.eval()
            
            # Move to CPU - can be changed to GPU if available
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            
            # Define context categories and their prototype sentences
            self.context_prototypes = {
                "combat": [
                    'The enemy charges forward with weapons drawn, ready to strike.',
                    'A fierce battle erupts as warriors clash with intense fury.',
                    'Swords swing and arrows fly in a chaotic fight for survival.',
                    'The opponent advances menacingly, intent on violence.'
                ],
                "danger": [
                    'A deadly threat looms close, danger hanging in the air.',
                    'The ground trembles as a perilous situation unfolds.',
                    'Fear grips the heart as an enemy approaches with malice.',
                    'A life-threatening trap is set, waiting to ensnare the unwary.'
                ],
                "discovery": [
                    'A hidden treasure is uncovered in an ancient ruin.',
                    'New lands stretch out before the eyes of the explorer.',
                    'A secret passage opens, revealing unknown wonders.',
                    'An unexpected artifact surfaces from the depths of history.'
                ],
                "social": [
                    'Friends gather to share stories and laughter over a meal.',
                    'A heated debate unfolds among allies in a crowded hall.',
                    'Negotiations begin with a rival faction to forge peace.',
                    'A celebration brings the community together in joy.'
                ],
                "puzzle": [
                    'A complex riddle awaits solving to unlock the next step.',
                    'Intricate mechanisms must be deciphered to proceed.',
                    'A maze of choices presents a challenging conundrum.',
                    'Clues are pieced together to reveal a hidden truth.'
                ],
                "travel": [
                    'The journey continues across vast, untamed wilderness.',
                    'A long road stretches ahead under an open sky.',
                    'The party treks through rugged terrain to reach their goal.',
                    'A caravan moves steadily through unfamiliar lands.'
                ],
                "rest": [
                    'The group settles down for a quiet night by the campfire.',
                    'A peaceful respite is taken in a safe haven.',
                    'Weary travelers find solace in a cozy inn.',
                    'Rest comes easily under the shelter of ancient trees.'
                ],
                "planning": [
                    'Strategies are drawn up to tackle the upcoming challenge.',
                    'The team discusses the best course of action for tomorrow.',
                    'A detailed plan is crafted to outwit the enemy.',
                    'Tactics are debated to ensure a successful mission.'
                ],
                "magical": [
                    'A surge of arcane energy crackles through the air.',
                    'Mystical runes glow with otherworldly power.',
                    'A sorcerer casts a spell, bending reality to their will.',
                    'An enchanted forest whispers with unseen magic.'
                ],
                "reward": [
                    'Victory brings a bounty of gold and precious relics.',
                    'The quest ends with a well-earned prize in hand.',
                    'A grateful villager offers a token of immense value.',
                    'Success yields treasures beyond imagination.'
                ],
                "loss": [
                    'A cherished companion falls in the heat of battle.',
                    'The mission fails, leaving only grief in its wake.',
                    'A valuable possession slips through desperate fingers.',
                    'Defeat weighs heavy on the hearts of the defeated.'
                ],
                "stealth": [
                    'Shadows cloak the figure creeping through the night.',
                    'Silent steps avoid detection by watchful guards.',
                    'A hidden approach keeps the enemy unaware.',
                    'Stealthy maneuvers evade a dangerous foe.'
                ],
                "neutral": [
                    'The day passes without incident or excitement.',
                    'A mundane task occupies the time with little note.',
                    'Nothing of importance stirs in the quiet surroundings.',
                    'Events unfold in an ordinary, uneventful manner.'
                ]
            }
            
            # Define narrative tone prototypes
            self.tone_prototypes = {
                "positive": "Successful, victorious, happy, fortunate events. The characters triumph over challenges, find valuable treasures, or make new allies who offer assistance.",
                "negative": "Failed, defeated, sad, unfortunate events. The characters face setbacks, lose valuable items, or discover troubling news about their quest.",
                "tense": "Anxious, worrying, stressful, uncertain situations. The air is thick with tension as characters face difficult choices or await imminent danger.",
                "surprising": "Unexpected, shocking, sudden, surprising events. A sudden plot twist, an ambush, or a revelation that changes everything the characters thought they knew.",
                "mysterious": "Enigmatic, cryptic, unclear, or puzzling situations. Strange symbols appear, prophecies unfold, or mysterious figures offer cryptic advice.",
                "hopeful": "Optimistic, encouraging, promising developments. A new lead emerges, help arrives unexpectedly, or a seemingly impossible task shows signs of progress.",
                "somber": "Melancholic, serious, grave, or solemn moments. Reflecting on past failures, honoring fallen comrades, or facing the weight of responsibility.",
                "humorous": "Funny, comedic, lighthearted, or amusing episodes. Comical mishaps, witty banter between characters, or absurd situations that bring levity.",
                "heroic": "Brave, courageous, valiant, or noble actions. Characters standing against overwhelming odds, making self-sacrifices, or performing extraordinary feats.",
                "magical": "Enchanting, wondrous, awe-inspiring, or supernatural experiences. Witnessing powerful magic, entering magical realms, or experiencing supernatural beauty.",
                "horrific": "Frightening, disturbing, terrifying, or macabre situations. Encountering undead horrors, witnessing dark rituals, or facing psychological terrors.",
                "peaceful": "Calm, tranquil, restful, or harmonious moments. Safe havens, moments of respite, or peaceful interactions away from danger.",
                "chaotic": "Wild, disordered, unpredictable, or frenzied situations. Battles that turn into mayhem, magical accidents, or scenes of confusion and disorder.",
                "neutral": "Balanced events without strong emotional tone. Ordinary conversations, routine travel, or moments with mixed or subtle emotions."
            }
            
            # Pre-compute embeddings for all prototypes
            self.context_embeddings = self._encode_prototypes(self.context_prototypes)
            self.tone_embeddings = self._encode_prototypes(self.tone_prototypes)
            
            print("Context feature extraction model loaded successfully")
            self.model_loaded = True
            
        except Exception as e:
            print(f"Error loading context feature extraction model: {e}")
            self.model_loaded = False
    
    def _encode_prototypes(self, prototypes: Dict[str, List[str]]) -> Dict[str, torch.Tensor]:
        """Encode prototype sentences into embeddings"""
        encodings = {}
        
        for category, texts in prototypes.items():
            # Get embeddings for the prototype texts
            embeddings = [self._get_embedding(text) for text in texts]
            encodings[category] = torch.stack(embeddings)
            
        return encodings
    
    def _get_embedding(self, text: str) -> torch.Tensor:
        """Get embedding for a piece of text using the model"""
        # Handle empty text
        if not text:
            # Return zeros tensor with correct dimensions
            return torch.zeros(384)  # MiniLM-L6-v2 has 384 dimensions
            
        # Tokenize text
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        ).to(self.device)
        
        # Get model outputs
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Average pool over token embeddings (excluding special tokens)
        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']
        
        # Average pooling (excluding padding tokens)
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, 1)
        sum_mask = torch.sum(mask_expanded, 1)
        embedding = sum_embeddings / sum_mask
        
        # Return the embedding for the first sequence in the batch
        return embedding[0].cpu()
    
    def _get_closest_prototype(self, embedding: torch.Tensor, prototype_embeddings: Dict[str, torch.Tensor]) -> tuple:
        """Find the closest prototype to the given embedding"""
        similarities = {}
        
        for category, proto_embeddings in prototype_embeddings.items():
            # Calculate cosine similarity
            similarity = torch.nn.functional.cosine_similarity(
                embedding.unsqueeze(0), proto_embeddings
            ).mean().item()
            similarities[category] = similarity
            
        # Get the highest similarity category and its score
        best_category = max(similarities.items(), key=lambda x: x[1])
        
        return best_category[0], best_category[1]  # (category, similarity_score)
    
    def _extract_semantic_keywords(self, text: str, top_k: int = 5) -> List[str]:
        """Extract keywords that best represent the semantic content of the text"""
        # A simple function to extract rough keywords
        # In a full implementation, this would use KeyBERT or another ML approach
        
        # For now, use a simple frequency-based approach with stopwords filtering
        stopwords = set(['a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 
                         'is', 'are', 'was', 'were', 'be', 'been', 'being', 'you', 'your', 'they', 'their', 'it', 'its'])
        
        # Tokenize text into words
        words = text.lower().split()
        
        # Remove stopwords and short words
        filtered_words = [word for word in words if word not in stopwords and len(word) > 3]
        
        # Count word frequencies
        word_freq = {}
        for word in filtered_words:
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1
                
        # Get top k most frequent words
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Return just the words (not the counts)
        return [word for word, _ in sorted_words[:top_k]]
    
    def extract_features(self, game_state) -> Dict:
        """Extract context features from the game state"""
        if not self.model_loaded or not game_state:
            return {
                "context_type": "neutral",
                "narrative_tone": "neutral",
                "intensity": 0.5,
                "context_confidence": 0.5
            }
        
        try:
            # Get context type and narrative tone directly from game state
            context_type = getattr(game_state, 'context_type', 'neutral')
            narrative_tone = getattr(game_state, 'narrative_tone', 'neutral')
            context_intensity = getattr(game_state, 'context_intensity', 0.5)
            
            # Return the features
            return {
                "context_type": context_type,
                "narrative_tone": narrative_tone,
                "intensity": context_intensity,
                "context_confidence": 0.8  # High confidence since we're using explicit context
            }
            
        except Exception as e:
            print(f"Error extracting context features: {e}")
            return {
                "context_type": "neutral",
                "narrative_tone": "neutral",
                "intensity": 0.5,
                "context_confidence": 0.5
            }
