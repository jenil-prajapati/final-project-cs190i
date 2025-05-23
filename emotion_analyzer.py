from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

class EmotionAnalyzer:
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        
    def analyze_emotion(self, text):
        # Get VADER sentiment scores
        vader_scores = self.vader.polarity_scores(text)
        
        # Get TextBlob sentiment
        blob = TextBlob(text)
        textblob_sentiment = blob.sentiment
        
        # Combine analyses to determine emotional state
        emotional_state = {
            'primary_emotion': self._determine_primary_emotion(vader_scores, textblob_sentiment),
            'intensity': vader_scores['compound'],
            'confidence': (textblob_sentiment.subjectivity + 1) / 2,  # Scale 0-1
            'raw_scores': {
                'vader': vader_scores,
                'textblob': {
                    'polarity': textblob_sentiment.polarity,
                    'subjectivity': textblob_sentiment.subjectivity
                }
            }
        }
        
        return emotional_state
    
    def _determine_primary_emotion(self, vader_scores, textblob_sentiment):
        compound = vader_scores['compound']
        
        if compound >= 0.5:
            return 'confident' if textblob_sentiment.subjectivity > 0.5 else 'happy'
        elif compound <= -0.5:
            return 'fearful' if textblob_sentiment.subjectivity > 0.5 else 'angry'
        elif -0.5 < compound < 0:
            return 'uncertain' if textblob_sentiment.subjectivity > 0.5 else 'neutral'
        else:
            return 'neutral' 