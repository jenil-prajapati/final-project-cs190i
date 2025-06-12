# Emotion-Aware Interactive Storytelling

An AI-powered interactive storytelling system that adapts narratives based on emotional analysis. This project combines natural language processing for emotion detection with context-aware story generation to create dynamic, emotionally responsive gaming experiences.

## Core Features

- 🎮 Interactive text-based storytelling with emotional awareness
- 🧠 Advanced emotion analysis using DistilRoBERTa model
- 🔄 Context-aware emotion transformation system
- 📊 Real-time emotional state tracking
- 🌍 Dynamic context adaptation
- 📈 Performance metrics and testing framework

## Technical Implementation

### Emotion Analysis System
- Base emotion detection using `j-hartmann/emotion-english-distilroberta-base`
- 7 primary emotions: anger, disgust, fear, joy, neutral, sadness, surprise
- Intensity scoring (0.0-1.0) for emotion strength
- Context-aware emotion transformation based on narrative situation

### Context Processing
- Semantic analysis using `sentence-transformers/all-MiniLM-L6-v2`
- Multiple context types: combat, danger, discovery, social, etc.
- Narrative tones: positive, negative, tense, mysterious, etc.
- Dynamic intensity adjustment based on context

## Prerequisites

- Python 3.8 or higher
- 8GB RAM minimum (16GB recommended)
- Git for version control
- Basic understanding of NLP concepts

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd emotion_storytelling
   ```

2. Create and activate a virtual environment:
   ```bash
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate

   # On Windows
   python -m venv venv
   venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. Ensure your virtual environment is activated
2. Launch the Streamlit interface:
   ```bash
   streamlit run app.py
   ```
3. Access the application at `http://localhost:8501`

## Project Structure

```
emotion_storytelling/
├── app.py                 # Main Streamlit application
├── emotion_analyzer.py    # Emotion analysis engine
├── context_feature_extractor.py  # Context processing
├── game_state.py         # State management
├── test_emotion_accuracy.py      # Testing framework
├── test_data.json        # Test cases
└── requirements.txt      # Dependencies
```

## Performance Metrics

Current system performance (as of latest testing):
- Base Model: 81.0% accuracy, MAE = 0.206
- Context-Aware: 76.2% accuracy, MAE = 0.170
- Balanced test set across 7 emotions
- Context adaptation shows 17.4% MAE reduction

## Development Status

### Implemented Features
- ✅ Base emotion detection
- ✅ Context-aware transformation
- ✅ Testing framework
- ✅ Performance metrics
- ✅ Real-time analysis

## Troubleshooting

Common issues and solutions:
1. **Model Loading Errors**: Ensure sufficient RAM and proper internet connection for initial model download
2. **Accuracy Issues**: Check context settings and transformation rules
3. **Performance Problems**: Monitor system resources and consider GPU acceleration


## Acknowledgments

- Emotion analysis model: `j-hartmann/emotion-english-distilroberta-base`
- Semantic analysis: `sentence-transformers/all-MiniLM-L6-v2`
- Testing methodology based on standard NLP evaluation metrics 
