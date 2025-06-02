# Emotion-Aware Interactive Storytelling

An AI-powered Dungeon Master (DM) that creates dynamic, emotion-aware interactive storytelling experiences using open-source models.

## Setup Instructions

1. Clone this repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. Activate your virtual environment if not already activated
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Features

- Interactive text-based storytelling
- Emotion analysis of player inputs
- Adaptive story generation based on emotional context
- Dynamic difficulty adjustment
- Rich narrative generation using TinyLlama (open-source LLM)
- No API keys or paid services required

## Technical Details

This project uses:
- TinyLlama-1.1B-Chat: A lightweight open-source language model for story generation
- VADER and TextBlob: For sentiment analysis
- Streamlit: For the web interface
- PyTorch: For running the language model
- Transformers: Hugging Face's library for accessing and running the model

## Project Structure

- `app.py`: Main Streamlit application
- `story_engine.py`: Core storytelling logic and LLM integration
- `emotion_analyzer.py`: Sentiment analysis tools
- `game_state.py`: Game state management
- `prompts.py`: LLM prompt templates 