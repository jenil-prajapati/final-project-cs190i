# Emotion-Aware Interactive Storytelling

An AI-powered Dungeon Master (DM) that creates dynamic, emotion-aware interactive storytelling experiences. This application uses natural language processing to analyze player emotions and adapts the story accordingly, creating a personalized and immersive gaming experience.

## Features

- 🎮 Interactive text-based storytelling
- 😊 Detailed emotion analysis of player inputs (beyond simple positive/negative)
- 🔄 Adaptive story generation based on emotional context
- 📊 Dynamic difficulty adjustment
- 🤖 Rich narrative generation using Microsoft Phi-3-mini
- 📈 Emotional journey tracking with nuanced emotion detection
- 🎒 Inventory system
- 📍 Location tracking

## Prerequisites

- Python 3.8 or higher
- At least 8GB RAM (16GB recommended)
- Git (for cloning the repository)

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

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. First-time setup may take a few minutes as it downloads the required AI models.

## Running the Application

1. Ensure your virtual environment is activated
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. Open your browser and go to `http://localhost:8501`

## How to Play

1. Enter your character's name when prompted
2. Read the initial scenario presented by the AI Dungeon Master
3. Type your actions or responses in the text input area
4. Click "Submit Action" to continue the story
5. The system will analyze your emotional state and adapt the story accordingly
6. View your emotional journey and game stats in the sidebar
7. Use "Start Over" to begin a new adventure

## API Key Setup

1. Visit [Google AI Studio](https://aistudio.google.com/apikey)
2. Create or sign in with your Google account
3. Click "Create API Key" to generate a new key
4. Create a `.env` file in the project root:
   ```bash
   GEMINI_API_KEY=your_api_key_here
   ```

## Project Structure

- `app.py`: Main Streamlit application and UI
- `story_engine.py`: Core storytelling logic using Gemini AI
- `emotion_analyzer.py`: Emotion analysis using DistilRoBERTa
- `game_state.py`: Game state management and difficulty adjustment
- `requirements.txt`: Project dependencies
- `.env`: Environment file for API keys (not tracked in git)

## Technical Details

- **Story Generation**: Uses Google's Gemini 2.0 Flash Lite model for dynamic narrative generation
- **Emotion Analysis**: Uses DistilRoBERTa-base for detailed emotion detection (joy, sadness, fear, anger, surprise, etc.)
- **State Management**: Persistent game state with Streamlit's session state
- **UI Framework**: Built with Streamlit for a clean, responsive interface
- **Performance**: Optimized for local execution with minimal resource usage

## Troubleshooting

1. **ModuleNotFoundError**: Make sure you've activated the virtual environment and installed requirements
2. **Memory Issues**: Close other memory-intensive applications when running the game
3. **Slow Initial Load**: The first load may take time as models are downloaded and cached
4. **Streamlit Connection Error**: Check if port 8501 is available on your machine

## Hardware Considerations

- Minimum: 8GB RAM, modern dual-core CPU
- Recommended: 16GB RAM, quad-core CPU
- Models are cached after first use for faster subsequent launches
- Total disk space needed: ~2GB for models

## Future Enhancements

- Save/load game functionality
- Multiple story branches
- Character attributes and skills
- Integration with image generation for scene visualization
- Multiplayer support
- Model optimization for even faster performance

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements. 