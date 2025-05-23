# Emotion-Aware Interactive Storytelling

An AI-powered Dungeon Master (DM) that creates dynamic, emotion-aware interactive storytelling experiences. This application uses natural language processing to analyze player emotions and adapts the story accordingly, creating a personalized and immersive gaming experience.

## Features

- 🎮 Interactive text-based storytelling
- 😊 Real-time emotion analysis of player inputs
- 🔄 Adaptive story generation based on emotional context
- 📊 Dynamic difficulty adjustment
- 🤖 Rich narrative generation using GPT-3.5
- 📈 Emotional journey tracking
- 🎒 Inventory system
- 📍 Location tracking

## Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Git (for cloning the repository)

## Getting Your OpenAI API Key

1. Visit [OpenAI's platform](https://platform.openai.com/signup)
2. Sign up or log in to your account
3. Go to [API Keys section](https://platform.openai.com/api-keys)
4. Click "Create new secret key"
5. Copy your API key (you won't be able to see it again)
6. Note: OpenAI requires a payment method, but new accounts get $5 free credit

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

4. Create a `.env` file in the project root:
   ```bash
   echo "OPENAI_API_KEY=your_api_key_here" > .env
   ```

5. Replace `your_api_key_here` in the `.env` file with your actual OpenAI API key

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

## Project Structure

- `app.py`: Main Streamlit application and UI
- `story_engine.py`: Core storytelling logic and OpenAI GPT integration
- `emotion_analyzer.py`: Sentiment analysis using VADER and TextBlob
- `game_state.py`: Game state management and difficulty adjustment
- `requirements.txt`: Project dependencies
- `.env`: Environment variables (API key)

## Technical Details

- **Emotion Analysis**: Combines VADER and TextBlob for robust sentiment analysis
- **Story Generation**: Uses GPT-3.5-turbo for dynamic narrative creation
- **State Management**: Persistent game state with Streamlit's session state
- **UI Framework**: Built with Streamlit for a clean, responsive interface

## Troubleshooting

1. **ModuleNotFoundError**: Make sure you've activated the virtual environment and installed requirements
2. **API Key Error**: Verify your OpenAI API key is correctly set in the `.env` file
3. **Streamlit Connection Error**: Check if port 8501 is available on your machine

## Cost Considerations

- The application uses OpenAI's GPT-3.5-turbo model
- Current pricing: ~$0.002 per 1000 tokens
- New OpenAI accounts get $5 free credit
- Set usage limits in your OpenAI account to control costs

## Future Enhancements

- Save/load game functionality
- Multiple story branches
- Character attributes and skills
- More sophisticated emotion analysis
- Integration with image generation for scene visualization
- Multiplayer support

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements. 