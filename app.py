import streamlit as st
from story_engine import StoryEngine
from emotion_analyzer import EmotionAnalyzer
from game_state import GameState

# Initialize session state
if 'game_state' not in st.session_state:
    st.session_state.game_state = GameState()
if 'story_engine' not in st.session_state:
    st.session_state.story_engine = StoryEngine()
if 'emotion_analyzer' not in st.session_state:
    st.session_state.emotion_analyzer = EmotionAnalyzer()
if 'player_input' not in st.session_state:
    st.session_state.player_input = ''

# Page config
st.set_page_config(page_title="Emotion-Aware Interactive Storytelling", layout="wide")

# Title and description
st.title("🎲 Emotion-Aware Interactive Storytelling")
st.markdown("""
This is an AI-powered interactive storytelling system that adapts to your emotional state.
Enter your actions and watch as the story unfolds based on your choices and emotions!
""")

# Player name input (if not already set)
if not st.session_state.game_state.player_name:
    player_name = st.text_input("Enter your character's name:")
    if player_name:
        st.session_state.game_state.player_name = player_name
        # Generate initial scenario with player name
        initial_scenario = st.session_state.story_engine.generate_initial_scenario(player_name)
                
        # Add the story event
        st.session_state.game_state.add_story_event(
            initial_scenario,
            {'primary_emotion': 'neutral', 'intensity': 0.5}
        )
        st.rerun()

# Main game interface
if st.session_state.game_state.player_name:
    # Display game state
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("Game Stats")
        st.write(f"Player: {st.session_state.game_state.player_name}")
    
    with col1:
        # Display story context
        st.subheader("Story")
        for event in st.session_state.game_state.story_context:
            st.write(event['text'])
            st.write("---")
        
        # Player input
        player_input = st.text_area("What would you like to do?", value=st.session_state.player_input)
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Submit Action"):
                if player_input:
                    st.session_state.player_input = ''  # Clear the input field
                    # Analyze emotion with context awareness
                    emotion_data = st.session_state.emotion_analyzer.analyze_emotion(
                        player_input, 
                        st.session_state.game_state
                    )
                    
                    # Generate story continuation
                    response = st.session_state.story_engine.generate_story_continuation(
                        player_input,
                        emotion_data,
                        st.session_state.game_state
                    )
                    
                    # Update game state
                    st.session_state.game_state.add_story_event(response, emotion_data)
                    st.rerun()
        
        with col2:
            if st.button("Start Over"):
                st.session_state.game_state = GameState()
                st.rerun()

# Display emotional state (if available)
if st.session_state.game_state.emotional_history:
    st.sidebar.subheader("Your Emotional Journey")
    latest_emotion = st.session_state.game_state.emotional_history[-1]
    st.sidebar.write(f"Current emotion: {latest_emotion['primary_emotion']}")
    st.sidebar.write(f"Intensity: {latest_emotion['intensity']:.2f}")
    
    # Safely display confidence if it exists
    if 'confidence' in latest_emotion:
        st.sidebar.write(f"Confidence: {latest_emotion['confidence']:.2f}")
    
    # Display secondary emotions if available
    if 'secondary_emotions' in latest_emotion and latest_emotion['secondary_emotions']:
        st.sidebar.subheader("Secondary Emotions")
        for emotion, score in latest_emotion['secondary_emotions'].items():
            st.sidebar.write(f"{emotion}: {score:.2f}")