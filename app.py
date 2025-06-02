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
        # Generate initial scenario
        initial_scenario = st.session_state.story_engine.generate_initial_scenario()
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
        st.write(f"Location: {st.session_state.game_state.current_location}")
        st.write(f"Difficulty: {st.session_state.game_state.difficulty_level:.2f}")
        
        # Display inventory
        st.subheader("Inventory")
        for item in st.session_state.game_state.inventory:
            st.write(f"- {item}")
    
    with col1:
        # Display story context
        st.subheader("Story")
        for event in st.session_state.game_state.story_context:
            st.write(event['text'])
            st.write("---")
        
        # Initialize session state for input counter if it doesn't exist
        if 'input_key_counter' not in st.session_state:
            st.session_state.input_key_counter = 0
        
        # Create a unique key for the text area
        input_key = f"action_input_{st.session_state.input_key_counter}"
        
        # Player input with empty default value and unique key
        player_input = st.text_area("What would you like to do?", 
                                  value="",
                                  key=input_key,
                                  height=100)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Submit Action"):
                if player_input:
                    # Analyze emotion
                    emotion_data = st.session_state.emotion_analyzer.analyze_emotion(player_input)
                    
                    # Generate story continuation
                    response = st.session_state.story_engine.generate_story_continuation(
                        player_input,
                        emotion_data,
                        st.session_state.game_state
                    )
                    
                    # Update game state
                    st.session_state.game_state.add_story_event(response, emotion_data)
                    
                    # Increment the counter to force a new text area on next render
                    st.session_state.input_key_counter += 1
                    st.rerun()
        
        with col2:
            if st.button("Start Over"):
                st.session_state.game_state = GameState()
                # Reset the counter to force a new text area
                st.session_state.input_key_counter = 0
                st.rerun()

# Display emotional state (if available)
if st.session_state.game_state.emotional_history:
    st.sidebar.subheader("Your Emotional Journey")
    latest_emotion = st.session_state.game_state.emotional_history[-1]
    st.sidebar.write(f"Current emotion: {latest_emotion['primary_emotion']}")
    st.sidebar.write(f"Intensity: {latest_emotion['intensity']:.2f}")
    # Only show confidence if it exists
    if 'confidence' in latest_emotion:
        st.sidebar.write(f"Confidence: {latest_emotion['confidence']:.2f}") 