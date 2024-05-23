import streamlit as st
from transformers import AutoTokenizer, FlaubertForSequenceClassification
import torch
import pandas as pd

custom_css = """
<style>
.stApp {
    background-color: #333333; /* Black background color */
    color: white;
}

.stMarkdown, .stText, .stWrite {
    color: white; /* Ensure other texts remain white */
}

.stTitle {
    color: white; /* White color for the title */
}

h1, h2, h3, h4, h5, h6 {
    color: white; /* Titles color */
    text-align: center; /* Center the titles */
}

.stSelectbox > label {
    color: white; /* White color for selectbox label */
}

.stButton button {
    background-color: #1E90FF; /* Blue background */
    color: white; /* White text */
}

.css-1aehpvj .st-cb, .css-1aehpvj .st-cb div {
    color: white; /* Change checkbox color to white */
}

.stTextInput div, .stTextArea div {
        color: white !important;
    }
</style>
"""

# Inject the custom CSS
st.markdown(custom_css, unsafe_allow_html=True)

# Lazy load model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    model_path = 'AntoineTrabia/FrenchSongDifficulty'
    model = FlaubertForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

# Function to predict difficulty level
def predict_difficulty(sentence, model, tokenizer):
    # Tokenize the input sentence
    inputs = tokenizer(sentence, return_tensors="pt")
    
    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Apply softmax to get probabilities
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # Define difficulty levels
    difficulty_levels = ["A1", "A2", "B1", "B2", "C1", "C2"]
    
    # Get the predicted difficulty level
    predicted_level = difficulty_levels[torch.argmax(probs)]
    
    return predicted_level

# Streamlit app
st.title("Lyrical Lingo")

st.markdown(f"<h2 style='margin-top: 0px; margin-bottom: 0px; padding-top: 5px; color: grey'> French Song Difficulty Estimator </h2>", unsafe_allow_html=True)

st.markdown("<div style='height: 35px;'></div>", unsafe_allow_html=True)  # Add big spacing


st.write("""
Enhance your French language skills through the power of music! Listening to French songs is one of the best ways to learn the language, but how do you choose the right songs to learn from?

Don't worry, <b>Lyrical Lingo</b> has got you covered!

Select from our curated list of songs, organized alphabetically and ranging from French classics to contemporary hits, or insert your own favorites. Then, discover their difficulty level, from A1 (beginner) to C2 (advanced), and start learning with songs adapted to your level!
""", unsafe_allow_html=True)


st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)  # Add big spacing

# Load predefined songs
predefined_songs = pd.read_csv('https://raw.githubusercontent.com/msperand/Machine_Learning_Project/main/Streamlit/songs.csv')

# Combine song name and singer for the dropdown menu
predefined_songs['Song_Display'] = predefined_songs['Song Name'] + " by " + predefined_songs['Singer']

# List of options for the selectbox
options = ["Choose my own"] + predefined_songs['Song_Display'].tolist()

# Find the index of "La Vie en Rose by Edith Piaf"
default_option = "La Vie en Rose by Edith Piaf"
default_index = options.index(default_option)

# Dropdown menu for predefined songs, setting "La Vie en Rose" as default
selected_song = st.selectbox(
    "Select one of the following songs:",
    options,
    index=default_index
)

if selected_song == "Choose my own":
    # Text area for user input
    user_title = st.text_input("Enter the title of your song:")
    user_author = st.text_input("Enter the author of your song:")
    user_text = st.text_area("Enter the lyrics of your song (one sentence per line):")

difficulty_mapping = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 6}
reverse_mapping = {v: k for k, v in difficulty_mapping.items()}

if st.button("Predict Difficulty Levels"):
    sentences = ""
    if selected_song != "Choose my own":
        # Extract the song name part from the selected option
        song_name = selected_song.split(" by ")[0]
        song_info = predefined_songs.loc[predefined_songs['Song Name'] == song_name].iloc[0]
        sentences = song_info['Lyrics']
        title = song_info['Song Name']
        author = song_info['Singer']
    elif user_text:
        sentences = user_text
        author = user_author
        title = user_title 

    if sentences:
        # Load the model and tokenizer only when needed
        model, tokenizer = load_model_and_tokenizer()
        
        sentence_list = sentences.split('\n')
        results = []
        difficulty_count = {"A1": 0, "A2": 0, "B1": 0, "B2": 0, "C1": 0, "C2": 0}
        
        for sentence in sentence_list:
            if sentence.strip():  # Ensure the sentence is not empty
                difficulty_level = predict_difficulty(sentence.strip(), model, tokenizer)
                results.append((sentence.strip(), difficulty_level))
                difficulty_count[difficulty_level] += 1
        
        st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)  # Add big spacing
        
        st.markdown(f"<h1 style='margin-bottom: 0px; padding-bottom: 0px'>{title}</h1>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='margin-top: 0px; margin-bottom: 0px; padding-top: 5px; color: grey'>{author}</h2>", unsafe_allow_html=True)

        # Calculate percentages and weighted average
        total_sentences = len([s for s in sentence_list if s.strip()])
        if total_sentences > 0:
            weighted_sum = 0
            for level, count in difficulty_count.items():
                percentage = (count / total_sentences) * 100
                weighted_sum += difficulty_mapping[level] * count

            weighted_average = weighted_sum / total_sentences
            closest_level = round(weighted_average)
            overall_difficulty = reverse_mapping[closest_level]

            st.markdown(f"""
                <div style='background-color: #1E90FF; 
                            color: white; 
                            text-align: center; 
                            padding: 10px; 
                            font-size: 100px; 
                            width: 200px; 
                            height: 200px; 
                            margin: 10px auto 30px auto; 
                            font-weight: bold; 
                            display: flex; 
                            justify-content: center; 
                            align-items: center;
                            border-radius: 15px;'>  
                    {overall_difficulty}
                </div>
            """, unsafe_allow_html=True)

            st.markdown("<div style='height: 75px;'></div>", unsafe_allow_html=True)  # Add big spacing
            
            st.markdown("<span style='font-size: 32px; color: grey;'><b>See details below:</b></span>", unsafe_allow_html=True)

            st.markdown("<span style='font-size: 24px; color: grey;'><b>Here is a breakdown for each sentence:</b></span>", unsafe_allow_html=True)
            for sentence, level in results:
                st.write(f"{sentence}\n: {level}\n\n")

            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)  # Add big spacing
            
            st.markdown("<span style='font-size: 24px; color: grey;'><b>Difficulty Level Distribution:</b></span>", unsafe_allow_html=True)
            for level, count in difficulty_count.items():
                percentage = (count / total_sentences) * 100
                st.write(f"**{level}:** {percentage:.2f}%")

    else:
        st.write("Please select a song or directly enter song lyrics.")
