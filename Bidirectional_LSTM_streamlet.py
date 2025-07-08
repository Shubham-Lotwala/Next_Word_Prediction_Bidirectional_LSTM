import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# App title and description
st.title("Next Word Prediction")

# File paths
model_path = "./Bidirectional_LSTM.keras"
text_path = "shakespeare-hamlet.txt"

# Load resources
with st.spinner("Loading resources..."):
    # Load and preprocess text
    try:
        with open(text_path, "r", encoding="utf-8") as file:
            raw_text = file.read()
        cleaned_text = raw_text.lower()
    except Exception as e:
        st.error(f"Error loading text file: {e}")
        st.stop()

    # Initialize and fit tokenizer
    text_tokenizer = Tokenizer()
    text_tokenizer.fit_on_texts([cleaned_text])
    total_words = len(text_tokenizer.word_index) + 1

    # Create reverse word mapping
    reverse_word_mapping = {
        index: word for word, index in text_tokenizer.word_index.items()
    }

    # Load model
    try:
        model = load_model(model_path)
        sequence_length = model.input_shape[1]
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# User input section
st.subheader("Text Input")
seed_text = st.text_input("Enter starting text:", "to be or not to be")
generate_btn = st.button("Predict Next Word")

# Generation process
if generate_btn:
    current_text = seed_text

    with st.spinner("Predicting next word..."):
        # Convert text to sequence
        token_sequence = text_tokenizer.texts_to_sequences([current_text])[0]

        # Pad/trim sequence
        if len(token_sequence) < sequence_length:
            padded_sequence = pad_sequences(
                [token_sequence], maxlen=sequence_length, padding="post"
            )[0]
        else:
            padded_sequence = token_sequence[-sequence_length:]

        # Prepare model input
        model_input = padded_sequence.reshape(1, sequence_length)

        # Get prediction
        word_probabilities = model.predict(model_input, verbose=0)
        next_word_probs = word_probabilities[0][-1]
        next_word_probs[0] = 0  # Ignore padding token

        # Select and add new word
        predicted_index = np.argmax(next_word_probs)
        predicted_word = reverse_word_mapping.get(predicted_index, "")

    # Display results
    st.subheader("Prediction Result")
    st.markdown(f"**Seed text:** `{seed_text}`")
    st.markdown(f"**Predicted next word:** `{predicted_word}`")
    st.markdown(f"**Full phrase:** `{seed_text} {predicted_word}`")

    # Explanation
    st.markdown("""
    ### How This Works:
    1. Your input text is converted to numerical tokens
    2. The sequence is padded/trimmed to match the model's requirements
    3. The neural network predicts probabilities for all possible next words
    4. The highest probability word is selected (excluding padding)
    5. The results are displayed with alternative options
    """)

# Sidebar information
st.sidebar.header("About This App")
st.sidebar.markdown("**Using Model**: LSTM Bidirectional")

st.sidebar.info(
    "Note: Prediction quality is not Perfect because Model is trained on only 4000 words"
)
