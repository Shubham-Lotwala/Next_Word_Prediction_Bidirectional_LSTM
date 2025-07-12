import gradio as gr
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import os

# File paths - use direct paths
model_path = "Bidirectional_LSTM.keras"
text_path = "shakespeare-hamlet.txt"

# Load resources once at startup
print("Loading resources...")
try:
    # Load and preprocess text
    with open(text_path, "r", encoding="utf-8") as file:
        raw_text = file.read()
    cleaned_text = raw_text.lower()
    
    # Initialize and fit tokenizer
    text_tokenizer = Tokenizer()
    text_tokenizer.fit_on_texts([cleaned_text])
    total_words = len(text_tokenizer.word_index) + 1
    
    # Create reverse word mapping
    reverse_word_mapping = {
        index: word for word, index in text_tokenizer.word_index.items()
    }
    
    # Load model
    model = load_model(model_path)
    sequence_length = model.input_shape[1]
    print("Resources loaded successfully!")
    
except Exception as e:
    print(f"Error during initialization: {e}")
    raise e

def predict_next_word(seed_text):
    """Predict the next word given seed text"""
    try:
        current_text = seed_text.lower()
        token_sequence = text_tokenizer.texts_to_sequences([current_text])[0]
        
        # Pad/trim sequence
        if len(token_sequence) < sequence_length:
            padded_sequence = pad_sequences(
                [token_sequence], maxlen=sequence_length, padding="post"
            )[0]
        else:
            padded_sequence = token_sequence[-sequence_length:]
        
        # Prepare model input and predict
        model_input = padded_sequence.reshape(1, sequence_length)
        word_probabilities = model.predict(model_input, verbose=0)
        next_word_probs = word_probabilities[0][-1]
        next_word_probs[0] = 0  # Ignore padding token
        
        # Get predicted word
        predicted_index = np.argmax(next_word_probs)
        predicted_word = reverse_word_mapping.get(predicted_index, "<unknown>")
        
        full_phrase = f"{seed_text} {predicted_word}"
        
        return predicted_word, full_phrase
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Error", "Could not generate prediction"

# Create Gradio interface
with gr.Blocks(title="Next Word Predicton", theme="soft") as app:
    gr.Markdown("# 🎭 Next Word Predicton using Bidirectional LSTM")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="Enter starting text",
                placeholder="To be or not to..."
            )
            predict_btn = gr.Button("Predict Next Word", variant="primary")
            
        with gr.Column():
            next_word = gr.Textbox(label="Predicted next word", interactive=False)
            full_phrase = gr.Textbox(label="Full phrase", interactive=False)
    
    gr.Markdown("> **Note**: Prediction quality is limited as the model was trained on only 4000 words")
    
    with gr.Accordion("How This Works", open=True):
        gr.Markdown("""
        1. Your input text is converted to numerical tokens
        2. The sequence is padded/trimmed to match the model's requirements
        3. The neural network predicts probabilities for all possible next words
        4. The highest probability word is selected 
        5. Results are displayed
        """)
    
    predict_btn.click(
        fn=predict_next_word,
        inputs=input_text,
        outputs=[next_word, full_phrase]
    )

if __name__ == "__main__":
    app.launch()