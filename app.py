from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
import sys
import tensorflow.keras.preprocessing.text as keras_text
sys.modules['keras.preprocessing.text'] = keras_text

app = Flask(__name__)

# Set the sequence length used during training (adjust as needed)
text_seq_len = 50

# Dictionaries to store models, histories, and tokenizers
models = {}
histories = {}
tokenizers = {}

@tf.keras.utils.register_keras_serializable()
class MyGRU(tf.keras.layers.GRU):
    def __init__(self, *args, **kwargs):
        kwargs.pop("time_major", None)
        super().__init__(*args, **kwargs)

@tf.keras.utils.register_keras_serializable()
class MyLSTM(tf.keras.layers.LSTM):
    def __init__(self, *args, **kwargs):
        kwargs.pop("time_major", None)
        super().__init__(*args, **kwargs)

@tf.keras.utils.register_keras_serializable()
class MyBI_LSTM(tf.keras.layers.Bidirectional):
    def __init__(self, *args, **kwargs):
        kwargs.pop("time_major", None)
        super().__init__(*args, **kwargs)

@tf.keras.utils.register_keras_serializable()
class MyBI_GRU(tf.keras.layers.Bidirectional):
    def __init__(self, *args, **kwargs):
        kwargs.pop("time_major", None)
        super().__init__(*args, **kwargs)

# Model 1: GRU (unchanged)
models['GRU'] = tf.keras.models.load_model("Models/GRU/GRU.h5", custom_objects={'GRU': MyGRU})
with open("Models/GRU/gru_history.pkl", "rb") as f:
    histories['GRU'] = pickle.load(f)
with open("Models/GRU/gru_tokenizer.pkl", "rb") as f:
    tokenizers['GRU'] = pickle.load(f)

# Model 2: LSTM – load with custom object mapping key 'LSTM'
models['LSTM'] = tf.keras.models.load_model("Models/LSTM/LSTM.h5", custom_objects={'LSTM': MyLSTM})
with open("Models/LSTM/lstm_history.pkl", "rb") as f:
    histories['LSTM'] = pickle.load(f)
with open("Models/LSTM/lstm_tokenizer.pkl", "rb") as f:
    tokenizers['LSTM'] = pickle.load(f)

# Model 3: Bidirectional-LSTM – map both 'Bidirectional' and 'LSTM'
models['Bidirectional-LSTM'] = tf.keras.models.load_model(
    "Models/BIDIRECTIONAL_LSTM/Bi_di_LSTM.h5",
    custom_objects={'Bidirectional': MyBI_LSTM, 'LSTM': MyLSTM}
)
with open("Models/BIDIRECTIONAL_LSTM/bi_di_lstm_history.pkl", "rb") as f:
    histories['Bidirectional-LSTM'] = pickle.load(f)
with open("Models/BIDIRECTIONAL_LSTM/bi_di_lstm_tokenizer.pkl", "rb") as f:
    tokenizers['Bidirectional-LSTM'] = pickle.load(f)

# Model 4: Bidirectional-GRU – map both 'Bidirectional' and 'GRU'
models['Bidirectional-GRU'] = tf.keras.models.load_model(
    "Models/BIDIRECTIONAL_GRU/Bi_di_GRU.h5",
    custom_objects={'Bidirectional': MyBI_GRU, 'GRU': MyGRU}
)
with open("Models/BIDIRECTIONAL_GRU/bi_di_gru_history.pkl", "rb") as f:
    histories['Bidirectional-GRU'] = pickle.load(f)
with open("Models/BIDIRECTIONAL_GRU/bi_di_gru_tokenizer.pkl", "rb") as f:
    tokenizers['Bidirectional-GRU'] = pickle.load(f)


def generate_story(model, tokenizer, text_seq_len, seed_text, n_words):
    text = []
    for _ in range(n_words):
        # Convert seed text to a sequence of integers
        encoded = tokenizer.texts_to_sequences([seed_text])[0]
        # Pad the sequence to ensure consistent length
        encoded = pad_sequences([encoded], maxlen=text_seq_len, padding='pre')
        # Predict the next word probabilities
        pred = model.predict(encoded)
        # Choose the word with the highest probability
        y_pred = np.argmax(pred, axis=1)[0]
        predicted_word = ''
        # Map the integer back to the word
        for word, index in tokenizer.word_index.items():
            if index == y_pred:
                predicted_word = word
                break
        seed_text = seed_text + ' ' + predicted_word
        text.append(predicted_word)
    return ' '.join(text)

@app.route("/")
def home():
    return render_template("index.html", histories=histories)

@app.route("/generate-story", methods=["POST"])
def generate_story_endpoint():
    data = request.get_json()
    seed_text = data.get("seed_text", "")
    num_words = data.get("num_words", 100)
    model_name = data.get("model_name", "GRU")
    selected_model = models.get(model_name)
    selected_tokenizer = tokenizers.get(model_name)
    if selected_model is None or selected_tokenizer is None:
        return jsonify({"error": "Model or tokenizer not found"}), 400
    generated_story = generate_story(selected_model, selected_tokenizer, text_seq_len, seed_text, num_words)
    return jsonify({"generated_story": generated_story})

if __name__ == "__main__":
    app.run(debug=True, port=7860)
