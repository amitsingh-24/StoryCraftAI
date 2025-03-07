#!/usr/bin/env python3
"""
Automatic Story Generation Script

This script scrapes short stories from http://www.classicshorts.com/,
preprocesses the text, builds and trains a neural network model to predict the next word,
and generates new story text from a seed phrase.

Available model types:
    - "bi_di_gru"   : Bidirectional GRU (default)
    - "bi_di_lstm"  : Bidirectional LSTM
    - "gru"         : Unidirectional GRU
    - "lstm"        : Unidirectional LSTM
"""

import os
import re
import string
import pickle
import requests
import nltk
from bs4 import BeautifulSoup
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from nltk.tokenize import sent_tokenize, word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GRU, LSTM, Embedding, Bidirectional

# Download necessary NLTK data
nltk.download('punkt_tab')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Using GPU(s):", tf.config.list_physical_devices('GPU'))
    except RuntimeError as e:
        print("Error setting GPU memory growth:", e)
else:
    print("No GPU detected. Check your runtime settings.")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
print("TensorFlow Version:", tf.__version__)

def get_page(url: str, headers: dict) -> BeautifulSoup:
    """
    Retrieve and parse a webpage.

    Args:
        url (str): URL of the page.
        headers (dict): HTTP headers for the request.

    Returns:
        BeautifulSoup: Parsed HTML content.
    """
    response = requests.get(url, headers=headers)
    try:
        response.raise_for_status()
    except Exception:
        pass  # In production, log this error
    return BeautifulSoup(response.text, 'html.parser')


def get_story_text(link: str) -> str:
    """
    Fetch the full story text from a given URL by cleaning the HTML.

    Args:
        link (str): Story URL.

    Returns:
        str: Cleaned story text.
    """
    regex = re.compile(r'[\n\r\t]')
    headers = {'User-Agent': 'Mozilla/5.0'}
    page_html = get_page(link, headers)
    paragraphs = page_html.find_all("div", class_="StoryPara")
    total_text = ""
    for paragraph in paragraphs:
        total_text += regex.sub(" ", paragraph.text.strip())
    return total_text


def get_listings(max_stories: int = 100) -> str:
    """
    Scrape the Classic Short Stories listings and return concatenated story text.

    Args:
        max_stories (int): Maximum number of stories to scrape.

    Returns:
        str: Combined text of all scraped stories.
    """
    story_count = 0
    bad_titles = {'tlrm', 'fiddler', 'frog', 'ItalianMaster', 'luck'}
    base_url = "http://www.classicshorts.com"
    listings_url = f"{base_url}/bib.html"
    headers = {'User-Agent': 'Mozilla/5.0'}
    raw_text = ""
    
    page_html = get_page(listings_url, headers)
    listing_elements = page_html.find_all("div", class_="biolisting")
    
    for elem in listing_elements:
        story_id = elem.attrs['onclick'][11:-2]
        if story_id not in bad_titles:
            current_url = f"{base_url}/stories/{story_id}.html"
            raw_text += get_story_text(current_url)
            story_count += 1
        if story_count == max_stories:
            break
    return raw_text

def clean_text(sentences: list) -> list:
    """
    Tokenize and clean sentences by removing punctuation, non-alphabetic tokens,
    and converting text to lowercase.

    Args:
        sentences (list): List of sentence strings.

    Returns:
        list: List of cleaned tokens.
    """
    tokens = []
    for sentence in sentences:
        tokens.extend(word_tokenize(sentence))
    translator = str.maketrans('', '', string.punctuation)
    tokens = [token.translate(translator) for token in tokens]
    tokens = [token.lower() for token in tokens if token.isalpha()]
    return tokens


def prepare_data() -> tuple:
    """
    Scrape, tokenize, and generate training sequences from the text data.

    Returns:
        tuple: (X, y, vocabulary_size, seq_length, tokenizer)
            X (np.ndarray): Input sequences.
            y (np.ndarray): One-hot encoded target words.
            vocabulary_size (int): Vocabulary size.
            seq_length (int): Length of input sequences.
            tokenizer (Tokenizer): Fitted tokenizer.
    """
    # Scrape and pre-process stories.
    stories = get_listings()
    # Optionally remove header noise (adjust slicing if needed)
    stories = stories[81:]
    sentences = sent_tokenize(stories)
    tokens = clean_text(sentences)
    
    # Create sequences of 51 tokens (50 as input, 1 as output)
    seq_total_length = 50 + 1
    lines = []
    for i in range(seq_total_length, len(tokens)):
        sequence = tokens[i - seq_total_length:i]
        lines.append(' '.join(sequence))
        if i > 120000:  # Limit data size for faster training
            break

    # Tokenize the sequences into integers.
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    sequences = tokenizer.texts_to_sequences(lines)
    sequences = np.array(sequences)
    
    # Split sequences into inputs (X) and targets (y)
    X, y = sequences[:, :-1], sequences[:, -1]
    vocabulary_size = len(tokenizer.word_index) + 1
    y = to_categorical(y, num_classes=vocabulary_size)
    seq_length = X.shape[1]
    
    return X, y, vocabulary_size, seq_length, tokenizer

def generate_story(model: tf.keras.Model,
                   tokenizer: Tokenizer,
                   text_seq_len: int,
                   seed_text: str,
                   n_words: int) -> str:
    """
    Generate new text based on a seed phrase by predicting one word at a time.

    Args:
        model (tf.keras.Model): Trained model.
        tokenizer (Tokenizer): Fitted tokenizer.
        text_seq_len (int): Expected input sequence length.
        seed_text (str): Initial seed text.
        n_words (int): Number of words to generate.

    Returns:
        str: Generated text.
    """
    generated = []
    for _ in range(n_words):
        encoded = tokenizer.texts_to_sequences([seed_text])[0]
        encoded = pad_sequences([encoded], maxlen=text_seq_len, padding='pre')
        pred = model.predict(encoded, verbose=0)
        y_pred = np.argmax(pred, axis=1)[0]
        predicted_word = next((word for word, index in tokenizer.word_index.items() if index == y_pred), '')
        seed_text += ' ' + predicted_word
        generated.append(predicted_word)
    return ' '.join(generated)

def build_model(model_type: str, vocabulary_size: int, seq_length: int) -> tf.keras.Model:
    """
    Build and return a model based on the selected architecture.

    Args:
        model_type (str): Type of model architecture ('bi_di_gru', 'bi_di_lstm', 'gru', 'lstm').
        vocabulary_size (int): Size of the vocabulary.
        seq_length (int): Length of input sequences.

    Returns:
        tf.keras.Model: Compiled Keras model.
    """
    model = Sequential()
    model.add(Embedding(vocabulary_size, 50, input_length=seq_length))
    
    if model_type == "bi_di_gru":
        model.add(Bidirectional(GRU(100, return_sequences=True)))
        model.add(Dropout(0.2))
        model.add(GRU(120))
    elif model_type == "bi_di_lstm":
        model.add(Bidirectional(LSTM(100, return_sequences=True)))
        model.add(Dropout(0.2))
        model.add(LSTM(120))
    elif model_type == "gru":
        model.add(GRU(100, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(GRU(120))
    elif model_type == "lstm":
        model.add(LSTM(100, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(120))
    else:
        raise ValueError("Invalid model type. Choose from 'bi_di_gru', 'bi_di_lstm', 'gru', or 'lstm'.")
    
    model.add(Dense(140, activation='relu'))
    model.add(Dense(vocabulary_size, activation='softmax'))
    
    model.build(input_shape=(None, seq_length))
    model.summary()
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def main():
    # Set the model type here. Options: 'bi_di_gru', 'bi_di_lstm', 'gru', 'lstm'
    MODEL_TYPE = "bi_di_gru"
    
    # Prepare the data
    X, y, vocabulary_size, seq_length, tokenizer = prepare_data()
    
    # Build and train the selected model
    model = build_model(MODEL_TYPE, vocabulary_size, seq_length)
    history = model.fit(X, y, batch_size=512, epochs=300)
    
    # Save the model, tokenizer, and training history.
    model_filename = f"./{MODEL_TYPE.upper()}_model.h5"
    tokenizer_filename = f"{MODEL_TYPE}_tokenizer.pkl"
    history_filename = f"{MODEL_TYPE}_history.pkl"
    
    model.save(model_filename)
    print(f"Model saved to {model_filename}")
    
    with open(tokenizer_filename, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Tokenizer saved to {tokenizer_filename}")
    
    with open(history_filename, 'wb') as file:
        pickle.dump(history.history, file)
    print(f"Training history saved to {history_filename}")
    
    # Plot training accuracy and loss
    epochs_range = range(len(history.history['accuracy']))
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history.history['accuracy'])
    plt.title(f"{MODEL_TYPE.upper()} Model Training Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history.history['loss'])
    plt.title(f"{MODEL_TYPE.upper()} Model Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.show()
    
    # Generate stories from a few seed texts.
    seeds = [
        "The country was in chaos but",
        "I walked out of the store dissatisfied and it",
    ]
    
    for seed in seeds:
        generated_text = generate_story(model, tokenizer, seq_length, seed, 50)
        print("\nSeed:", seed)
        print("Generated Story:", generated_text)

if __name__ == "__main__":
    main()

