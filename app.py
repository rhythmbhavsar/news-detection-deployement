from flask import Flask, request, jsonify
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import joblib
from tensorflow.keras.utils import to_categorical
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

# Ensure necessary NLTK data is available
NLTK_DATA_PATH = os.path.join(os.path.dirname(__file__), 'nltk_data')
nltk.data.path.append(NLTK_DATA_PATH)
print(str(NLTK_DATA_PATH))
try:
    nltk.download('punkt', download_dir=NLTK_DATA_PATH)
    nltk.download('punkt_tab', download_dir=NLTK_DATA_PATH)
    nltk.download('stopwords', download_dir=NLTK_DATA_PATH)
    nltk.download('wordnet', download_dir=NLTK_DATA_PATH)
except Exception as e:
    print(f"Error downloading NLTK data: {e}")

app = Flask(__name__)

# Load the tokenizer and label encoder
tokenizer_label_path = 'tokenizer_label_encoder.joblib'
model_path = 'my_model5.h5'

if not os.path.exists(tokenizer_label_path):
    raise FileNotFoundError(f"Missing file: {tokenizer_label_path}. Ensure it is in the correct location.")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Missing file: {model_path}. Ensure it is in the correct location.")

try:
    objects = joblib.load(tokenizer_label_path)
    tokenizer = objects['tokenizer']
    label_encoder = objects['target_label_encoder']
except Exception as e:
    raise Exception(f"Error loading tokenizer and label encoder: {e}")

# Load the trained model without recompiling
try:
    model = load_model(model_path, compile=False)
except Exception as e:
    raise Exception(f"Error loading model: {e}")

def preprocess_text(text):
    """Preprocess the input text by tokenizing, removing stopwords, and lemmatizing."""
    if not isinstance(text, str) or not text.strip():
        return ""

    tokens = nltk.word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    return ' '.join(tokens)

@app.route('/', methods=['GET'])
def welcome():
    """Welcome route for the API."""
    return "Welcome to the Text Prediction API! Use the '/predict' endpoint to get predictions."

@app.route('/predict', methods=['POST'])
def predict():
    """Predicts the class of the input text."""
    print("Running predict")
    try:
        data = request.get_json()

        if 'text' not in data:
            return jsonify({'error': 'Missing "text" field in request body'}), 400

        input_text = data['text']

        if not input_text or not isinstance(input_text, str):
            return jsonify({'error': 'Input text is missing or invalid'}), 400

        print(f"Received text: {input_text}")

        # Preprocess the input text
        input_text_processed = preprocess_text(input_text)

        # Prepare the input sequence for the model
        max_sequence_length = 50
        X_input_seq = tokenizer.texts_to_sequences([input_text_processed])
        X_input_padded = pad_sequences(X_input_seq, maxlen=max_sequence_length)

        # Get predictions
        print("Starting prediction")
        predictions = model.predict(X_input_padded)
        print(f"Raw predictions: {predictions}")

        predicted_class = label_encoder.inverse_transform([predictions.argmax()])[0]
        print(f"Predicted class: {predicted_class}")
        return jsonify({'predicted_class': predicted_class})

    except Exception as e:
        print(str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
