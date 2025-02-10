from flask import Flask, request, jsonify
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import joblib
from keras.utils import to_categorical
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

# Download necessary NLTK data and handle potential errors
try:
    NLTK_DATA_PATH = os.path.join(os.path.dirname(__file__), 'nltk_data')
    nltk.data.path.append(NLTK_DATA_PATH)
except Exception as e:
    print(f"Error downloading NLTK data: {e}")
    raise

app = Flask(__name__)

# Load the tokenizer and label encoder
try:
    objects = joblib.load('tokenizer_label_encoder.joblib')
    tokenizer = objects['tokenizer']
    label_encoder = objects['target_label_encoder']
except FileNotFoundError:
    raise Exception("The 'tokenizer_label_encoder.joblib' file is missing. Please ensure it is in the correct location.")

# Load the trained model
try:
    model = load_model('my_model5.h5')
except FileNotFoundError:
    raise Exception("The 'my_model5.h5' file is missing. Please ensure it is in the correct location.")

def preprocess_text(text):
    """Preprocesses the input text by tokenizing, removing stopwords, and lemmatizing."""
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    return ' '.join(lemmatized_tokens)

@app.route('/', methods=['GET'])
def welcome():
    """Welcome route for the API."""
    return "Welcome to the Text Prediction API! Use the '/predict' endpoint to get prediction."

@app.route('/predict', methods=['POST'])
def predict():
    """Predicts the class of the input text."""
    try:
        print("Running predict")
        input_text = request.form['text']
        print(input_text)
        if not input_text:
            return jsonify({'error': 'Input text is missing'}), 400

        # Preprocess the input text
        #input_text_processed = preprocess_text(input_text)

        # Prepare the input sequence for the model
        max_sequence_length = 50
        X_input_seq = tokenizer.texts_to_sequences([input_text])
        X_input_padded = pad_sequences(X_input_seq, maxlen=max_sequence_length)

        # Get predictions
        print("Starting prediction")
        predictions = model.predict(X_input_padded)
        print(str(predictions))
        predicted_class = label_encoder.inverse_transform([predictions.argmax()])[0]
        print(predicted_class)
        return jsonify({'predicted_class': predicted_class})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run()
