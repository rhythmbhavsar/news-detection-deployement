from flask import Flask, request, jsonify
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import joblib
from keras.utils import to_categorical
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Load the tokenizer and label encoder
objects = joblib.load('tokenizer_label_encoder.joblib')
tokenizer = objects['tokenizer']
label_encoder = objects['target_label_encoder']

# Load the trained model
model = load_model('my_model5.h5')

def preprocess_text(text):
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    return ' '.join(lemmatized_tokens)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_text = data['text']

    input_text_processed = preprocess_text(input_text)  

    max_sequence_length = 50 
    X_input_seq = tokenizer.texts_to_sequences([input_text_processed])
    X_input_padded = pad_sequences(X_input_seq, maxlen=max_sequence_length)

    predictions = model.predict(X_input_padded)
    predicted_class = label_encoder.inverse_transform([predictions.argmax()])[0]

    return jsonify({'predicted_class': predicted_class})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  
