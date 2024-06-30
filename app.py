from flask import Flask, request, jsonify
import tensorflow as tf
import joblib
import numpy as np
import struct  # for reading binary data
from nltk.tokenize import word_tokenize

app = Flask(__name__)

# Load the BiLSTM model
bilstm_model = tf.keras.models.load_model(r'C:\Users\dines\Downloads\dissertation\cnn_bilstm_model.h5')

# Load the 1D CNN model
cnn_model = tf.keras.models.load_model(r'C:\Users\dines\Downloads\dissertation\cnn_1d_model.h5')

# Load the scaler and imputer
scaler = joblib.load(r'C:\Users\dines\Downloads\dissertation\scaler.pkl')
imputer = joblib.load(r'C:\Users\dines\Downloads\dissertation\imputer.pkl')

# Function to load Word2Vec embeddings from binary file
def load_word2vec_model_binary(embeddings_path):
    embeddings = {}
    with open(embeddings_path, 'rb') as f:
        # Read header: the first line contains the number of words and the vector dimension
        header = f.readline()
        vocab_size, vector_size = map(int, header.split())
        binary_length = np.dtype(np.float32).itemsize * vector_size
        for _ in range(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == b' ':
                    word = b''.join(word).decode('utf-8')
                    break
                if ch != b'\n':
                    word.append(ch)
            embeddings[word] = np.fromstring(f.read(binary_length), dtype=np.float32)
    return embeddings

# Load Word2Vec embeddings from .bin file
word2vec_model_path = (r'C:\Users\dines\Downloads\dissertation\word2vec_model.bin')  # Replace with your Word2Vec embeddings .bin file path
word2vec_model = load_word2vec_model_binary(word2vec_model_path)

def preprocess_tweet(tweet, model, max_length):
    # Tokenize the tweet
    tokens = word_tokenize(tweet)
    
    # Convert tweet to embedding
    embedding = tweet_to_embedding(tokens, model, max_length)
    
    return np.array([embedding])

def tweet_to_embedding(tweet, model, max_length):
    embedding = []
    for word in tweet:
        if word in model:
            embedding.append(model[word])
        else:
            embedding.append(np.zeros(len(model['word'])))  # Use the length of vector for unknown words
    if len(embedding) < max_length:
        embedding += [np.zeros(len(model['word']))] * (max_length - len(embedding))
    return np.array(embedding[:max_length])

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Extract text and numerical features from request
    text = data.get('text')
    numerical_features = np.array(data.get('numerical_features')).reshape(1, -1)

    # Preprocess text
    text_embedding = preprocess_tweet(text, word2vec_model, 100)
    
    # Preprocess numerical features
    numerical_features = imputer.transform(numerical_features)
    numerical_features = scaler.transform(numerical_features)

    # Make predictions with both models
    bilstm_prediction = bilstm_model.predict([text_embedding, numerical_features])
    cnn_prediction = cnn_model.predict([text_embedding, numerical_features])

    return jsonify({
        'bilstm_prediction': bilstm_prediction.tolist(),
        'cnn_prediction': cnn_prediction.tolist()
    })

if __name__ == '__main__':
    app.run()
