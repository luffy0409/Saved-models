import os
import numpy as np
import joblib
from keras.models import load_model
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer, BertModel
import torch
import nltk
from flask import Flask, request, jsonify

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

app = Flask(__name__)

# Define the directory path
drive_base_path = '.'

# Load models and preprocessors
cnn_bilstm_model = load_model(os.path.join(drive_base_path, 'cnn_bilstm_model.h5'))
cnn_1d_model = load_model(os.path.join(drive_base_path, 'cnn_1d_model.h5'))
w2v_model = Word2Vec.load(os.path.join(drive_base_path, 'word2vec_model.bin'))
scaler = joblib.load(os.path.join(drive_base_path, 'scaler.pkl'))
imputer = joblib.load(os.path.join(drive_base_path, 'imputer.pkl'))

# Load custom BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained(os.path.join(drive_base_path, 'tokenizer'))
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Load the custom BERT model state dictionary
custom_bert_state_dict = torch.load(os.path.join(drive_base_path, 'custom_bert_model.pth'), map_location=torch.device('cpu'))

# Extract only the keys relevant to the BERT model
bert_state_dict = {k.replace('bert.', ''): v for k, v in custom_bert_state_dict.items() if k.startswith('bert.')}

# Load the state dictionary into the BERT model
bert_model.load_state_dict(bert_state_dict)

# Set the model to evaluation mode
bert_model.eval()

# Function to convert a tweet to its Word2Vec embeddings
def tweet_to_embedding(tweet, model, max_length):
    embedding = []
    for word in tweet:
        if word in model.wv:
            embedding.append(model.wv[word])
        else:
            embedding.append(np.zeros(model.vector_size))
    if len(embedding) < max_length:
        embedding += [np.zeros(model.vector_size)] * (max_length - len(embedding))
    return np.array(embedding[:max_length])

# Function to preprocess new tweets
def preprocess_tweet(tweet, w2v_model, max_length, followers, following, action):
    tokens = word_tokenize(tweet)
    embedding = tweet_to_embedding(tokens, w2v_model, max_length)
    
    # Numerical features
    numerical_features = np.array([[followers, following, action]])
    numerical_features = imputer.transform(numerical_features)
    numerical_features = scaler.transform(numerical_features)
    
    return np.array([embedding]), numerical_features

# Function to preprocess new tweets for BERT
def preprocess_tweet_bert(tweet, tokenizer, max_length):
    inputs = tokenizer(tweet, return_tensors='pt', max_length=max_length, truncation=True, padding='max_length')
    return inputs['input_ids'], inputs['attention_mask']

# Function to predict spam or not
def predict_spam(tweet, cnn_bilstm_model, cnn_1d_model, w2v_model, bert_model, tokenizer, max_length, followers, following, action):
    tweet_embedding, numerical_features = preprocess_tweet(tweet, w2v_model, max_length, followers, following, action)
    
    # Preprocess tweet for BERT
    input_ids, attention_mask = preprocess_tweet_bert(tweet, tokenizer, max_length)
    
    # Make predictions
    bilstm_prediction = cnn_bilstm_model.predict([numerical_features, tweet_embedding])
    cnn1d_prediction = cnn_1d_model.predict([tweet_embedding, numerical_features])
    
    # BERT model prediction
    with torch.no_grad():
        bert_outputs = bert_model(input_ids, attention_mask=attention_mask)
        bert_prediction = torch.sigmoid(bert_outputs.pooler_output).numpy()
    
    # Weighted average of the predictions
    final_prediction = (0.3 * bilstm_prediction + 0.3 * cnn1d_prediction + 0.4 * bert_prediction)
    
    # Determine if it's spam or not based on final_prediction
    is_spam = (final_prediction > 0.5).astype(int)[0][0]  # Extract the single boolean value
    prediction_label = 'Spam' if is_spam else 'Not Spam'
    
    return prediction_label

# Flask route for prediction
@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        new_tweet = request.form['tweet']
        followers = float(request.form['followers'])
        following = float(request.form['following'])
        action = float(request.form['action'])
        
        max_length = 110  # Adjust based on your model's input requirements
        result = predict_spam(new_tweet, cnn_bilstm_model, cnn_1d_model, w2v_model, bert_model, tokenizer, max_length, followers, following, action)
        return jsonify({'tweet': new_tweet, 'prediction': result})
    return '''
        <form method="post">
            <label for="tweet">Enter your tweet:</label><br>
            <input type="text" id="tweet" name="tweet"><br>
            <label for="followers">Followers:</label><br>
            <input type="number" id="followers" name="followers"><br>
            <label for="following">Following:</label><br>
            <input type="number" id="following" name="following"><br>
            <label for="action">Action:</label><br>
            <input type="number" id="action" name="action"><br>
            <input type="submit" value="Predict">
        </form>
    '''

if __name__ == '__main__':
    app.run(debug=True)
