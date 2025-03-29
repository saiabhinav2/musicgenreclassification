from flask import Flask, render_template, request, redirect, url_for
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the pre-trained model
# Use absolute path to ensure the correct file is being loaded
model = tf.keras.models.load_model('C:/Users/sai abhinav/OneDrive/Desktop/miniproject/models/trained_model.h5')

# Load the dataset and prepare the label encoder and scaler
df = pd.read_csv('features_3_sec.csv')
df=df.drop(labels="filename",axis=1)

# Encode the labels (target variable)
class_encod = df.iloc[:, -1]  # Assuming the last column is the label
converter = LabelEncoder()
y = converter.fit_transform(class_encod)

# Drop the label column to get the feature matrix (X)
X = df.drop(labels="label", axis=1)  # Assuming the last column is the label

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Feature extraction function
def extract_features(audio_file):
    # Load the audio file (e.g., rock.wav)
    y, sr = librosa.load(audio_file, duration=3)  # Load 30 seconds of the audio file

    # Initialize a dictionary to hold the features
    features = {}

    # Extract the length
    features['length'] = int(np.size(y))

    # Chroma feature (STFT)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    features['chroma_stft_mean'] = np.mean(chroma_stft)
    features['chroma_stft_var'] = np.var(chroma_stft)

    # RMS energy
    rms = librosa.feature.rms(y=y)
    features['rms_mean'] = np.mean(rms)
    features['rms_var'] = np.var(rms)

    # Spectral centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features['spectral_centroid_mean'] = np.mean(spectral_centroid)
    features['spectral_centroid_var'] = np.var(spectral_centroid)

    # Spectral bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
    features['spectral_bandwidth_var'] = np.var(spectral_bandwidth)

    # Spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features['rolloff_mean'] = np.mean(rolloff)
    features['rolloff_var'] = np.var(rolloff)

    # Zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)
    features['zero_crossing_rate_mean'] = np.mean(zcr)
    features['zero_crossing_rate_var'] = np.var(zcr)

    # Harmony and percussive features
    harmony = librosa.effects.harmonic(y)
    percussive = librosa.effects.percussive(y)
    features['harmony_mean'] = np.mean(harmony)
    features['harmony_var'] = np.var(harmony)
    features['perceptr_mean'] = np.mean(percussive)
    features['perceptr_var'] = np.var(percussive)

    # Tempo (BPM)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features['tempo'] = tempo

    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(1, 21):
        features[f'mfcc{i}_mean'] = np.mean(mfccs[i-1])
        features[f'mfcc{i}_var'] = np.var(mfccs[i-1])

    return features

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle the file upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)
    
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join('uploads', filename)
        file.save(file_path)

        # Step 1: Extract features from the uploaded file
        features = extract_features(file_path)

        # Convert the extracted features into a dataframe
        features_df = pd.DataFrame([features])

        # Standardize the extracted features
        features_scaled = scaler.transform(features_df)

        # Step 2: Predict the label using the pre-trained model
        prediction = model.predict(features_scaled)
        predicted_index = np.argmax(prediction, axis=1)

        # Step 3: Convert the predicted index back to the genre label
        predicted_label = converter.inverse_transform(predicted_index)[0]

        return render_template('index.html', prediction=predicted_label,filename=filename)

if __name__ == '__main__':
    app.run(debug=True)
