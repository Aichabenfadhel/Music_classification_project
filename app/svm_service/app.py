import streamlit as st
import tensorflow as tf
import librosa
import numpy as np
from tensorflow.image import resize
from tensorflow.keras.models import load_model
import os

@st.cache_resource()
def load_model():
    try:
        model = tf.keras.models.load_model("Trained_model.h5")
        model.summary()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load and preprocess audio file
def load_and_preprocess_file(file_path, target_shape=(150, 150)):
    data = []
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    chunk_duration = 4
    overlap_duration = 2
    # Convert duration to samples
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate

    # Calculate the number of chunks
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1

    # Iterate over each chunk
    for i in range(num_chunks):
        # Calculate start and end indices of the chunk
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
        # Extract the chunk of audio
        chunk = audio_data[start:end]
        # Mel-spectrogram part
        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
        # Resize matrix based on provided target shape
        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        # Append data to list
        data.append(mel_spectrogram)

    return np.array(data)

def model_prediction(X_test):
    trained_model = load_model()
    if trained_model is not None:
        try:
            y_pred = trained_model.predict(X_test)
            predicted_categories = np.argmax(y_pred, axis=1)
            unique_elements, counts = np.unique(predicted_categories, return_counts=True)
            max_count = np.max(counts)
            max_elements = unique_elements[counts == max_count]
            return max_elements[0]
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return None
    else:
        st.error("Model could not be loaded. Please check the model path or file.")
        return None

st.header("Music Genre Classification")

# Upload audio file
test_mp3 = st.file_uploader("Upload an audio file", type=["mp3"])

# Save the file locally if uploaded
if test_mp3 is not None:
    filepath = f"Test_Music/{test_mp3.name}"
    with open(filepath, "wb") as f:
        f.write(test_mp3.getbuffer())

# Play audio button
if st.button("Play Audio") and test_mp3 is not None:
    st.audio(filepath)

# Predict button
if st.button("Predict") and test_mp3 is not None:
    with st.spinner("Please wait ..."):
        X_test = load_and_preprocess_file(filepath)
        result_index = model_prediction(X_test)
        if result_index is not None:
            label = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
            st.markdown("**:blue[Model prediction]: It's a :red[{}] music**".format(label[result_index]))
        else:
            st.error("Prediction failed due to an error with the model.")

# Ensure Test_Music directory exists
os.makedirs("Test_Music", exist_ok=True)
