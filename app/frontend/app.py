import streamlit as st
from app.svm_service.app import load_and_preprocess_file, model_prediction
import os

st.header("Music Genre Classification")
st.image("image.jpg", use_column_width=True)

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
        label = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
        st.markdown("**:blue[Model prediction]: It's a :red[{}] music**".format(label[result_index]))

# Ensure Test_Music directory exists
os.makedirs("Test_Music", exist_ok=True)
