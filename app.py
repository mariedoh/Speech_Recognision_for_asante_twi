import streamlit as st
from functions import *
import librosa
import torch
from transformers import WhisperProcessor



def load_audio(file_path, target_sr=16000):
    audio, sr = librosa.load(file_path, sr=target_sr)
    return audio

st.write("Speech Recognition - Asante Twi")
st.write("Upload a twi audio file")

processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model, tokenizer = load_model()

uploaded_file = st.file_uploader("Upload a twi audio file", type=["mp3", "wav", "m4a"])
if uploaded_file:
    st.write(f"Processing file: {uploaded_file.name}")

     # Save the uploaded file temporarily
    with open("temp_audio_file", "wb") as f:
        f.write(uploaded_file.read())
    # Load and preprocess the audio file
    audio_data = load_audio("temp_audio_file", target_sr=16000)
    
    # Process the audio using the WhisperProcessor
    inputs = processor(audio_data, return_tensors="pt", sampling_rate=16000)
    
    # Move model and inputs to the correct device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    # Generate transcription with your fine-tuned model
    with torch.no_grad():
        predicted_ids = model.generate(inputs["input_features"])
    
    # Decode the predicted transcription using your tokenizer
    transcription = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
    
    # Output the transcription
    st.write("Transcription:")
    st.text_area("Transcribed Text", transcription[0].encode('latin1', errors='ignore').decode('utf-8', errors='ignore'), height=200)