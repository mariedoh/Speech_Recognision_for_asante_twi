import streamlit as st

st.write("Speech Recognition - Asante Twi")
st.write("Upload a twi audio file")

uploaded_file = st.file_uploader("Upload a twi audio file")
if uploaded_file:
    #process the file and work with the model
    pass