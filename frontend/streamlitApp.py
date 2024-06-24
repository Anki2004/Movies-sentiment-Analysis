import streamlit as st
import requests

st.title("Sentiment Analysis")

text = st.text_area("Enter the text for analysis", height = 100)

if st.button('Analyze Sentiment'):
    if text:
        response = requests.post('http://localhost:5000/predict', json={'text':text})
        print(response.text)
        result = response.json()

        st.write(f"sentiment: {result['sentiment']}")
        st.write(f"confidence: {result['confidence']: .2f}")

    else:
        st.write("Please enter some text")