from flask import Flask, request, jsonify
import tensorflow as tf
import pickle
import numpy as np

app = Flask(__name__)
model = tf.keras.models.load_model('models/model.h5')
with open('models/tokenizer.pickle','rb') as handle:
    tokenizer = pickle.load(handle)

@app.route('/predict', methods = ['POST'])
def predict():
    data = request.json
    text = data['text']

    sequences = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen = 100)

    prediction = model.predic(padded)[0][0]
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    confidence = float(prediction) if sentiment =='Positive' else float(1 - prediction)

    return jsonify({
        'sentiment': sentiment,
        'confidence' : confidence
    })

if __name__=="__main__":
    app.run(debug = True)