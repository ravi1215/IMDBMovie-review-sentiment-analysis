from keras.models import load_model
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np

model = load_model("model.h5")
tokenizer = joblib.load("tokenizer.pkl")

def predictive_system(review):
    sequences = tokenizer.texts_to_sequences([review])
    padded_sequence = pad_sequences(sequences, maxlen=200)
    prediction = model.predict(padded_sequence)[0][0]  
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    
    if prediction <= 0.2:
        level = "Bad"
    elif prediction <= 0.4:
        level = "Average"
    elif prediction <= 0.6:
        level = "Good"
    elif prediction <= 0.8:
        level = "Very Good"
    else:
        level = "Outstanding"
    
    labels = ['Negative', 'Positive']
    scores = [1 - prediction, prediction]  
    plt.bar(labels, scores, color=['red', 'green'])
    plt.xlabel("Sentiment")
    plt.ylabel("Confidence Score")
    plt.title("Sentiment Analysis")
    plt.ylim(0, 1)
    
    plt_file = "/content/prediction_plot.png"
    plt.savefig(plt_file)
    plt.close()

    return f"Sentiment: {sentiment}\nLevel: {level}", plt_file

title = "IMDB MOVIE SENTIMENT ANALYSIS APPLICATION"
description = "Enter a movie review to see its sentiment (Positive/Negative) and rating (Bad, Average, Good, Very Good, Outstanding)."

app = gr.Interface(
    fn=predictive_system,
    inputs=gr.Textbox(label="Enter the Review"), 
    outputs=[
        gr.Textbox(label="Output"),  
        gr.Image(label="Sentiment Confidence Graph")  
    ],
    title=title,
    description=description
)

app.launch(share=True)

