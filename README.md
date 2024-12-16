# IMDB Movie Review Sentiment Analysis

This project involves analyzing IMDB movie reviews to determine the sentiment (positive or negative) and rating (Bad, Average, Good, Very Good, Outstanding) of the reviews using Deep learning techniques.
The sentiment analysis model is based on classification and utilizes one-hot encoding for text data preprocessing.

## Project Structure

The project consists of two main folders:

1. **imdb** - Contains:
    - `imdb_sa.ipynb`: Jupyter notebook for training the sentiment analysis model.
    - `imdb_dataset.csv`: Dataset used for training the model.
    - `model.h5`: Trained deep learning model.
    - `tokenizer.pkl`: Tokenizer used for text processing.
    
2. **movie_sentiment_analysis** - Contains:
    - `app.py`: Python script to launch the Gradio web interface for the sentiment analysis app.
    - `requirements.txt`: Python dependencies for the application.
    - `model.h5`: Trained model file.
    - `tokenizer.pkl`: Tokenizer file.

## Features:
- **Sentiment Prediction**: Classifies movie reviews as either Positive or Negative.
- **Rating Prediction**: Assigns a rating based on sentiment score: Bad, Average, Good, Very Good, or Outstanding.
- **Visualization**: Confidence scores of sentiment analysis are displayed in a bar graph.

## Technologies Used:
- **TensorFlow/Keras**: For building and training the sentiment classification model.
- **One-Hot Encoding**: Used for text data preprocessing.
- **Gradio**: For building the user interface and allowing real-time interaction with the model.
- **Matplotlib**: For plotting the sentiment analysis confidence graph.

## Usage:

1. Open the app in your browser (it will show the link after running `app.py`).
2. Enter a movie review in the input box.
3. The model will predict the sentiment (Positive/Negative) and provide a rating (Bad, Average, Good, Very Good, Outstanding).
4. A confidence graph will also be displayed showing the sentiment classification results.

## Model Training:

The model was trained on the IMDB dataset, using one-hot encoding for text preprocessing and classification for sentiment analysis.


## Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/ravi1215/IMDBMovie-review-sentiment-analysis.git
    cd IMDBMovie-review-sentiment-analysis
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Make sure to have the following files:
    - `model.h5`
    - `tokenizer.pkl`

## Run the Application

To start the Gradio app and run the sentiment analysis application, execute:

```bash
python app.py
```
