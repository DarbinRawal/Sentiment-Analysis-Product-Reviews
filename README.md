# Sentiment Analysis for Product Reviews

This project uses a pre-trained Logistic Regression model to perform sentiment analysis on product reviews. It classifies reviews as either positive or negative.

## Features
- An intuitive Streamlit interface to input reviews.
- Pre-trained sentiment analysis model using Logistic Regression.
- Customizable for analyzing other product reviews.

## Files
- `app.py`: The main Streamlit application script.
- `reviews.csv`: Dataset used for training the model.
- `tfidf_vectorizer.pkl`: Pre-trained TF-IDF vectorizer.
- `logistic_model.pkl`: Pre-trained Logistic Regression model.
- `requirements.txt`: List of Python dependencies.

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd Sentiment-Analysis-Product-Reviews
2. **Install dependencies**:
   pip install -r requirements.txt

3. **Run the application**:
   streamlit run app.py

4. Enter a review on the Streamlit app to get a sentiment prediction (positive or negative).
   
