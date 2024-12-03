import pickle
import streamlit as st
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import pandas as pd

# Download required NLTK data (if not already downloaded)
nltk.download("stopwords")
nltk.download("punkt")

# Load pre-trained TF-IDF vectorizer
with open("tfidf_vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

# Load pre-trained Logistic Regression model
with open("logistic_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Function to preprocess the input text (from ReviewText column)
def preprocess_text(ReviewText):
    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(ReviewText.lower())  # Convert to lowercase and tokenize
    # Remove non-alphabetic tokens and stop words
    return " ".join([word for word in tokens if word.isalpha() and word not in stop_words])

# Function to predict sentiment
def predict_sentiment(ReviewText):
    # Preprocess the user input review
    processed_review = preprocess_text(ReviewText)
    
    # Transform the processed review into the same feature space as during training
    features = vectorizer.transform([processed_review])
    
    # Predict sentiment using the Logistic Regression model
    prediction = model.predict(features)[0]
    
    # Return sentiment label (you can modify this to return more detailed output if needed)
    return prediction

# Streamlit interface
st.title("Sentiment Analysis for Product Reviews")

# Text area for user input (we assume this is from ReviewText)
review_input = st.text_area("Enter a review:")

# Button to trigger prediction
if st.button("Predict Sentiment"):
    if review_input.strip():  # Check if input is not empty
        # Call the prediction function
        SentimentLabel = predict_sentiment(review_input)
        st.write(f"The sentiment of the review is: {SentimentLabel}")
    else:
        st.write("Please enter a review to predict sentiment.")

# Optionally, if you want to load a sample of reviews to test (for the entire dataset or a sample)
# This will help you test the model directly from a sample dataset

# Replacing st.cache with st.cache_data to handle caching of the dataset
@st.cache_data
def load_data():
    return pd.read_csv("reviews.csv")  # Load your actual dataset

# Display a preview of the dataset if needed
if st.button("Show Sample Data"):
    df = load_data()
    st.write(df.head())  # Display the first few rows of the dataset to the user
