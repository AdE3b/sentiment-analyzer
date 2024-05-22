import streamlit as st
import pickle

# Load the model and vectorizer from files
model_filename = 'log_reg_classifier.pkl'
vectorizer_filename = 'vectorizer.pkl'

with open(model_filename, 'rb') as model_file:
    loaded_model = pickle.load(model_file)

with open(vectorizer_filename, 'rb') as vectorizer_file:
    loaded_vectorizer = pickle.load(vectorizer_file)

# Function to predict sentiment from a user input review
def predict_sentiment(review):
    # Transform the user input using the loaded vectorizer
    review_vect = loaded_vectorizer.transform([review])
    # Predict using the loaded model
    predicted_sentiment = loaded_model.predict(review_vect)[0]
    # Get probabilities of each class (negative and positive)
    probabilities = loaded_model.predict_proba(review_vect)[0]
    probability_negative = probabilities[0]
    probability_positive = probabilities[1]
    # Convert prediction to sentiment label
    sentiment = 'Positive' if predicted_sentiment == 1 else 'Negative'
    return sentiment, probability_negative, probability_positive

# Streamlit app
st.title('Sentiment Analysis')

st.write("Enter a text to check its sentiment:")
new_review = st.text_input("")

if st.button('Analyze Sentiment'):
    if new_review:
        sentiment, prob_neg, prob_pos = predict_sentiment(new_review)
        st.write("### Predicted Sentiment:", sentiment)
        st.write("#### Probability Negative:", prob_neg)
        st.write("#### Probability Positive:", prob_pos)
    else:
        st.write("Please enter a review text to analyze.")


