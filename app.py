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

import streamlit as st


st.title('Sentiment Analysis')

st.write("Enter a text to check its sentiment:")
new_review = st.text_input("")

if st.button('Analyze Sentiment'):
    if new_review:
        sentiment, prob_neg, prob_pos = predict_sentiment(new_review)
        
        if sentiment == 'Positive':
            sentiment_color = "green"
        else:
            sentiment_color = "red"
        
        # Display the sentiment with the specified color
        st.markdown(f"### Predicted Sentiment: <span style='color:{sentiment_color}'>{sentiment}</span>", unsafe_allow_html=True)
        
        # Display the probabilities in white color
        st.markdown(f"#### Probability Negative: <span style='color:white'>{prob_neg}</span>", unsafe_allow_html=True)
        st.markdown(f"#### Probability Positive: <span style='color:white'>{prob_pos}</span>", unsafe_allow_html=True)
    else:
        st.write("Please enter a review text to analyze.")
