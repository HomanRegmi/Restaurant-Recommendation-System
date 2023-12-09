import os
import re
import string
import numpy as np
import pandas as pd
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import nltk
import string
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import spacy
import streamlit as st
import warnings
warnings.filterwarnings('ignore')
import base64
import sys

os.environ["STREAMLIT_SERVER_OFFICIAL"] = "false"

nltk.download('stopwords')
from nltk.corpus import stopwords

import spacy

# Load the English language model
nlp = spacy.load("en_core_web_sm")

# Define an initial placeholder DataFrame
df = pd.DataFrame()

df_yelp_business = pd.read_csv('../processed data/df_yelp_business.csv')
df_yelp_review = pd.read_csv('../processed data/df_yelp_review.csv')
aggregated_reviews = pd.read_csv('../processed data/aggregated_reviews.csv')
result_df = pd.read_csv('../processed data/result_df.csv')

categories = result_df.drop('business_id', axis=1).columns.unique()
cities = df_yelp_business['city'].unique().tolist()

def remove_stopwords(words):
        # Initialize an empty list to store the filtered words
        filtered_words = []

        # Iterate through the words and keep only those not in the stop words list
        for word in words:
            if word.lower() not in stop_words:
                filtered_words.append(word)
        return filtered_words

# Set page title and icon
st.set_page_config(
    page_title="Restaurant  Recommendations",
    page_icon="🍔"
)

@st.experimental_memo
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

#If you want a plain backgorund you can delete the code upto line 97
#To change the background image place the URL of the image which you want to place in the background-image: URL 
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
background-image: url("https://e0.pxfuel.com/wallpapers/1000/420/desktop-wallpaper-background-for-restaurants-background-resturant-thumbnail.jpg");
}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)


# Define Streamlit app
st.title("Restaurant Recommendation System")
st.subheader("Project owner: Homan Regmi")

# Define the custom theme for the sidebar, to change the color replace the color code in background-color
custom_theme = """
<style>
[data-testid="stSidebar"] {
    background-color: #E3CB92; /* Coffee brown color */
}
[data-testid="stSidebar"][aria-expanded="true"] {
    box-shadow: 1px 0 5px rgba(0, 0, 0, 0.1);
}
</style>
"""

# Apply the custom theme to the Streamlit app
st.markdown(custom_theme, unsafe_allow_html=True)

# Sidebar with user inputs
st.sidebar.header("User Inputs")

# Create sidebar widgets for user input
city = st.sidebar.selectbox("Select a City", cities)
Category = st.sidebar.multiselect("Select food Categories", categories)
stars = st.sidebar.slider("Business stars", 0.0, 5.0, 0.0, 0.1)
user_input = st.sidebar.text_input("Enter your review to match")

# User interacts with the app
if st.sidebar.button("Submit"):    
    
    if not Category:
        raise ValueError("Error: Atleast one category should be selected")
    else:      
        # Split the input text into words
        user_input = user_input.title()
        words = user_input.split()

        # Check if the input contains at least 5 words
        if 0 < len(words) < 5:
            st.error("Error: Minimum 5 words are required in the input text.")
        else:
            # Create the input DataFrame for prediction
            ab = pd.DataFrame({category: [1] for category in Category})

            # Column name to exclude
            column_to_exclude = 'business_id'

            # Select all columns except the one to exclude
            X_train = result_df.drop(columns=[column_to_exclude])

            # Fill remaining columns with 0
            for column in X_train.columns:
                if column not in ab.columns:
                    ab[column] = 0

            # Reorder columns to match X_train
            ab = ab[X_train.columns]

            columns_to_encode = X_train.columns

            # Convert categorical columns to numerical using One-Hot Encoding
            abc_encoded = pd.get_dummies(result_df, columns=columns_to_encode).drop('business_id', axis=1)
            ab_encoded = pd.get_dummies(ab, columns=columns_to_encode)

            # Make sure both DataFrames have the same columns by aligning them
            abc_encoded, ab_encoded = abc_encoded.align(ab_encoded, axis=1, fill_value=0)

            # Initialize and fit the KNN model
            knn_model = NearestNeighbors(n_neighbors=10)  # Change the number of neighbors as needed
            knn_model.fit(abc_encoded)  # Drop the 'business_id' column for modeling

            # Find the k-nearest neighbors for your input data
            distances, indices = knn_model.kneighbors(ab_encoded)

            # Get the recommended business_ids
            recommendations = result_df.iloc[indices[0]]['business_id']

            # Filter df_yelp_business to get the rows with business_ids in recommendations
            recommended_businesses = df_yelp_business[df_yelp_business['business_id'].isin(recommendations)]

            tfidfVect = TfidfVectorizer()
            tfidf = tfidfVect.fit_transform(aggregated_reviews['text'])

            agg_tfidfVect = TfidfVectorizer()
            agg_tfidfVect = agg_tfidfVect.fit(aggregated_reviews['text'])


            # Define a list of common stop words
            stop_words = ["a", "the","there","behind", "in", "and", "on", "with", "at", "an", "of", "for", "i", "want", "to", "is", "you", "it"]

            filtered_words = remove_stopwords(words)

            # Join the filtered words to form the output text
            user_input_text = " ".join(filtered_words)
            user_input_text = [user_input_text]

            agg_tfidf = agg_tfidfVect.transform(user_input_text)
            feature_names = agg_tfidfVect.get_feature_names_out()

            #Fitting for aggregated text 
            nbrs = NearestNeighbors(n_neighbors=10).fit(tfidf)

            #Finding nearest neighbours for the input string
            distances, indices = nbrs.kneighbors(agg_tfidf)

            names_similar = pd.Series(indices.flatten()).map(aggregated_reviews.reset_index()['business_id'])

            # Filter df_yelp_business to get the rows with business_ids in recommendations
            recommended_businesses_1 = df_yelp_business[df_yelp_business['business_id'].isin(names_similar)]

            Final_df = pd.concat([recommended_businesses, recommended_businesses_1], axis=0)

            # If you want to reset the index of the combined DataFrame
            Final_df = Final_df.reset_index(drop=True)

            # Filter recommended_businesses for businesses in Toronto with 3.0 stars
            filtered_businesses = Final_df[(Final_df['city'] == city) & (Final_df['stars'] >= stars)]

            recommendations = filtered_businesses.merge(
                    aggregated_reviews[aggregated_reviews['average_stars'] >= stars][['business_id', 'average_stars']],
                    on='business_id',
                    how='inner'
                )
            
            #Choose the columns which ever you want to display
            recommendations = recommendations[['business_id','name','address','average_stars']]

            # Output recommendations
            st.subheader("Recommendations")
            # Create a container to display recommendations
            recommendations_container = st.container()

            # Iterate through the recommendations and display them
            for index, row in recommendations.iterrows():
                with recommendations_container:
                    st.markdown('<div style="border: 1px solid #2196F3; border-radius: 5px; padding: 2.5px; margin: 5px 0; box-shadow: 2.5px 2.5px 2.5px #888888; background-color: #F5F5F5;">', unsafe_allow_html=True)
                    st.markdown(f'<div style="font-size: 24px; font-weight: bold;">{row["name"]} ({row["business_id"]})</div>', unsafe_allow_html=True)
                    st.markdown(f'<div style="font-size: 18px;">Address: {row["address"]}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div style="font-size: 18px;">Average Stars: {row["average_stars"]}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

            # Add space between recommendation cards
            st.markdown('<hr>', unsafe_allow_html=True)

if "Submit" in st.session_state and st.session_state.Submit:
    st.dataframe(df)