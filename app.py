import streamlit as st
import joblib
import pickle
import nltk
import difflib
from text_cleaner import TextCleaner   

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin


# Download NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')


# LOAD MODEL
model = pickle.load(open("drug_condition_model.pkl", "rb"))
drug_list = joblib.load("drug_list.pkl")


# Convert drug list to lowercase
drug_list_lower = [d.lower() for d in drug_list]


# Streamlit UI
st.title("💊 Drug Review Condition Predictor")

st.write("Predict disease condition from drug name and review.")


st.sidebar.title("About Project")

st.sidebar.info(
"""
Drug Condition Prediction using NLP.

Model: TF-IDF + Machine Learning  

Goal: Predict medical condition from review text.
"""
)

# ---------- Drug Input Mode ----------

input_method = st.radio(
    "Choose Drug Input Method",
    ["Select from dataset", "Enter manually"]
)


# Dropdown drug selection
if input_method == "Select from dataset":
    drug = st.selectbox("Select Drug Name", drug_list)

# Custom drug input
else:
    drug = st.text_input("Enter Drug Name")


# Review input
review = st.text_area("Enter Review")


# ---------- Prediction ----------

if st.button("Predict"):

    if drug.strip() == "":
        st.warning("Please enter the drug name.")

    elif review.strip() == "":
        st.warning("Please enter the review.")

    elif len(review.split()) < 3:
        st.warning("Review must contain at least 3 words.")

    else:

        # If manual drug input, check similarity
        if input_method == "Enter manually":

            if drug.lower() not in drug_list_lower:

                matches = difflib.get_close_matches(drug, drug_list, n=1, cutoff=0.6)

                st.warning("Drug not found in dataset.")

                if matches:
                    st.info(f"Did you mean: **{matches[0]}** ?")

        # Give higher priority to review
        text = drug + " " + review + " " + review + " " + review

        prediction = model.predict([text])[0]

        st.success(f"Predicted Condition: **{prediction}**")