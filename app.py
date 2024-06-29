import streamlit as st
import pandas as pd
import chardet
import re
import json
from io import StringIO
import nltk
from nltk.corpus import stopwords
from pycaret.classification import *
import os
import joblib
import streamlit.components.v1 as components

def set_custom_style():
    st.markdown(
        """
        <style>
        .main {
            background-color: #FFFFFF; /* Set your desired background color */
        }
        /* Apply black color to all text elements */
        h1, h2, h3, h4, h5, h6, label, div, span, .stMarkdown {
            color: black !important; /* Set text color to black */
        }
        /* Apply custom styles to Streamlit button */
        .stButton button, .link-button{
            background-color: MediumSeaGreen !important; /* Green background */
            color: white !important; /* White text */
            text-align: center; /* Center align text */
            display: block; /* Ensure the button fills the container */
            margin: 0 auto; /* Center align button horizontally */
            padding: 0.5rem 1rem; /* Add padding to the button */
            border: none; /* Remove default border */
            transition: background-color 0.1s ease, border-color 0.3s ease; /* Smooth transition for hover effect */
        }
        .stButton button:hover,  .link-button:hover{
            background-color: white !important; /* White background on hover */
            color: black !important; /* Black text on hover */
            border: 2px solid #556B2F !important; /* Dark green border on hover */
        }
        /* Style for the text input box */
        .stTextInput input {
            background-color: lightgrey !important; /* Light grey background */
            color: black !important; /* Brown text */
            border: 2px solid #556B2F !important; /* Initial border color */
        }
        .stTextInput input:focus {
            border: 2px solid #556B2F !important /* White border on focus */
            color: black !important; /* White text on focus */
            background-color: lightgrey !important; /* Grey background on focus */
        }
        .stTextInput input:hover {
            border-color: white; /* Set the input border color on hover */
        }
        ::selection {
            background-color: white !important; /* Grey background for selected text */
            color: black !important; /* White text for selected text */
        }
        
        
        </style>
        """,
        unsafe_allow_html=True
    )

set_custom_style()

st.title('Welcome to DepressioCare')
# Initialize processed_input


# Caching function to load data and model
@st.cache_data
def load_preprocessed_data(csv_paths, json_path):
    def detect_encoding(file_path):
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
        return result['encoding']
    
    # Load and concatenate multiple CSV files
    dfs = []
    for path in csv_paths:
        encoding = detect_encoding(path)
        with open(path, 'r', encoding=encoding, errors='replace') as f:
            content = f.read()
        df = pd.read_csv(StringIO(content))
        dfs.append(df)
    combined_df = pd.concat(dfs, ignore_index=True)

    # Load JSON data
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    # Ensure all lists in the JSON data have the same length
    max_length = max(len(v) if isinstance(v, list) else 1 for v in json_data.values())
    for k, v in json_data.items():
        if isinstance(v, list) and len(v) < max_length:
            json_data[k] = v + [None] * (max_length - len(v))
        elif not isinstance(v, list):
            json_data[k] = [v] * max_length

    df_json = pd.DataFrame(json_data)

    # Replace underscores with spaces
    df_json = df_json.replace('_', ' ', regex=True) 

    # Process signals in the DataFrame
    for column in df_json.columns:
        if df_json[column].dtype == 'object':  # Check if column contains strings
            df_json[column] = df_json[column].apply(lambda x: re.sub(r'[^\w\s]', '', x) if pd.notna(x) else x)

        return combined_df, df_json

@st.cache_resource
def load_model_from_disk(model_path):
    # Load the pre-trained model
    best_model_symptoms_all = joblib.load(model_path)
    return best_model_symptoms_all

# Paths to CSV files and JSON file
csv_paths = [
    r'C:\Users\Acer\OneDrive\Desktop\UKM\Sem 5\FYP\dataset\tweets_final_1_clean.csv',
    r'C:\Users\Acer\OneDrive\Desktop\UKM\Sem 5\FYP\dataset\tweets_final_2_clean.csv',
    r'C:\Users\Acer\OneDrive\Desktop\UKM\Sem 5\FYP\dataset\tweets_final_3_clean.csv',
    r'C:\Users\Acer\OneDrive\Desktop\UKM\Sem 5\FYP\dataset\tweets_final_4_clean.csv',
    r'C:\Users\Acer\OneDrive\Desktop\UKM\Sem 5\FYP\dataset\tweets_final_5_clean.csv',
    r'C:\Users\Acer\OneDrive\Desktop\UKM\Sem 5\FYP\dataset\tweets_final_6_clean.csv'
]
json_file_path = r"C:\Users\Acer\OneDrive\Desktop\UKM\Sem 5\FYP\dataset\depression_lexicon.json"
model_path = r"C:\Users\Acer\OneDrive\Desktop\UKM\Sem 5\FYP\dataset\best_model_symptoms_all_pipeline.pkl"

combined_df, df_json = load_preprocessed_data(csv_paths, json_file_path)
best_model_symptoms_all = load_model_from_disk(model_path)

# Function to remove non-ASCII characters
def remove_non_ascii(text):
    return re.sub(r'[^\x00-\x7F]+', '', text)

# Function to remove punctuation
def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)

# Function to remove links
def remove_links(text):
    return re.sub(r'http\S+|www\S+|https\S+', '', text)

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Define the list of words to keep
words_to_keep = {"i", "me", "she", "mine", "myself", "cant", "your", "not", "down", "do", "have", "so"}

adjusted_stop_words = stop_words - words_to_keep

# Function to remove selective stopwords
def remove_selected_stopwords(text, adjusted_stop_words):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in adjusted_stop_words]
    return ' '.join(filtered_words)

# Function to find depression signals in text
def find_depression_signals(text, signals):
    pattern = r'\b(?:' + '|'.join(signals) + r')\b'
    matches = re.findall(pattern, text)
    return matches

if os.path.exists(model_path):
    # Load the pre-trained model
    best_model_symptoms_all = joblib.load(model_path)

    # Streamlit UI for user input
    user_input = st.text_input('Enter your text based on explaination of your feelings and situations.', placeholder="Enter text...")

    # Initialize processed_input
    processed_input = ""  # Initialize or set to None initially

    # Manage session state
    if 'page' not in st.session_state:
        st.session_state.page = 0
        st.session_state.matched_signals_1 = []
        st.session_state.matched_signals_2 = []
        st.session_state.matched_signals_5 = []
        st.session_state.count_symptoms_all = 0
        st.session_state.prediction_made = False  # Add a flag to track prediction
    

    
    if st.session_state.page == 0 and st.button('Analyze'):
        st.session_state.page = 1
    
    if st.session_state.page == 1 and not st.session_state.prediction_made:

        if user_input:
            # Process user input
            processed_input=""
            processed_input = user_input.lower()
            processed_input = remove_non_ascii(processed_input)
            processed_input = remove_punctuation(processed_input)
            processed_input = remove_links(processed_input)
            processed_input = remove_selected_stopwords(processed_input, adjusted_stop_words)

            # Initialize lists to store matched signals
            matched_signals_1 = []
            matched_signals_2 = []
            matched_signals_5 = []

            # Find matched signals in user input
            for index, row in df_json.iterrows():
                signal_1 = row['signal_1']
                signal_2 = row['signal_2']
                signal_5 = row['signal_5']

                if pd.notna(signal_1):
                    matches_1 = find_depression_signals(processed_input, [signal_1])
                    if matches_1:
                        matched_signals_1.extend(matches_1)

                if pd.notna(signal_2):
                    matches_2 = find_depression_signals(processed_input, [signal_2])
                    if matches_2:
                        matched_signals_2.extend(matches_2)

                if pd.notna(signal_5):
                    matches_5 = find_depression_signals(processed_input, [signal_5])
                    if matches_5:
                        matched_signals_5.extend(matches_5)

            # Store results in session state
            st.session_state.matched_signals_1 = matched_signals_1
            st.session_state.matched_signals_2 = matched_signals_2
            st.session_state.matched_signals_5 = matched_signals_5

            # Print matched signals
            st.subheader('Based on your text, here are some terms that are related with depressive symptoms:')
            st.write(f"You have shown sign of little interest or pleasure in doing things which is: {', '.join(matched_signals_1)}" if matched_signals_1 else "You have no signs of little interest or pleasure in doing things")
            st.write(f"You have shown sign of feeling down, depressed or hopeless which is: {', '.join(matched_signals_2)}" if matched_signals_2 else "You have no signs of feeling down, depressed or hopeless")
            st.write(f"You have shown sign of poor appetite or overeating which is: {', '.join(matched_signals_5)}" if matched_signals_5 else "You have no signs of poor appetite or overeating.")

            # Calculate count of all symptoms
            count_symptoms_all = len(matched_signals_1) + len(matched_signals_2) + len(matched_signals_5)
            st.session_state.count_symptoms_all = count_symptoms_all

            st.write("")
            st.write(f"You have shown {count_symptoms_all} depressive symptoms.")

            if st.button('Next'):
                st.session_state.page = 2
                st.session_state.prediction_made = True

        else:
            st.warning("Please enter a description.")
    
    if st.session_state.page == 2:
        # Predict count_symptoms_all using the model
        count_symptoms_all =st.session_state.count_symptoms_all

        # Predict count_symptoms_all using the model
        new_data_symptoms_all = pd.DataFrame({
            'Tweet_Processed': [processed_input],
            'count_symptoms_all': [count_symptoms_all]  # Placeholder, will be predicted
        })

        # Load the best model
        best_model_symptoms_all_new = load_model_from_disk(model_path)

        # Make predictions on the new data
        predictions_symptoms_all_new = predict_model(best_model_symptoms_all_new, data=new_data_symptoms_all)

        # Display predictions for new data
        #st.subheader('Predictions for New Data')
        #st.write(predictions_symptoms_all_new)
        
        st.write("Here is your result,")
        predict_score = predictions_symptoms_all_new['prediction_score'].iloc[0]

        # Print interpretation based on prediction
        label = predictions_symptoms_all_new['prediction_label'].iloc[0]
        if label == 1:
            st.write(f"Based on prediction score of {predict_score:.2f}, you MAY HAVE depression. Please seek professional help for further information. ")
        else:
            st.write(f"Based on prediction score of {predict_score:.2f}, you MAY NOT HAVE depression. Please seek professional help for further information.")
        
        if st.button('Restart'):
            st.session_state.page = 0
            st.session_state.matched_signals_1 = []
            st.session_state.matched_signals_2 = []
            st.session_state.matched_signals_5 = []
            st.session_state.count_symptoms_all = 0
            st.session_state.prediction_made = False
            st.experimental_rerun()

else:
    st.error("Model file not found. Please ensure the model file 'best_model_symptoms_all_pipeline.pkl' exists.")

button_link_html = """
    <button onclick="window.open('https://a188254.wixsite.com/depressiocare', '_blank')" class="link-button" >Back</button>
    """
components.html(button_link_html)