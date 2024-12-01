import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

# Download necessary NLTK resources (only the first time you run this script)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize the lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Step 1: Load your dataset
data = pd.read_excel('/Users/vleramurtezi/Desktop/Thesis/Data/LDA/LDA_dataset.xlsx')  # Replace with your file path
# Assuming the review text is in a column named 'review_text'
reviews = data['review_text'].tolist()

# Step 2: Define a preprocessing function
def preprocess_text(text):
    # Ensure the text is a string
    if not isinstance(text, str):
        return []
    
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation and special characters using regex
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords and lemmatize each word
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return tokens

# Step 3: Apply preprocessing to each review
data['preprocessed_review'] = [preprocess_text(review) for review in reviews]

# Step 4: Save the dataset with preprocessed reviews
data.to_excel('/Users/vleramurtezi/Desktop/Thesis/Data/LDA/preprocessed_reviews.xlsx', index=False)  # or data.to_csv('preprocessed_reviews.csv', index=False)

# Previewing the results to confirm
print("Sample of preprocessed reviews:")
print(data[['review_text', 'preprocessed_review']].head())
