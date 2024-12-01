import pandas as pd
import re

# Load the dataset
file_path = r'/Users/vleramurtezi/Desktop/Thesis/Data/Dataset_Goodreads.xlsx'
data = pd.read_excel(file_path)

# Drop rows with missing reviews
data = data.dropna(subset=['Review'])

# Remove special characters, extra whitespace, and HTML tags
def clean_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s]', '', text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Apply cleaning to the Review column
data['Review'] = data['Review'].apply(lambda x: clean_text(str(x)) if isinstance(x, str) else x)

# Remove duplicate reviews if necessary
data = data.drop_duplicates(subset=['Review'])

# Save the cleaned data (optional)
cleaned_file_path = r'/Users/vleramurtezi/Desktop/Thesis/Data/Cleaned_Reviews.xlsx'
data.to_excel(cleaned_file_path, index=False)

print("Data cleaning complete. Cleaned data saved to", cleaned_file_path)
