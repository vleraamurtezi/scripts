import pandas as pd
from gensim.corpora.dictionary import Dictionary
import pickle

# Load the dataset
file_path = "/Users/vleramurtezi/Desktop/Thesis/Data/LDA/preprocessed_reviews.xlsx"
data = pd.read_excel(file_path)

# Convert the 'preprocessed_review' column into a list of tokenized reviews
reviews_list = data['preprocessed_review'].apply(eval).tolist()  # eval if reviews are saved as strings of lists

# Create dictionary and corpus
dictionary = Dictionary(reviews_list)
corpus = [dictionary.doc2bow(review) for review in reviews_list]

# Save dictionary and corpus for later use
with open('dictionary.pkl', 'wb') as f:
    pickle.dump(dictionary, f)

with open('corpus.pkl', 'wb') as f:
    pickle.dump(corpus, f)

print("Dictionary and Corpus saved successfully.")
