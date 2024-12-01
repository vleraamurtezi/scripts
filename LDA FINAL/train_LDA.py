import pandas as pd
import gensim
from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import multiprocessing

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Enhanced pre-processing function for additional cleaning if needed
def preprocess_text(text):
    # Remove non-word characters and numbers
    text = re.sub(r'\W', ' ', text)  # Remove non-word characters
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    tokens = word_tokenize(text.lower())
    # Lemmatize and remove stopwords and single-character tokens
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english') and len(word) > 1]
    return tokens

# Function to train LDA with coherence evaluation
def train_lda_model(corpus, dictionary, tokens, num_topics, passes=10):
    lda_model = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, passes=passes)
    
    # Adjust the CoherenceModel to avoid multiprocessing
    coherence_model = CoherenceModel(model=lda_model, texts=tokens, dictionary=dictionary, coherence='c_v', processes=1)
    coherence_score = coherence_model.get_coherence()
    return lda_model, coherence_score

def main():
    # Load your preprocessed dataset
    data_path = '/Users/vleramurtezi/Desktop/Thesis/Data/LDA/preprocessed_reviews.xlsx'
    df = pd.read_excel(data_path)

    # Tokenize and preprocess the reviews using the 'preprocessed_review' column
    df['tokens'] = df['preprocessed_review'].apply(preprocess_text)

    # Create dictionary and corpus for LDA
    dictionary = corpora.Dictionary(df['tokens'])
    corpus = [dictionary.doc2bow(text) for text in df['tokens']]

    # Try different numbers of topics to find the best coherence score
    best_model = None
    best_coherence = 0
    for num_topics in range(3, 11):  # Adjust range as needed
        model, coherence = train_lda_model(corpus, dictionary, df['tokens'], num_topics=num_topics)
        if coherence > best_coherence:
            best_model = model
            best_coherence = coherence
        print(f"Number of topics: {num_topics}, Coherence Score: {coherence}")

    # Display the best number of topics based on coherence
    print(f"Best model has {best_model.num_topics} topics with a coherence score of {best_coherence}")

    # Save the best model, dictionary, and corpus for later use
    best_model.save('/Users/vleramurtezi/Desktop/Thesis/Data/LDA FINAL/best_lda_model.gensim')
    dictionary.save('/Users/vleramurtezi/Desktop/Thesis/Data/LDA FINAL/dictionary.gensim')
    gensim.corpora.MmCorpus.serialize('/Users/vleramurtezi/Desktop/Thesis/Data/LDA FINAL/corpus.mm', corpus)

    print("Best LDA model, dictionary, and corpus saved.")

if __name__ == "__main__":
    multiprocessing.freeze_support()  # Ensures safe multiprocessing start
    main()
