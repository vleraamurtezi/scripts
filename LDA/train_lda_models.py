import pickle
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel

def main():
    # Load preprocessed data
    file_path = "/Users/vleramurtezi/Desktop/Thesis/Data/LDA/preprocessed_reviews.xlsx"
    data = pd.read_excel(file_path)

    # Create a dictionary and corpus from preprocessed reviews
    texts = data['preprocessed_review'].apply(eval).tolist()  # Ensure tokens are lists
    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # Save dictionary for future use
    with open('dictionary.pkl', 'wb') as f:
        pickle.dump(dictionary, f)

    # Range of topics to try
    topic_range = [5, 7, 10, 12, 15]

    # Track the best model and coherence score
    best_lda_model = None
    best_coherence = -1
    best_n_topics = 0

    # Train LDA models with different topic numbers
    for n_topics in topic_range:
        print(f"Training LDA model with {n_topics} topics...")
        lda_model = LdaModel(corpus=corpus, num_topics=n_topics, id2word=dictionary, passes=10, random_state=42)
        
        # Calculate coherence score
        coherence_model = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_score = coherence_model.get_coherence()
        print(f"Coherence Score for {n_topics} topics: {coherence_score}")
        
        # Update best model if coherence is better
        if coherence_score > best_coherence:
            best_lda_model = lda_model
            best_coherence = coherence_score
            best_n_topics = n_topics

    # Save the best model
    model_path = f'best_lda_model_{best_n_topics}_topics.model'
    best_lda_model.save(model_path)
    print(f"Best model saved with {best_n_topics} topics and a coherence score of: {best_coherence}")

if __name__ == '__main__':
    main()
