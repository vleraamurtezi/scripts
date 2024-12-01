import pickle
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LdaModel

# Define paths to the saved model and preprocessed data
model_path = 'best_lda_model_15_topics.model'  # Update if necessary
file_path = '/Users/vleramurtezi/Desktop/Thesis/Data/LDA/preprocessed_reviews.xlsx'  # Replace with the actual path to your dataset

# Define topics to exclude and the mapping for renaming
irrelevant_topics = [2, 4, 6, 8, 9, 13, 14]
topic_rename_mapping = {0: 1, 1: 2, 3: 3, 5: 4, 7: 5, 10: 6, 11: 7, 12: 8}  # Map original topics to new sequential names

# Load the pre-trained LDA model
lda_model = LdaModel.load(model_path)

# Load the dataset containing all relevant metadata and preprocessed reviews
data = pd.read_excel(file_path)
texts = data['preprocessed_review'].apply(eval).tolist()  # Ensure tokens are lists
dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# Function to filter out irrelevant topics from each document's topic distribution
def filter_and_rename_topics(lda_model, corpus, exclude_topics, rename_mapping):
    # Get the topic distribution for each document
    all_topics = lda_model.get_document_topics(corpus)

    def remove_and_rename_topics(doc_topics):
        # Filter topics, rename based on mapping, and return as dictionary
        return {f"topic_{rename_mapping[topic]}": weight 
                for topic, weight in doc_topics if topic not in exclude_topics and topic in rename_mapping}

    # Apply filtering and renaming
    filtered_topics_per_doc = [remove_and_rename_topics(doc) for doc in all_topics]

    # Convert to DataFrame with each topic as a separate column, filling missing columns with 0
    filtered_df = pd.DataFrame(filtered_topics_per_doc).fillna(0)

    # Ensure columns are ordered as topic_1 to topic_8
    ordered_columns = [f"topic_{i}" for i in range(1, 9)]
    filtered_df = filtered_df.reindex(columns=ordered_columns, fill_value=0)

    # Determine the dominant topic per document, based on renamed topics
    filtered_df['topic_categorical'] = filtered_df.idxmax(axis=1)  # Finds the topic with the highest weight
    
    return filtered_df

# Apply the filtering function to get topic distributions with renamed topics
filtered_topic_df = filter_and_rename_topics(lda_model, corpus, irrelevant_topics, topic_rename_mapping)

# Merge the filtered topics back into the original data
# Selecting only essential columns from the original dataset
final_df = pd.concat([data[['Genre', 'Book', 'Author', 'review_text', 'Starts', 
                            'VADER_neg', 'VADER_neu', 'VADER_pos', 'VADER_compound',
                            'RoBERTa_neg', 'RoBERTa_neu', 'RoBERTa_pos', 
                            'vader_sentiment', 'roberta_sentiment', 'preprocessed_review']], 
                      filtered_topic_df], axis=1)

# Save the merged DataFrame to an Excel file
final_df.to_excel('filtered_topics_output_sequential.xlsx', index=False)
print("Filtered topics saved to 'filtered_topics_output_sequential.xlsx'")