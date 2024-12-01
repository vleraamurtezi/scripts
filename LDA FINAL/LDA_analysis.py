import pandas as pd
import gensim
from gensim import corpora

# Load the original dataset, dictionary, and best LDA model
data_path = '/Users/vleramurtezi/Desktop/Thesis/Data/LDA/preprocessed_reviews.xlsx'
df = pd.read_excel(data_path)
dictionary = corpora.Dictionary.load('/Users/vleramurtezi/Desktop/Thesis/Data/LDA FINAL/dictionary.gensim')
corpus = gensim.corpora.MmCorpus('/Users/vleramurtezi/Desktop/Thesis/Data/LDA FINAL/corpus.mm')
best_model = gensim.models.LdaModel.load('/Users/vleramurtezi/Desktop/Thesis/Data/LDA FINAL/best_lda_model.gensim')

# Assign topic distributions for each review, excluding Topic 3
topic_distributions = []
for bow in corpus:
    # Get topic probabilities for each review
    topics = best_model.get_document_topics(bow, minimum_probability=0.0)
    
    # Extract probabilities for all topics, setting Topic 3 (index 2) to 0.0
    filtered_topics = []
    for idx, prob in enumerate(topics):
        if idx == 2:  # Skip Topic 3
            filtered_topics.append(0.0)
        else:
            filtered_topics.append(prob[1])  # Extract probability

    topic_distributions.append(filtered_topics)

# Create a DataFrame with the filtered topic distributions (excluding Topic 3)
# Topic 3 is still included but all values are set to 0.0 for consistency
topic_df = pd.DataFrame(topic_distributions, columns=[f'topic_{i+1}' for i in range(best_model.num_topics)])

# Verify the structure of topic_df to ensure no NaN values
topic_df.fillna(0.0, inplace=True)

# Determine the most prevalent topic for each review, excluding Topic 3 in calculations
topic_df['topic_categorical'] = topic_df.drop(columns='topic_3').idxmax(axis=1)

# Combine the original dataset with the filtered LDA topic results
final_df = pd.concat([df, topic_df], axis=1)

# Save the combined dataset to Excel
final_df.to_excel('/Users/vleramurtezi/Desktop/Thesis/Data/LDA FINAL/final_combined_dataset_excluding_topic3.xlsx', index=False)
print("Final combined dataset with LDA results (excluding Topic 3) created and saved.")

# Additional Step: Extract keywords and probabilities for each topic, excluding Topic 3
num_words = 10  # Number of top words to display per topic

# Display each topic with its top words
topic_keywords = []
for topic_id in range(best_model.num_topics):
    if topic_id == 2:  # Skip Topic 3
        continue
    words = best_model.show_topic(topic_id, topn=num_words)
    word_list = [word for word, prob in words]
    probability_list = [prob for word, prob in words]
    topic_keywords.append({
        "Topic": f"Topic {topic_id + 1}",
        "Words": word_list,
        "Probabilities": probability_list
    })
    print(f"Topic {topic_id + 1}: {word_list}")

# Convert the topic details to a DataFrame
topic_details_df = pd.DataFrame(topic_keywords)

# Save the topics and keywords (excluding Topic 3) to an Excel file
topic_details_df.to_excel('/Users/vleramurtezi/Desktop/Thesis/Data/LDA FINAL/topic_details_excluding_topic3.xlsx', index=False)
print("Topic details with keywords and probabilities (excluding Topic 3) saved.")