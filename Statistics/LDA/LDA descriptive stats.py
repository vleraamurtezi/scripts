import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = '/Users/vleramurtezi/Desktop/Thesis/Data/LDA/filtered_topics_output_sequential.xlsx'
df = pd.read_excel(file_path)

# Map the topic numbers to names
topic_mapping = {
    'topic_1': 'General Fiction Analysis',
    'topic_2': 'Personal Reflection and Reader Experience',
    'topic_3': 'Fantasy and Adventure Elements',
    'topic_4': 'Blogging and Self-Expression',
    'topic_5': 'Classic Literature and Adventure',
    'topic_6': 'General Book Appeal',
    'topic_7': 'Crime and Mystery Genre',
    'topic_8': 'Romance and Relationships'
}

# Apply the mapping to create a new column with named topics
df['Named_Topic'] = df['topic_categorical'].map(topic_mapping)

# Define the order of topics for consistent display
topic_order = [
    'General Fiction Analysis', 'Personal Reflection and Reader Experience',
    'Fantasy and Adventure Elements', 'Blogging and Self-Expression',
    'Classic Literature and Adventure', 'General Book Appeal',
    'Crime and Mystery Genre', 'Romance and Relationships'
]

# Visualization 1: Topic Frequency Distribution
topic_counts = df['Named_Topic'].value_counts().reindex(topic_order, fill_value=0)
plt.figure(figsize=(12, 6))
topic_counts.plot(kind='bar', color='skyblue', edgecolor='black', width=0.7)
plt.title("Topic Frequency Distribution")
plt.xlabel("Topic")
plt.ylabel("Frequency")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Visualization 2: Genre-Topic Distribution
genre_topic_counts = df.groupby(['Genre', 'Named_Topic']).size().unstack(fill_value=0).reindex(columns=topic_order, fill_value=0)
plt.figure(figsize=(14, 8))
genre_topic_counts.plot(kind='bar', stacked=True, figsize=(14, 8), color=plt.cm.Paired.colors)
plt.title("Genre-Topic Distribution")
plt.xlabel("Genre")
plt.ylabel("Frequency")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Topic", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
