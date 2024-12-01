import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your saved topic distributions
data = pd.read_excel('filtered_topics_output_sequential.xlsx')

# Define a dictionary mapping the renamed topics to descriptive names
topic_name_mapping = {
    'topic_1': 'General Fiction Analysis',
    'topic_2': 'Personal Reflection and Reader Experience',
    'topic_3': 'Fantasy and Adventure Elements',
    'topic_4': 'Blogging and Self-Expression',
    'topic_5': 'Classic Literature and Adventure',
    'topic_6': 'General Book Appeal',
    'topic_7': 'Crime and Mystery Genre',
    'topic_8': 'Romance and Relationships'
}

# Select only topic columns that match the new sequential naming
topic_columns = [f'topic_{i}' for i in range(1, 9)]

# Ensure these columns are numeric
data[topic_columns] = data[topic_columns].apply(pd.to_numeric, errors='coerce')

# Rename columns using the topic name mapping
data.rename(columns=topic_name_mapping, inplace=True)

# Group by genre and compute the mean topic weight for the named topics
named_topic_columns = list(topic_name_mapping.values())
genre_topic_weights = data.groupby('Genre')[named_topic_columns].mean()

# Plot heatmap with adjustments for label display
plt.figure(figsize=(14, 10))
sns.heatmap(genre_topic_weights, annot=True, cmap="YlGnBu", cbar=True, fmt=".2f")

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.title('Average Topic Weights per Genre')
plt.xlabel('Topics')
plt.ylabel('Genres')
plt.tight_layout()  # Ensures labels fit within the figure area
plt.show()
